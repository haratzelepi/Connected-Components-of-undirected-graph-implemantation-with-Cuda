# Connected Components on GPU with CUDA

This project implements a **label propagation** algorithm to compute the
**connected components** of large undirected graphs using **CUDA** on a single
NVIDIA GPU.

The graph is stored in **CSR (Compressed Sparse Row)** format, built from a
Matrix Market `.mtx` file via the provided `mmio` and `converter` code. The CUDA
implementation updates vertex labels in parallel until convergence and counts
the number of distinct labels as connected components.

---

## Files

- `cc_cuda.cu`  
  Main CUDA implementation. Contains:
  - host code that loads the graph (using `cooReader`),
  - device kernels for three parallel strategies,
  - timing and correctness check (number of connected components),
  - CSV logging of results.

- `converter.c`, `converter.h`  
  Utilities to convert from COO to CSR format and to read Matrix Market files.

- `mmio.c`, `mmio.h`  
  Matrix Market I/O utilities.

- `results.csv`  
  Created/extended at runtime. Stores the timings and configuration used.

---

## Parallel Strategies

`cc_cuda.cu` implements **three** different ways to map the work to CUDA threads:

1. **Thread per row** (`IMPLEMENTATION == 0`)  
   - Each thread is responsible for multiple rows (grid-stride loop).
   - Simple mapping, good baseline.

2. **Warp per row** (`IMPLEMENTATION == 1`)  
   - Each warp processes one CSR row at a time.
   - Threads in the warp cooperate using warp-level primitives to find the
     minimum label.

3. **Block per row** (`IMPLEMENTATION == 2`)  
   - Each block processes one CSR row at a time.
   - Threads in a block use shared memory to reduce to the minimum label.
   - Good for high-degree vertices.

You select the strategy in the code via:

```c
#define IMPLEMENTATION 1  // 0: thread_per_row, 1: warp_per_row, 2: block_per_row
```

The kernel launch configuration is controlled by:

```c
#define NUM_BLOCKS         1024    // number of CUDA blocks
#define THREADS_PER_BLOCK  512     // threads per block (multiple of 32 for warp-per-row)
```

## Environment (Google Colab)

The project is designed to run on **Google Colab** with a GPU runtime.

### Steps

1. Open a Colab notebook.
2. Change the runtime type to **GPU**  
   (`Runtime` → `Change runtime type` → `Hardware accelerator: GPU`).
3. For Cuda to run on google colab you must run :
   ```bash
    !apt-get update -qq
    !apt-get install -y -qq nvidia-cuda-toolkit
    !nvcc --version
    ```
4. Mount Google Drive:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')

5. Place your project under (for example):

    ```bash
    /content/drive/MyDrive/cuda_project/
    ```
This folder should contain:

```bash
cc_cuda.cu
converter.c
converter.h
mmio.c
mmio.h
com-Orkut.mtx           # and/or other .mtx graphs
```

## Compilation and Execution

After you have connected to your Drive and changed to the project folder, you
can compile and run like this :

```bash
%cd /content/drive/MyDrive/cuda_project/

#To compile
!nvcc -O3 -arch=sm_75 --use_fast_math cc_cuda.cu converter.o mmio.o -o test

# Run on a specific graph (example: com-Orkut)
!./test /content/drive/MyDrive/cuda_project/com-Orkut.mtx
``` 

You can also run on other graphs, for example:
```bash
!./test /content/drive/MyDrive/cuda_project/com-LiveJournal.mtx
!./test /content/drive/MyDrive/cuda_project/ca-CondMat.mtx
```

## Output

A typical run prints:

- the chosen implementation,
- the number of blocks and threads per block,
- the number of connected components,
- the total duration in seconds.

Example:

```text
Implementation: warp_per_row
Blocks used (manual): 1024 | Threads per block (manual): 512
Number of connected components: 1
Duration: 0.123456 seconds
```

It also appends a line to results.csv (creating the file with a header if
needed) with the following columns:

```text
graph,implementation,blocks,threads_per_block,numCC,duration
```

## Changing the Configuration

To explore performance:

1. Change the implementation in `cc_cuda.cu`:

   ```c
   #define IMPLEMENTATION 0  // or 1 or 2
   ```
2. Change `NUM_BLOCKS` (e.g. `64, 128, 256, 512, 1024, ...`) while keeping  
   `THREADS_PER_BLOCK` fixed (typically `512`, and always a multiple of `32`
   for the warp-per-row strategy).
3. Recompile and rerun. 
