#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mmio.h"
#include "converter.h"
#include <cuda_runtime.h>

// ============================================================
// Choose implementation:
// 0 = Thread per row
// 1 = Warp per row
// 2 = Block per row
// ============================================================
#define IMPLEMENTATION 2

// You will sweep BOTH manually for all implementations
#define NUM_BLOCKS (1024*1024)

// For warp-per-row: MUST be multiple of 32
#define THREADS_PER_BLOCK 512

#if (IMPLEMENTATION == 0)
  #define IMPL_NAME "thread_per_row"
#elif (IMPLEMENTATION == 1)
  #define IMPL_NAME "warp_per_row"
#elif (IMPLEMENTATION == 2)
  #define IMPL_NAME "block_per_row"
#else
  #error "Invalid IMPLEMENTATION"
#endif

// ------------------------------------------------------------
// CUDA error checking (minimal)
// ------------------------------------------------------------
#define CUDA_CHECK(call) do { \
  cudaError_t err = (call); \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(1); \
  } \
} while (0)

// ============================================================
// 0) THREAD PER ROW (grid-stride over nodes)
// ============================================================
__global__ void coloring_step_thread_per_row(const int *indexes, const int *indices,
                                             int *colors, int N, int *changed)
{
    int tid    = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < N; i += stride) {
        for (int j = indexes[i]; j < indexes[i + 1]; ++j) {
            int v = indices[j];

            int cmin = (colors[i] < colors[v]) ? colors[i] : colors[v];

            int old = atomicMin(&colors[i], cmin);
            if (old > cmin) atomicExch(changed, 1);

            old = atomicMin(&colors[v], cmin);
            if (old > cmin) atomicExch(changed, 1);
        }
    }
}

// ============================================================
// 1) WARP PER ROW (manual NUM_BLOCKS + grid-stride over warps)
// ============================================================
__inline__ __device__ int warpReduceMin(int val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = min(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__global__ void coloring_step_warp_per_row(const int *indexes, const int *indices,
                                           int *colors, int N, int *changed)
{
    // blockDim MUST be multiple of 32
    int globalThread = blockIdx.x * blockDim.x + threadIdx.x;
    int warpIdGlobal = globalThread >> 5;        // /32
    int lane         = threadIdx.x & 31;         // %32

    int totalWarps = (gridDim.x * blockDim.x) >> 5; // total threads / 32
    if (totalWarps == 0) return;

    // Grid-stride over warps so NUM_BLOCKS is fully manual
    for (int u = warpIdGlobal; u < N; u += totalWarps) {
        int start = indexes[u];
        int end   = indexes[u + 1];

        int localMin = colors[u];

        for (int jj = start + lane; jj < end; jj += 32) {
            int v  = indices[jj];
            int cv = colors[v];
            localMin = min(localMin, cv);
        }

        int minLabel = warpReduceMin(localMin);

        if (lane == 0) {
            int old = atomicMin(&colors[u], minLabel);
            if (old > minLabel) atomicExch(changed, 1);
        }

        for (int jj = start + lane; jj < end; jj += 32) {
            int v = indices[jj];
            int old = atomicMin(&colors[v], minLabel);
            if (old > minLabel) atomicExch(changed, 1);
        }
    }
}

// ============================================================
// 2) BLOCK PER ROW (manual NUM_BLOCKS + grid-stride over blocks)
// Each block processes one row at a time using all its threads,
// then jumps to the next row: u += gridDim.x
// ============================================================
__global__ void coloring_step_block_per_row(const int *indexes, const int *indices,
                                            int *colors, int N, int *changed)
{
    int tid = threadIdx.x;

    // shared min-reduction
    extern __shared__ int smin[];

    // Grid-stride over blocks so NUM_BLOCKS is fully manual
    for (int u = blockIdx.x; u < N; u += gridDim.x) {

        int start = indexes[u];
        int end   = indexes[u + 1];

        int localMin = colors[u];

        for (int jj = start + tid; jj < end; jj += blockDim.x) {
            int v  = indices[jj];
            int cv = colors[v];
            localMin = min(localMin, cv);
        }

        smin[tid] = localMin;
        __syncthreads();

        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (tid < offset) smin[tid] = min(smin[tid], smin[tid + offset]);
            __syncthreads();
        }

        int minLabel = smin[0];

        if (tid == 0) {
            int old = atomicMin(&colors[u], minLabel);
            if (old > minLabel) atomicExch(changed, 1);
        }
        __syncthreads();

        for (int jj = start + tid; jj < end; jj += blockDim.x) {
            int v = indices[jj];
            int old = atomicMin(&colors[v], minLabel);
            if (old > minLabel) atomicExch(changed, 1);
        }

        // Important: keep blocks in sync before next u iteration
        __syncthreads();
    }
}

// ============================================================
// Timing
// ============================================================
struct timespec t_start, t_end;

int main(int argc, char *argv[])
{
    if (argc < 2) {
        printf("Usage: %s <graph_file>\n", argv[0]);
        return 1;
    }
    char *str = argv[1];

#if (IMPLEMENTATION == 1)
    if ((THREADS_PER_BLOCK % 32) != 0) {
        printf("Error: THREADS_PER_BLOCK must be multiple of 32 for warp_per_row.\n");
        return 1;
    }
#endif

    int *indexes;
    int *indices;

    int N = cooReader(str, &indexes, &indices) - 1;
    if (N <= 0) {
        printf("Error: N <= 0\n");
        return 1;
    }

    int *colors = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) colors[i] = i;

    int nnz = indexes[N];

    int *d_indexes, *d_indices, *d_colors;
    int *d_changed;

    CUDA_CHECK(cudaMalloc((void**)&d_indexes, (N + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_indices, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_colors,  N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_changed, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_indexes, indexes, (N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_indices, indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colors,  colors,  N * sizeof(int),    cudaMemcpyHostToDevice));

    int blocks_used = NUM_BLOCKS;          // ALWAYS manual, for all implementations
    int tpb_used    = THREADS_PER_BLOCK;

    clock_gettime(CLOCK_REALTIME, &t_start);

    int h_changed;
    do {
        h_changed = 0;
        CUDA_CHECK(cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice));

#if (IMPLEMENTATION == 0)
        coloring_step_thread_per_row<<<blocks_used, tpb_used>>>(d_indexes, d_indices, d_colors, N, d_changed);

#elif (IMPLEMENTATION == 1)
        coloring_step_warp_per_row<<<blocks_used, tpb_used>>>(d_indexes, d_indices, d_colors, N, d_changed);

#elif (IMPLEMENTATION == 2)
        size_t shmem = (size_t)tpb_used * sizeof(int);
        coloring_step_block_per_row<<<blocks_used, tpb_used, shmem>>>(d_indexes, d_indices, d_colors, N, d_changed);
#endif

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));

    } while (h_changed);

    clock_gettime(CLOCK_REALTIME, &t_end);

    CUDA_CHECK(cudaMemcpy(colors, d_colors, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Count unique labels as connected components
    int numCC = 0;
    int *uniqueFlags = (int*)calloc(N, sizeof(int));

    for (int i = 0; i < N; i++) {
        int color = colors[i];
        if (color >= 0 && color < N) {
            if (!uniqueFlags[color]) {
                uniqueFlags[color] = 1;
                numCC++;
            }
        }
    }

    double duration = ((t_end.tv_sec - t_start.tv_sec) * 1000000
                      + (t_end.tv_nsec - t_start.tv_nsec) / 1000) / 1000000.0;

    printf("Implementation: %s\n", IMPL_NAME);
    printf("Blocks used (manual): %d | Threads per block (manual): %d\n", blocks_used, tpb_used);
    printf("Number of connected components: %d\n", numCC);
    printf("Duration: %lf seconds\n", duration);

    // results.csv columns:
    // graph,implementation,blocks,threads_per_block,numCC,duration
  // Write results.csv with header if file is empty or does not exist
FILE *fcheck = fopen("results.csv", "r");
int write_header = 0;

if (fcheck == NULL) {
    // file does not exist
    write_header = 1;
} else {
    fseek(fcheck, 0, SEEK_END);
    if (ftell(fcheck) == 0) {
        // file exists but is empty
        write_header = 1;
    }
    fclose(fcheck);
}

FILE *f = fopen("results.csv", "a");
if (f != NULL) {
    if (write_header) {
        fprintf(f, "graph,implementation,blocks,threads_per_block,numCC,duration\n");
    }

    fprintf(f, "%s,%s,%d,%d,%d,%lf\n",
            str,
            IMPL_NAME,
            blocks_used,
            tpb_used,
            numCC,
            duration);

    fclose(f);
}


    free(colors);
    free(uniqueFlags);

    CUDA_CHECK(cudaFree(d_indexes));
    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_colors));
    CUDA_CHECK(cudaFree(d_changed));

    return 0;
}
