#include "converter.h"


int coo2csr(
  int       ** const col,       /*!< CSC row start indices */
  int       ** const row,       /*!< CSC column indices */
  int const *  const row_coo,   /*!< COO row indices */
  int const *  const col_coo,   /*!< COO column indices */
  int const         nnz,       /*!< Number of nonzero elements */
  int const         n,         /*!< Number of rows/columns */
  int const         isOneBased /*!< Whether COO is 0- or 1-based */
) {

  // ----- cannot assume that input is already 0!
    for(int l = 0; l < n+1; l++) (*col)[l] = 0;


  // ----- find the correct column sizes
    for(int l = 0; l < nnz; l++)
        (*col)[col_coo[l] - isOneBased]++;

  // ----- cumulative sum
    for(int i = 0, cumsum = 0; i < n; i++) {
        int temp = (*col)[i];
        (*col)[i] = cumsum;
        cumsum += temp;
    }
    (*col)[n] = nnz;
  // ----- copy the row indices to the correct place
    for(int l = 0; l < nnz; l++) {
        int col_l;
        col_l = col_coo[l] - isOneBased;

        int dst = (*col)[col_l];
        (*row)[dst] = row_coo[l] - isOneBased;

        (*col)[col_l]++;
    }
  // ----- revert the column pointers
    for(int i = 0, last = 0; i < n; i++) {
        int temp = (*col)[i];
        (*col)[i] = last;
        last = temp;
    }

    return n+1;

}

/* Reads a MMfile */
int cooReader(char* name,int** CSRrows, int** CSRcols){

    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int M, N, nz;   
    int i;
    double *val;


    if ((f = fopen( name, "r")) == NULL) 
        exit(1);
    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }


    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */

    if (mm_is_complex(matcode) && mm_is_matrix(matcode) && 
            mm_is_sparse(matcode))
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    /* find out size of sparse matrix .... */

    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0)
        exit(1);


    /* reseve memory for matrices */

    int *I = (int *) malloc(nz * sizeof(int));
    int *J = (int *) malloc(nz * sizeof(int));


    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
    char line[1024];
    for (int k = 0; k < nz; ++k) {
        if (!fgets(line, sizeof(line), f)) {
            fprintf(stderr, "Unexpected EOF\n");
            exit(1);
        }
        int i, j;
        sscanf(line, "%d %d", &i, &j);
        I[k] = i - 1;
        J[k] = j - 1;
    }

    if (f !=stdin) fclose(f);

    /************************/
    /* now write out matrix */
    /************************/

    //mm_write_banner(stdout, matcode);
    //mm_write_mtx_crd_size(stdout, M, N, nz);
   
	
  *CSRcols = (int *) malloc(nz * sizeof(int));
  *CSRrows = (int *) malloc((N + 1)   * sizeof(int));
  
  int CSC_rows = coo2csr(CSRrows,  CSRcols, I, J,nz, N,0);
  
  printf("Graph: %s\n", name);
	printf("nzz  : %d\nrows : %d\n\n",nz,CSC_rows);
  free(I);
  free(J);
  return CSC_rows;
}