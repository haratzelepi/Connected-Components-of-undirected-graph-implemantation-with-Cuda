
#include <stdlib.h>
#include <stdio.h>
#include "mmio.h"


#ifndef READERCONVERTER_H
#define READERCONVERTER_H


int coo2csr(
  int       ** const col,       /*!< CSC row start indices */
  int       ** const row,       /*!< CSC column indices */
  int const *  const row_coo,   /*!< COO row indices */
  int const *  const col_coo,   /*!< COO column indices */
  int const         nnz,       /*!< Number of nonzero elements */
  int const         n,         /*!< Number of rows/columns */
  int const         isOneBased /*!< Whether COO is 0- or 1-based */
);

/* Reads a MMfile */
int cooReader(char* name, int **CSRrows, int **CSRcols);

#endif