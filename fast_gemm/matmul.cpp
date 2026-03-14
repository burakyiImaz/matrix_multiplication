#include "matmul.h"
#include <omp.h>

void FastMatmul::multiply(Matrix& A,Matrix& B,Matrix& C)
{

    int M = A.rows;
    int K = A.cols;
    int N = B.cols;

    const int MR = 4;
    const int NR = 8;

#pragma omp parallel for
    for(int i=0;i<M;i+=MR)
    {

        for(int j=0;j<N;j+=NR)
        {

            MicroKernel::kernel_4x8(
                K,
                &A.data[i*K],
                &B.data[j],
                &C.data[i*N+j],
                N
            );

        }

    }

}