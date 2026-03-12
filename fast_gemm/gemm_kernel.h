#pragma once
#include <immintrin.h>

class MicroKernel
{
public:

static void kernel_4x8(
    int K,
    const double* A,
    const double* B,
    double* C,
    int ldc
)
{

    __m256d c00 = _mm256_setzero_pd();
    __m256d c01 = _mm256_setzero_pd();

    __m256d c10 = _mm256_setzero_pd();
    __m256d c11 = _mm256_setzero_pd();

    __m256d c20 = _mm256_setzero_pd();
    __m256d c21 = _mm256_setzero_pd();

    __m256d c30 = _mm256_setzero_pd();
    __m256d c31 = _mm256_setzero_pd();


    for(int k=0;k<K;k++)
    {

        __m256d b0 = _mm256_loadu_pd(&B[k*8]);
        __m256d b1 = _mm256_loadu_pd(&B[k*8+4]);

        __m256d a0 = _mm256_broadcast_sd(&A[0*K + k]);
        __m256d a1 = _mm256_broadcast_sd(&A[1*K + k]);
        __m256d a2 = _mm256_broadcast_sd(&A[2*K + k]);
        __m256d a3 = _mm256_broadcast_sd(&A[3*K + k]);


        c00 = _mm256_fmadd_pd(a0,b0,c00);
        c01 = _mm256_fmadd_pd(a0,b1,c01);

        c10 = _mm256_fmadd_pd(a1,b0,c10);
        c11 = _mm256_fmadd_pd(a1,b1,c11);

        c20 = _mm256_fmadd_pd(a2,b0,c20);
        c21 = _mm256_fmadd_pd(a2,b1,c21);

        c30 = _mm256_fmadd_pd(a3,b0,c30);
        c31 = _mm256_fmadd_pd(a3,b1,c31);

    }


    _mm256_storeu_pd(&C[0*ldc],c00);
    _mm256_storeu_pd(&C[0*ldc+4],c01);

    _mm256_storeu_pd(&C[1*ldc],c10);
    _mm256_storeu_pd(&C[1*ldc+4],c11);

    _mm256_storeu_pd(&C[2*ldc],c20);
    _mm256_storeu_pd(&C[2*ldc+4],c21);

    _mm256_storeu_pd(&C[3*ldc],c30);
    _mm256_storeu_pd(&C[3*ldc+4],c31);

}

};