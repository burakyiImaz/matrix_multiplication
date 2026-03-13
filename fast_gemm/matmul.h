#pragma once
#include "matrix.h"
#include "gemm_kernel.h"

class FastMatmul
{

public:

static void multiply(Matrix& A,Matrix& B,Matrix& C);

};
