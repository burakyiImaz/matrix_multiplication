
void MatmulNaive(const Matrix<double>&A, const Matrix<double>& B, const Matrix<double>&C){
    auto M= A.col();
    auto N = B.col();
    auto K = A.row();

    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            for(int k=0;k<K;k++){
                C(i,j)+= A(i,k)*B(k,j);
            }
        }
    }
    // Not: Bu sıralama özellikle column-major düzeni olan matrislerde memory access açısından daha cache-friendly olabilir
}

void MatmulNaive_Order(const Matrix<double>& A, const Matrix<double>& B, const Matrix<double>& C){
    auto M = A.row();
    auto N = B.col();
    auto K = A.col();

    for (int i=0; i<M; i++){        
        for (int k=0; k<K; k++){   
            for(int j=0; j<N; j++){ 
                C(i,j) += A(i,k) * B(k,j);
            }
        }
    }
    // Not: Bu sıralama özellikle row-major düzeni olan matrislerde  memory access açısından daha cache-friendly olabilir
}


template<int BLOCK>
void naive_block(const double *a, const double *mb, double *c, int N, int K){
    for(int i=0;i<BLOCK;i++, c+=N, a+=K){
        const double* b=mb;
        for(int k=0;k<BLOCK;k++, b+=N){
            for(int j=0;j<BLOCK;j++){
                c[j] += a[k] * b[j];
            }
        }
    }
}

oid MatmulNaive_Tile(const Matrix<double>& A, const Matrix<double>& B, const Matrix<double>& C, int tile_size){
    auto M = A.row();
    auto N = B.col();
    auto K = A.col();


    constexpr auto BLOCK=64; // Cache bloğu boyutu (örneğin 64KB)

    for(int ib=0;ib<M;ib+=BLOCK){
        for(int kb=0;kb<K;kb+=BLOCK){
            for(int jb=0;jb<N;jb+=BLOCK){
                const double* a= A(ib, kb);
                const double* mb= B(kb, jb);
                double* c= C(ib, jb);

                ikernel::naive_block<BLOCK>(a, mb, c, N, K);

        }
    }
}





template<int BLOCK>
void simd_block(const double *a, const double *mb, double *c, int N, int K){
    constexpr int simd_doubles= 128/(sizeof(double)*8); // 128-bit genişliğinde, kaç double sığar
    for(int i=0;i<BLOCK;i++, c+=N, a+=K){
        const double* b=mb;
        for(int k=0;k<BLOCK;k++, b+=N){
            __m128d a_reg = _mm_load_sd(&a[k]);
            a_reg         = _mm_unpacklo_pd(a_reg, a_reg); // a[k] değerini iki kere kopyalayarak 128-bit register'a yükle
            for(int j=0;j<BLOCK;j+=simd_doubles){ // SIMD genişliğine göre j'yi artır
                __m128d b_reg = _mm_load_pd(&b[j]);
                __m128d c_reg = _mm_load_pd(&c[j]);
                c_reg = _mm_add_pd(_mm_mul_pd(b_reg, a_reg), c_reg);
                _mm_store_pd(&c[j], c_reg);
            }
        }
    }
}



template<int BLOCK>
void avx_block(const double *a, const double *mb, double *c, int N, int K){
    constexpr int avx_doubles= 256/(sizeof(double)*8); // 256-bit genişliğinde, kaç double sığar
    for(int i=0;i<BLOCK;i++, c+=N, a+=K){
        const double* b=mb;
        for(int k=0;k<BLOCK;k++, b+=N){
            __m256d a_reg = _mm256_broadcast_sd(&a[k]); // a[k] değerini 256-bit register'a yükle ve tüm elemanlara kopyala
            for(int j=0;j<BLOCK;j+=avx_doubles){ // AVX genişliğine göre j'yi artır
                __m256d b_reg = _mm256_load_pd(&b[j]);
                __m256d c_reg = _mm256_load_pd(&c[j]);
                c_reg = _mm256_add_pd(_mm256_mul_pd(b_reg, a_reg), c_reg);
                _mm256_store_pd(&c[j], c_reg);
            }
        }
    }
}




void matMul_Avx_Cache(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C)
{
    auto M = A.row();
    auto K = A.col();
    auto N = B.col();
    constexpr auto Mc = 180, Nc = 96, Kc = 240, Nr = 12, Mr = 4;
    for(int ib = 0; ib < M; ib += Mc){
        for(int kb = 0; kb < K; kb += Kc){
            for(int jb = 0; jb < N; jb += Nc){
                const double* ma = &A(ib, kb);
                const double* mb = &B(kb, jb);
                double*       mc = &C(ib, jb);
                for(int i2 = 0; i2 < Mc; i2 += Mr){
                    for(int j2 = 0; j2 < Nc; j2 += Nr){
                        const double* a = &ma[i2 * K];
                        const double* b = &mb[j2];
                        double*       c = &mc[i2 * N + j2];
                        ikernels::avx_block<Nr, Mr, Kc, Nc>(c, a, b, N, K);
} } } }




template<int Nr, int Mr, int Kc, int Nc>
void avx_cache(const double* a,
                const double* mb,
                double* c,
                int N,
                int K)
{
    constexpr int avx_doubles = 256 / (sizeof(double) * 8); // Equals 4
    for(int i = 0; i < Mr; ++i, c += N, a += K){
        const double* b = mb;
        for(int k = 0; k < Kc; ++k, b += N){
            __m256d a_reg = _mm256_broadcast_sd(&a[k]);
            for(int j = 0; j < Nr; j += avx_doubles){
                __m256d b_reg = _mm256_loadu_pd(&b[j]);
                __m256d c_reg = _mm256_loadu_pd(&c[j]);
                c_reg = _mm256_fmadd_pd(a_reg, b_reg, c_reg);
                _mm256_storeu_pd(&c[j], c_reg);
} }
} }


template<int Nr, int Mr, int Kc, int Nc>
void avx_regs(const double* ma, const double* b, double* c, int N, int K) //CPU register aware implementation
{
    constexpr int CREG_CNT{Mr * Nr / 4};
    std::array<__m256d, CREG_CNT> res;
    for(int idx = 0; idx < CREG_CNT; ++idx)
        res[idx] = _mm256_setzero_pd();
    for(int k = 0; k < Kc; ++k, b += N){
        const double* a = ma;
        int idx = 0;
        for(int i = 0; i < Mr; ++i, a += K){
            __m256d areg = _mm256_broadcast_sd(&a[k]);
            for(int j = 0; j < Nr; j += 4, ++idx){
                res[idx] = _mm256_fmadd_pd(areg, _mm256_loadu_pd(&b[j]), res[idx]);
            }
} }
    int idx = 0;
    for(int i = 0; i < Mr; ++i, c += N){
        for(int j = 0; j < Nr; j += 4, ++idx){
            load_inc_store_double(&c[j], res[idx]);
        }
} }


template<int Nr, int Mr, int Kc, int Nc>
void avx_regs_bpack(const double* ma, const double* b, double* c, int N, int K)
{
    constexpr int CREG_CNT{Mr * Nr / 4};
    std::array<__m256d, CREG_CNT> res;
    for(int idx = 0; idx < CREG_CNT; ++idx)
        res[idx] = _mm256_setzero_pd();
    for(int k = 0; k < Kc; ++k, b += Nc){
        const double* a = ma;
        int idx = 0;
        for(int i = 0; i < Mr; ++i, a += K){
            __m256d areg = _mm256_broadcast_sd(&a[k]);
            for(int j = 0; j < Nr; j += 4, ++idx){
                res[idx] = _mm256_fmadd_pd(areg, _mm256_loadu_pd(&b[j]), res[idx]);
            }
} }
    int idx = 0;
    for(int i = 0; i < Mr; ++i, c += N){
        for(int j = 0; j < Nr; j += 4, ++idx){
            load_inc_store_double(&c[j], res[idx]);
        }
} }


template<int Nr, int Mr, int Kc>
void avx_regs_reordered(const double* a, const double* b, double* c, int N)// Assuming b is already packed in column-major order
{
    constexpr int CREG_CNT{Mr * Nr / 4};
    std::array<__m256d, CREG_CNT> res;
    for(int idx = 0; idx < CREG_CNT; ++idx)
        res[idx] = _mm256_setzero_pd();
    for(int k = 0; k < Kc; ++k, b += Nr, a += Mr){
        int idx = 0;
        for(int i = 0; i < Mr; ++i){
            __m256d areg = _mm256_broadcast_sd(&a[i]);
            for(int j = 0; j < Nr; j += 4, ++idx){
                res[idx] = _mm256_fmadd_pd(areg, _mm256_loadu_pd(&b[j]), res[idx]);
            }
} }
    int idx = 0;
    for(int i = 0; i < Mr; ++i, c += N){
        for(int j = 0; j < Nr; j += 4, ++idx){
            load_inc_store_double(&c[j], res[idx]);
        }
} }


int main(){
    int M =2, N=4, K=2;
    Matrix<double> A(M, K);
    Matrix<double> B(K, N);
    Matrix<double> C(M, N);

    for(int i=0;i<M;i++){
        for(int j=0;j<K;j++){
            A(i,j) = i+j+1; 
        }
    }
    for(int j=0;j<K;j++){
        for(int k=0;k<N;k++){
            B(j,k) = j+k+1; 
        }
    }
    MatmulNaive(A, B, C);

    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            std::cout << C(i,j)<< " ";  
        }
        std::cout << std::endl;
    }

}