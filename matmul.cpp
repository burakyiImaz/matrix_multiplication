
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



void MatmulNaive_Tile(const Matrix<double>& A, const Matrix<double>& B, const Matrix<double>& C, int tile_size){
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