// Naive (en basit) matris çarpımı
void MatmulNaive(const Matrix<double>& A, const Matrix<double>& B, const Matrix<double>& C){
    auto M = A.col();  /* 
    """M, çarpılacak A matrisinin satır sayısını temsil eder.
    Bu genellikle C++ matris kütüphanelerinde row-major mı yoksa column-major mı olduğuna göre değişir. 
    Burada yanlışlıkla col() alınmış, normalde row() olmalı."""
    */
    auto N = B.col();  
    /* 
    """N, B matrisinin sütun sayısıdır. Sonuç matrisinin sütun sayısını belirler."""
    */
    auto K = A.row();  
    /* 
    """K, A matrisinin sütun sayısıdır. İç döngü çarpımı için kullanılır."""
    */

    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            for(int k=0;k<K;k++){
                C(i,j) += A(i,k) * B(k,j); 
                /*
                """Her C(i,j) elemanı için, A'nın i. satırı ile B'nin j. sütunu çarpılır ve toplanır.
                Bu en klasik üçlü döngü matris çarpımıdır ve her eleman bağımsız olarak hesaplanır.
                Avantaj: Basit ve anlaşılır.
                Dezavantaj: Büyük matrislerde cache performansı düşük olabilir."""
                */
            }
        }
    }
    /* 
    """Not: Bu döngü sırası column-major matrislerde hafıza erişimlerini sıralı yapacağı için cache dostudur.
    Column-major dizilerde aynı sütunun elemanları ardışık hafızadadır."""
    */
}

// Row-major dostu naive matris çarpımı
void MatmulNaive_Order(const Matrix<double>& A, const Matrix<double>& B, const Matrix<double>& C){
    auto M = A.col();
    auto N = B.col();
    auto K = A.row(); 


    for (int i=0; i<M; i++){        
        for (int k=0; k<K; k++){   
            for(int j=0; j<N; j++){ 
                C(i,j) += A(i,k) * B(k,j);
                /*
                """i-k-j döngü sırası, özellikle row-major (C++ standart) matrisler için memory erişimini daha cache-friendly yapar.
                Her satırdaki elemanlar ardışık olduğu için CPU cache daha verimli kullanılır."""
                */
            }
        }
    }
}

// Basit block (tile) matris çarpımı
template<int BLOCK>
void naive_block(const double *a, const double *mb, double *c, int N, int K){
    for(int i=0;i<BLOCK;i++, c+=N, a+=K){
        const double* b=mb;
        for(int k=0;k<BLOCK;k++, b+=N){
            for(int j=0;j<BLOCK;j++){
                c[j] += a[k] * b[j];
                /*
                """Bu fonksiyon bir BLOCK x BLOCK alt matris çarpımı yapar.
                Amaç: Büyük matrisleri küçük bloklara ayırarak cache dostu hale getirmek.
                i,k,j sıralaması ile satır ve sütunlar ardışık erişilir, cache miss azaltılır.
                Buradaki c+=N ve a+=K pointer artışı ile bir sonraki satıra geçilir."""
                */
            }
        }
    }
}

// Tile/block naive matris çarpımı wrapper
void MatmulNaive_Tile(const Matrix<double>& A, const Matrix<double>& B, const Matrix<double>& C, int tile_size){
    auto M = A.row();
    auto N = B.col();
    auto K = A.row();

    constexpr auto BLOCK=64; /* 
    """64x64 boyutunda blok kullanıyoruz.
    Cache boyutuna göre seçilmiş, L1 veya L2 cache'de rahatça sığacak boyut."""
    */

    for(int ib=0; ib<M; ib+=BLOCK){
        for(int kb=0; kb<K; kb+=BLOCK){
            for(int jb=0; jb<N; jb+=BLOCK){
                const double* a= A(ib, kb);
                const double* mb= B(kb, jb);
                double* c= C(ib, jb);

                ikernel::naive_block<BLOCK>(a, mb, c, N, K);
                /*
                """Blok bazlı çarpım ile matrisin her alt bloğu cache’e alınarak çarpılır.
                Büyük matrislerde global bellek yerine cache kullanımı performansı dramatik artırır."""
                */
            }
        }
    }
}

// SSE (128-bit SIMD) ile block çarpımı
template<int BLOCK>
void simd_block(const double *a, const double *mb, double *c, int N, int K){
    constexpr int simd_doubles=128/(sizeof(double)*8); 
    /* 
    """128-bit register kullanıyoruz, double 64-bit olduğundan 2 double aynı anda işlenir.
    SIMD (Single Instruction Multiple Data) ile paralel çarpım yapılır."""
    */
    for(int i=0;i<BLOCK;i++, c+=N, a+=K){
        const double* b=mb;
        for(int k=0;k<BLOCK;k++, b+=N){
            __m128d a_reg = _mm_load_sd(&a[k]);
            a_reg         = _mm_unpacklo_pd(a_reg, a_reg);
            for(int j=0;j<BLOCK;j+=simd_doubles){
                __m128d b_reg = _mm_load_pd(&b[j]);
                __m128d c_reg = _mm_load_pd(&c[j]);
                c_reg = _mm_add_pd(_mm_mul_pd(b_reg, a_reg), c_reg);
                _mm_store_pd(&c[j], c_reg);
                /*
                """SSE register ile 2 double aynı anda çarpılır ve toplanır.
                Bu işlem CPU’nun SIMD birimini kullanarak performansı iki kat artırır."""
                */
            }
        }
    }
}

// AVX (256-bit) ile block çarpımı
template<int BLOCK>
void avx_block(const double *a, const double *mb, double *c, int N, int K){
    constexpr int avx_doubles=256/(sizeof(double)*8); // 4 double aynı anda
    for(int i=0;i<BLOCK;i++, c+=N, a+=K){
        const double* b=mb;
        for(int k=0;k<BLOCK;k++, b+=N){
            __m256d a_reg = _mm256_broadcast_sd(&a[k]); 
            for(int j=0;j<BLOCK;j+=avx_doubles){
                __m256d b_reg = _mm256_load_pd(&b[j]);
                __m256d c_reg = _mm256_load_pd(&c[j]);
                c_reg = _mm256_add_pd(_mm256_mul_pd(b_reg, a_reg), c_reg);
                _mm256_store_pd(&c[j], c_reg);
                /*
                """AVX ile 4 double paralel çarpılır, register ve cache kullanımını optimize eder.
                Büyük matrislerde performans SSE’ye göre 2 kat artar."""
                */
            }
        }
    }
}

// Cache ve AVX ile optimize edilmiş matris çarpımı
void matMul_Avx_Cache(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C){
    auto M = A.row();
    auto K = A.col();
    auto N = B.col();
    constexpr auto Mc=180, Nc=96, Kc=240, Nr=12, Mr=4; 
    /* 
    """Blok boyutları L1/L2 cache boyutuna göre optimize edilmiştir.
    Mr, Nr: register tiling, Mc, Nc, Kc: cache tiling."""
    */
    for(int ib=0; ib<M; ib+=Mc){
        for(int kb=0; kb<K; kb+=Kc){
            for(int jb=0; jb<N; jb+=Nc){
                const double* ma = &A(ib, kb);
                const double* mb = &B(kb, jb);
                double* mc = &C(ib, jb);
                for(int i2=0; i2<Mc; i2+=Mr){
                    for(int j2=0; j2<Nc; j2+=Nr){
                        const double* a = &ma[i2*K];
                        const double* b = &mb[j2];
                        double* c = &mc[i2*N+j2];
                        ikernels::avx_block<Nr, Mr, Kc, Nc>(c, a, b, N, K);
                        /*
                        """Hem cache hem de CPU register optimize edilmiş çarpım.
                        Büyük matrislerde tek blok yerine, cache ve register bazlı küçük bloklarla işlem yapar.
                        Sonuç: L1/L2 cache miss azalır, CPU pipeline verimli çalışır, SIMD paralellik kullanılır."""
                        */
                    }
                }
            }
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