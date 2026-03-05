
void MatmulNaive(const Matrix<double>& A, const Matrix<double>& B, const Matrix<double>& C){
    auto M = A.row();
    auto N = B.col(); 
    auto K = A.col(); 

    for (int i=0; i<M; i++){        
        for (int j=0; j<N; j++){   
            for(int k=0; k<K; k++){ 
                C(i,j) += A(i,k) * B(k,j);
            }
        }
    }
    // Not: Bu sırayla döngü cache hit’i sütun bazlı olduğundan CPU cache açısından daha az verimli olabilir
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

void MatmulNaive_Tile(const Matrix<double>& A, const Matrix<double>& B, const Matrix<double>& C, int tile_size){
    auto M = A.row();
    auto N = B.col();
    auto K = A.col();


    constexpr auto BLOCK = 64; // Cache line boyutuna uygun bir tile size seçilebilir
    for(int ib=0; ib<M; ib+=BLOCK){
        for(int kb=0; kb<K; kb+=BLOCK){
            for(int jb=0; jb<N; jb+=BLOCK){
                // Tile boyutlarına göre alt matrislerin sınırlarını belirledim
                int i_max = std::min(ib + BLOCK, M);
                int k_max = std::min(kb + BLOCK, K);
                int j_max = std::min(jb + BLOCK, N);

                // Alt matrisler üzerinde çarpma işlemi
                for (int i=ib; i<i_max; i++){        
                    for (int k=kb; k<k_max; k++){   
                        for(int j=jb; j<j_max; j++){ 
                            C(i,j) += A(i,k) * B(k,j);
                        }
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