// Fonksiyon 1: Standart üçlü döngü, satır-sütun mantığı
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