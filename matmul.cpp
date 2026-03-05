
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