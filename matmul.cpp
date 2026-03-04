

void MatmulNaive(const Matrix<double>& A, const Matrix<double>& B, const Matrix<double>& C){
    auto M= A.row(); // with auto, automatically deduce the type of M
    auto N= B.col();
    auto K= A.col();
    for (int i=0; i<M; i++){
        for (int j=0;j<N;j++){
            for(int k=0;k<K;k++){
                C(i,j)+=A(i,k)*B(k,j);
            }
        }
    }

}
