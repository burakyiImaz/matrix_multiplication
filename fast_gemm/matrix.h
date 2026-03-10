#pragma once 
#include <vector>



class Matrix{
    int rows, cols;
    std::vector<double> data;
public:
    int rows, cols;
    std::vector<double> data;
    Matrix(int r,int c){
        rows = r;
        cols = c;
        data.resize(r*c);
    }
    
    inline double& operator()(int i, int j){
        return data[i*cols + j];
    }
    inline const double& operator()(int i, int j) const {
        return data[i*cols + j];
    }
    inline double *ptr(){
        return data.data();
    }
};