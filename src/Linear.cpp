
#include "Linear.h"
#include <iostream>
#include <cmath>

using namespace std;

double leakyReLU( double x, double a ){

    return max( x * a, x );

}

Tensor::Tensor( int _length, int _rows, int _columns, int _rPad, int _cPad ){

    length = _length;

    rows = _rows;

    columns = _columns;

    rowsP = rows + _rPad;

    columnsP = columns + _cPad;

    self = new double**[length];

    for(int l = 0; l < length; l++){

        self[l] = new double*[rowsP];

        for(int r = 0; r < rowsP; r++){

            self[l][r] = new double[columnsP]; 

            for(int c = 0; c < columnsP; c++)

                self[l][r][c] = 0;

        }
    
    }

}

Tensor::Tensor( int _length, int _rows, int _columns ) : Tensor( _length, _rows, _columns, 0, 0 ) {}

void Tensor::fill(double val){

    for(int l = 0; l < length; l++)
    for(int r = 0; r < rows; r++)
    for(int c = 0; c < columns; c++)

        self[l][r][c] = val;

}

void Tensor::add( Tensor* const ten ){

    for(int l = 0; l < length; l++)
    for(int r = 0; r < rows; r++)
    for(int c = 0; c < columns; c++)

        self[l][r][c] += ten->self[l][r][c];
    
}

void Tensor::multiplyScalar( double scalar ){

    for(int l = 0; l < length; l++)
    for(int r = 0; r < rows; r++)
    for(int c = 0; c < columns; c++)

        self[l][r][c] *= scalar;

}

void Tensor::randomize( double max, double min ){

    for(int l = 0; l < length; l++)
    for(int r = 0; r < rows; r++)
    for(int c = 0; c < columns; c++)

        self[l][r][c] = (double)(rand()) / (RAND_MAX + 1) * (max - min) + min;

}

void Tensor::pool( Tensor* ten, Tensor* grad, int size ){

    int rMax, cMax, r2, c2;

    double max;

    for(int l = 0; l < length; l++)
    for(int r = 0; r < rows; r++)
    for(int c = 0; c < columns; c++){

        rMax = r * size;
        cMax = c * size;

        max = ten->self[l][rMax][cMax];

        for(int r1 = 0; r1 < size; r1++)
        for(int c1 = 1; c1 < size; c1++){

            r2 = r * size + r1;
            c2 = c * size + c1;

            grad->self[l][r2][c2] = 0;

            if(ten->self[l][r2][c2] > max){

                rMax = r2;
                cMax = c2;
                max = ten->self[l][r2][c2];

            }

        }

        grad->self[l][rMax][cMax] = 1;
        self[l][r][c] = max;

    }
    
}

void Tensor::lReLU( double a ){

    for(int l = 0; l < length; l++)
    for(int r = 0; r < rows; r++)
    for(int c = 0; c < columns; c++)

        self[l][r][c] = leakyReLU( self[l][r][c], a );
    
}

void Tensor::print(){

    for(int i = 0; i < length; i++){

        cout << "\n";

        for(int r = 0; r < rows; r++){

            for(int c = 0; c < columns; c++)

                cout << self[i][r][c] << " ";

            cout << "\n";

        }

    }

}




Matrix::Matrix( int _rows, int _columns){

    srand( time(NULL) );

    rows = _rows;
    columns = _columns;

    self = new double*[rows];

    for(int r = 0; r < rows; r++){

        self[r] = new double[columns];

        for(int c = 0; c < columns; c++)

            self[r][c] = 0;

    }

}

Matrix::~Matrix(){

    for (int r = 0; r < rows; r++) 

        delete[] self[r];

    delete[] self;

}

void  Matrix::addScalar( double scalar ){

    for(int r = 0; r < rows; r++)
    for(int c = 0; c < columns; c++)

        self[r][c] += scalar;

}

void  Matrix::multiplyScalar( double scalar ){

    for(int r = 0; r < rows; r++)
    for(int c = 0; c < columns; c++)

        self[r][c] *= scalar;

}

void  Matrix::add( Matrix const* mat ){

    for(int r = 0; r < rows; r++)
    for(int c = 0; c < columns; c++)

        self[r][c] += mat->self[r][c];
        
}

void  Matrix::fill( double val ){

    for(int r = 0; r < rows; r++)
    for(int c = 0; c < columns; c++)

        self[r][c] = val;
        
}

void  Matrix::copy( Matrix const* mat ){

    for(int r = 0; r < rows; r++)
    for(int c = 0; c < columns; c++)

        self[r][c] = mat->self[r][c];

}

void  Matrix::randomize( double min, double max ){

    for(int r = 0; r < rows; r++)
    for(int c = 0; c < columns; c++)

        self[r][c] = (double)(rand()) / (RAND_MAX + 1) * (max - min) + min;

}

void  Matrix::print(){

    cout << "\n";

    for(int r = 0; r < rows; r++){

        for(int c = 0; c < columns; c++)

            cout << self[r][c] << " ";

        cout << "\n";

    }

}


Vector::Vector( int _length ){

    length = _length;

    self = new double[length];

    fill(0);

};

Vector::~Vector(){

    delete[] self;

}

void Vector::addScalar( double scalar ){

    for(int i = 0; i < length; i++)

        self[i] += scalar;

}

void Vector::multiplyScalar( double scalar ){

    for(int i = 0; i < length; i++)

        self[i] *= scalar;

}

void Vector::add( Vector const* vect ){

    for(int i = 0; i < length; i++)

        self[i] += vect->self[i];

}

void Vector::matrixVectorProduct( Matrix const* mat, Vector const* vect ){

    double dp;

    for(int l1 = 0; l1 < length; l1++){

        dp = 0;

        for(int l2 = 0; l2 < vect->length; l2++)

            dp += vect->self[l2] * mat->self[l1][l2];

        self[l1] = dp;

    }

}

void Vector::fill( double val ){

    for(int i = 0; i < length; i++)

        self[i] = val;

}

void Vector::copy( Vector const* vect ){

    for(int i = 0; i < length; i++)

        self[i] = vect->self[i];

}

void Vector::randomize( double min, double max ){

    for(int i = 0; i < length; i++)

        self[i] = (double)(rand()) / (RAND_MAX + 1) * (max - min) + min;

}

void Vector::lReLU( int a ){

    for(int i = 0; i < length; i++)

        self[i] = leakyReLU( self[i], a );

}

void Vector::flatten( Tensor* const ten ){

    for(int l = 0; l < ten->length; l++)
    for(int r = 0; r < ten->rows; r++)
    for(int c = 0; c < ten->columns; c++){

        self[l * ten->rows * ten->columns + r * ten->columns + c] = ten->self[l][r][c];

    }

}

void Vector::unFlatten( Tensor* const ten ){

    for(int l = 0; l < ten->length; l++)
    for(int r = 0; r < ten->rows; r++)
    for(int c = 0; c < ten->columns; c++)

        ten->self[l][r][c] = self[l * ten->rows * ten->columns + r * ten->columns + c];

}

void Vector::print(){

    cout << "\n";

    for(int i = 0; i < length; i++)

        cout << self[i] << "\n";

}
