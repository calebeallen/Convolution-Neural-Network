
#ifndef linear
#define linear

class Tensor {

    public:

        double*** self;

        int length, rows, columns, rowsP, columnsP;

        Tensor( int, int, int, int, int );

        Tensor( int, int, int );

        void fill( double );

        void add( Tensor* const );

        void multiplyScalar( double );

        void randomize( double, double );

        void pool( Tensor*, Tensor*, int );

        void lReLU( double );

        void print();
        
};

class Matrix {

    public:

        double** self;

        int rows, columns;

        Matrix( int, int );

        ~Matrix();

        void addScalar( double );

        void multiplyScalar( double );

        void add( Matrix const* );

        void fill( double );

        void copy( Matrix const* );

        void randomize( double, double );

        void print();

};

class Vector {

    public:

        double* self;

        int length;

        Vector( int );

        ~Vector();

        void addScalar( double );

        void multiplyScalar( double );

        void add( Vector const* );

        void matrixVectorProduct( Matrix const*, Vector const* );

        void fill( double );

        void copy( Vector const* );

        void randomize( double, double );

        void sigmoid();

        void ReLU();

        void lReLU( int );

        void flatten( Tensor* const );

        void unFlatten( Tensor* const );

        void print();
};

#endif