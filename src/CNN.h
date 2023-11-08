
#include "Linear.h"
#include "NN.h"
#include <vector>

class KernelSet {

    public:

        std::vector<Tensor*> kernel, kGrad, kGradM;

        int length, strideLen, kernelSize, depth;

        KernelSet( int, int, int, int, int, int );

        void convolve( Tensor* const, Tensor* );

        void applyGradient( double, double );

};

class CNN {

    public:

        std::vector<Tensor*> featureMap, fMapGrad, poolGrad, bias, biasGrad, biasGradM;

        std::vector<KernelSet*> kernelSet;

        std::vector<int> poolSize; 

        Vector* flatLayer, * flatLayerGrad;

        NN* nn;

        int cnnLayers, cLayers = 0, pLayers = 0;
        
        double learnRate, momentum, leakCoef;

        bool* isPoolLayer;

        //Creates a fully customizable Convolution Neural Network.
        CNN( int const, int const, int const, bool* const, int* const, int* const, int*, int const, int*, double const, double const, double const );

        void run();

        void train();

        void applyGradient( int );

};