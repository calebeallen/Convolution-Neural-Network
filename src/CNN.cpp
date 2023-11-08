
#include "CNN.h"
#include <iostream>

using namespace std;

KernelSet::KernelSet( int _length, int _strideLen, int _kernelSize, int _depth, int row, int col ){

    length = _length;

    strideLen = _strideLen;

    kernelSize = _kernelSize;

    depth = _depth;

    for(int i = 0; i < length; i++){

        kernel.push_back( new Tensor( depth, kernelSize, kernelSize ) );

        kGrad.push_back( new Tensor( depth, kernelSize, kernelSize ) );

        kGradM.push_back( new Tensor( depth, kernelSize, kernelSize ) );

        kernel[i]->randomize( -0.3, 0.3 );

    }

}

void KernelSet::convolve( Tensor* const input, Tensor* target){

    double dotProd;

    for(int l = 0; l < target->length; l++){

        for(int r = 0; r < target->rows; r++)
        for(int c = 0; c < target->columns; c++){

            dotProd = 0;

            for(int r1 = 0; r1 < kernelSize; r1++)
            for(int c1 = 0; c1 < kernelSize; c1++)
            for(int d = 0; d < depth; d++)

                dotProd += input->self[d][r * strideLen + r1][c * strideLen + c1] * kernel[l]->self[d][r1][c1];

            target->self[l][r][c] = dotProd;

        }

    }

}

void KernelSet::applyGradient( double scalar, double momentum ){

    for(int l = 0; l < length; l++){

        kGrad[l]->multiplyScalar(scalar);

        kGradM[l]->add( kGrad[l] );

        kernel[l]->add( kGradM[l] );

        kGradM[l]->multiplyScalar( momentum );

        kGrad[l]->fill(0);

    }

}


//Creates a fully customizable Convolution Neural Network.
CNN::CNN( int const _cnnLayers, int const inputLayerRows, int const inputLayerColumns, bool* const _isPoolLayer, int* const _channels, int* const _kernelSize, int* _strideLen, int const _nnLayers, int* _nodeCount, double const _learnRate, double const _momentum, double const _leakCoef ){

    cnnLayers = _cnnLayers;

    isPoolLayer = _isPoolLayer;

    learnRate = _learnRate;

    momentum = _momentum;

    leakCoef = _leakCoef;

    int fMapRows = inputLayerRows, fMapCols = inputLayerColumns, rPad = 0, cPad = 0;

    for(int l = 0; l < cnnLayers; l++){

        if(isPoolLayer[l])

            _strideLen[l] = _kernelSize[l];

        //calculate amount of padding needed
        rPad = (((fMapRows - _kernelSize[l]) / _strideLen[l] + 1) * _strideLen[l]) - (fMapRows - _kernelSize[l]);

        cPad = (((fMapCols - _kernelSize[l]) / _strideLen[l] + 1) * _strideLen[l]) - (fMapCols - _kernelSize[l]);

        //create feature map tensor for layer l
        featureMap.push_back( new Tensor( _channels[l], fMapRows, fMapCols, rPad, cPad ) );

        fMapGrad.push_back( new Tensor( _channels[l], fMapRows, fMapCols, rPad, cPad ) ); //maybe remove padding from gradient?

        if(isPoolLayer[l]){

            //store kernel size for pooling
            poolSize.push_back(_kernelSize[l]);
            
            //create pool "gradient" tensor. This will be the same size as the tensor that feeds into it and its only purpose is to store a 1 where a maximum value occurs.
            poolGrad.push_back( new Tensor( _channels[l], fMapRows, fMapCols, rPad, cPad ) );

            pLayers++;

        } else

            //create kernel set (weights between row l and l + 1) 
            kernelSet.push_back( new KernelSet( _channels[l + 1], _strideLen[l], _kernelSize[l], _channels[l], fMapRows, fMapCols) );


        //calculate feature map dimensions for next layer by dividing the difference between the kernel size and dimensions (with padding) of current layer by stride length.
        fMapRows = (fMapRows + rPad - _kernelSize[l]) / _strideLen[l] + 1;

        fMapCols = (fMapCols + cPad - _kernelSize[l]) / _strideLen[l] + 1;

        //if layer l is a convolution layer...
        if(!isPoolLayer[l]){

            //create bias tensor for layer l which is the same size as feature map tensor for layer l + 1;
            bias.push_back( new Tensor( _channels[l + 1], fMapRows, fMapCols ) );

            biasGrad.push_back( new Tensor( _channels[l + 1], fMapRows, fMapCols ) );

            biasGradM.push_back( new Tensor( _channels[l + 1], fMapRows, fMapCols ) );

            cLayers++;

        }

    }

    featureMap.push_back( new Tensor( _channels[cnnLayers], fMapRows, fMapCols ) );

    fMapGrad.push_back( new Tensor( _channels[cnnLayers], fMapRows, fMapCols ) );

    _nodeCount[0] = fMapRows * fMapCols * _channels[cnnLayers];

    //instantiate fully connected layers
    nn = new NN(_nnLayers, _nodeCount, learnRate, momentum, leakCoef);

    flatLayer = nn->A[0];

    flatLayerGrad = nn->aGrad[0];

}

void CNN::run(){

    for(int l = 0, c = 0, p = 0; l < cnnLayers; l++){

        if(!isPoolLayer[l]){

            kernelSet[c]->convolve( featureMap[l], featureMap[l + 1] );

            featureMap[l + 1]->add( bias[c] );

            featureMap[l + 1]->lReLU( leakCoef );

            c++;

        }else{

            featureMap[l + 1]->pool( featureMap[l], poolGrad[p], poolSize[p] );

            p++;

        }

    }

    //convert from 3D tensor to 1D vector
    flatLayer->flatten(featureMap[cnnLayers]);

    //forward propagation through fully connected layers
    nn->run();

}

void CNN::train(){

    Tensor* fmK, * fmGk, * fmJ, * fmGJ;

    KernelSet* kS;

    double DcDz;

    run();

    //back propagate through fully connected layer
    nn->train();

    //map vector back to tensor 
    flatLayerGrad->unFlatten(fMapGrad[cnnLayers]);

    flatLayerGrad->fill(0);

    //Back propagation
    for(int i = cnnLayers - 1, k = cLayers - 1, p = pLayers - 1; i >= 0; i--){

        fmK = featureMap[i];

        fmGk = fMapGrad[i];

        fmJ = featureMap[i + 1];

        fmGJ = fMapGrad[i + 1];

        //Convolution 
        if(!isPoolLayer[i]){

            kS = kernelSet[k];

            for(int l = 0; l < fmJ->length; l++)
            for(int r = 0; r < fmJ->rows; r++)
            for(int c = 0; c < fmJ->columns; c++){

                DcDz = fmGJ->self[l][r][c] * (fmJ->self[l][r][c] <= 0 ? leakCoef : 1);

                fmGJ->self[l][r][c] = 0;

                biasGrad[k]->self[l][r][c] += DcDz;

                for(int d = 0; d < kS->depth; d++)
                for(int r1 = 0; r1 < kS->kernelSize; r1++)
                for(int c1 = 0; c1 < kS->kernelSize; c1++){

                    kS->kGrad[l]->self[d][r1][c1] += fmK->self[d][r * kS->strideLen + r1][c * kS->strideLen + c1] * DcDz;

                    fmGk->self[d][r * kS->strideLen + r1][c * kS->strideLen + c1] += kS->kernel[l]->self[d][r1][c1] * DcDz;

                }

            }

            k--;

        //Pooling
        }else{

            for(int l = 0; l < fmJ->length; l++)
            for(int r = 0; r < fmJ->rows; r++)
            for(int c = 0; c < fmJ->columns; c++){

                DcDz = fmGJ->self[l][r][c];

                fmGJ->self[l][r][c] = 0;

                for(int r1 = 0; r1 < poolSize[p]; r1++)
                for(int c1 = 0; c1 < poolSize[p]; c1++)

                    fmGk->self[l][r * poolSize[p] + r1][c * poolSize[p] + c1] = poolGrad[p]->self[l][r * poolSize[p] + r1][c * poolSize[p] + c1] * DcDz;

            }

            p--;

        }

    }

}

void CNN::applyGradient(int itrs){

    double scalar = -learnRate / itrs;

    nn->applyGradient(itrs);

    for(int c = 0; c < cLayers; c++){

        kernelSet[c]->applyGradient( scalar, momentum );

        biasGrad[c]->multiplyScalar( scalar );

        biasGradM[c]->add( biasGrad[c] );

        bias[c]->add( biasGradM[c] );

        biasGradM[c]->multiplyScalar( momentum );

        biasGrad[c]->fill(0);

    }

}