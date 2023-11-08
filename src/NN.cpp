
#include "NN.h"
#include <iostream>

using namespace std;

NN::NN( int _layers, int* nodeCount, double _learnRate, double _momentum, double _leakCoef){

    layers = _layers;

    learnRate = _learnRate;

    momentum = _momentum;

    leakCoef = _leakCoef;

    //allocate mem
    A.push_back( new Vector(nodeCount[0]) );

    aGrad.push_back( new Vector(nodeCount[0]) );

    Y = new Vector( nodeCount[layers] );

    for(int i = 1; i <= layers; i++){

        A.push_back( new Vector(nodeCount[i]) );

        B.push_back( new Vector(nodeCount[i]) );

        W.push_back( new Matrix( nodeCount[i], nodeCount[i - 1] ) );

        aGrad.push_back( new Vector(nodeCount[i]) );

        bGrad.push_back( new Vector(nodeCount[i]) );
        
        bGradM.push_back( new Vector(nodeCount[i]) );

        wGrad.push_back( new Matrix( nodeCount[i], nodeCount[i - 1] ) );

        wGradM.push_back( new Matrix( nodeCount[i], nodeCount[i - 1] ) );

        W[i - 1]->randomize( -0.3, 0.3 );

    }
    
}

void NN::run(){

    for(int i = 0; i < layers; i++){

        A[i + 1]->matrixVectorProduct( W[i], A[i] );

        A[i + 1]->add(B[i]);

        A[i + 1]->lReLU(leakCoef);

    }

}

void NN::train(){

    double costArg, DcDz;

    //compute cost and deriv of cost with respect to final activation layer
    for(int j = 0; j < Y->length; j++){

        costArg = A[layers]->self[j] - Y->self[j];

        aGrad[layers]->self[j] = 2 * costArg;

        costSum += costArg * costArg;

    }

    //Back Propagation
    for(int l = layers - 1; l >= 0; l--){

        //for each activation in layer J
        for(int j = 0; j < A[l + 1]->length; j++){

            //compute ∂a/∂z 
            DcDz = (A[l + 1]->self[j] <= 0 ? leakCoef : 1) * aGrad[l + 1]->self[j];

            //clear ∂c/∂a val for layer l+1 activation J
            aGrad[l + 1]->self[j] = 0;

            for(int k = 0; k < A[l]->length; k++){

                //compute weight gradient by multiplying ∂a/∂z by activations in layer K
                wGrad[l]->self[j][k] += DcDz * A[l]->self[k];

                //compute activation gradient by taking a sum over layer J of ∂a/∂z * Wjk
                aGrad[l]->self[k] += DcDz * W[l]->self[j][k];

            }

            //compute bias gradient
            bGrad[l]->self[j] += DcDz;

        }

    }

}

void NN::applyGradient(int itrs){

    double scalar = -learnRate / itrs;

    for(int l = 0; l < layers; l++){

        //apply learning rate to gradient
        wGrad[l]->multiplyScalar( scalar );

        bGrad[l]->multiplyScalar( scalar );

        //add gradient term to previous gradient term
        wGradM[l]->add(wGrad[l]);

        bGradM[l]->add(bGrad[l]);

        //add negative gradient to weights and biases
        W[l]->add(wGradM[l]);

        B[l]->add(bGradM[l]);

        //multiply momentum scalar
        wGradM[l]->multiplyScalar( momentum );

        bGradM[l]->multiplyScalar( momentum );

        //reset gradients
        wGrad[l]->fill(0);

        bGrad[l]->fill(0);

    }
 
    //compute avg cost
    cost = costSum / itrs;

    costSum = 0;

}

void NN::printOutput(){

    cout<<"\n\n";

    for(int i = 0; i < A[layers]->length; i++){

        cout << i << " -> " << (int)(A[layers]->self[i] * 100) << "%  confident" << "\n";

    }

}
