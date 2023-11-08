
#include "src/CNN.h"
#include <iostream>

using namespace std;

//iterations per training session
int ITRS = 100;

//nodes in output layer
int const OUTPUTS = 5;

//total layers for convolution network
int CNN_LAYERS = 3;

//input image height
int INPUT_LAYER_ROWS = 25;

//input image width
int INPUT_LAYER_COLUMNS = 25;

//layer type
bool IS_POOL_LAYER[] = {false,true,false};

//channels per layer (3rd dimension of activation tensors)
//NOTE: channel count must be the same for pooling layers
int CHANNELS[] = {1,2,2,4}; 

int KERNEL_SIZE[] = {3,2,2};

int STRIDE_LEN[] = {1,1,1};

//total fully connected layers
int NN_LAYERS = 2;

//node counts for fully connected layers
int NODE_COUNT[] = {0,8,OUTPUTS};

double LEARN_RATE = 0.01;

double MOMENTUM = 0.9;

//parameter for leaky ReLU
double LEAK_COEF = 0.1;

Vector *yLayer;

Tensor *inputLayer;

//Training data generated from a seed. Data is meant for testing the network and proof-of-concept. 
void createTrainingData(int seed){
    
    for(int r = 0; r < inputLayer->rows; r++)
    for(int c = 0; c < inputLayer->columns; c++)

        inputLayer->self[0][r][c] = ((double)((r * inputLayer->columns + c) % ((seed + 1) * 2)) + (((double)(rand()) / (RAND_MAX + 1)) * (0.09 - 0.0001) + 0.0001)) / 10;

    yLayer->fill(0);
    yLayer->self[seed] = 1;

}


int main(){

    CNN* cnn = new CNN(CNN_LAYERS, INPUT_LAYER_ROWS, INPUT_LAYER_COLUMNS, IS_POOL_LAYER, CHANNELS, KERNEL_SIZE, STRIDE_LEN, NN_LAYERS, NODE_COUNT, LEARN_RATE, MOMENTUM, LEAK_COEF);

    int option, seed, trainAmt;
    
    double cost;

    srand( time(NULL) );

    inputLayer = cnn->featureMap[0];

    yLayer = cnn->nn->Y;

    //Simple command prompt interface
    while (true){
        
        cout<<"\nTest(0) Train(1) End(2): ";

        cin >> option;

        switch (option){

            //Test network
            case 0:

                cout << "\nSeed: ";

                cin >> seed;

                createTrainingData(seed);

                cnn->run();

                cout<<"\nOUTPUT LAYER";

                cnn->nn->printOutput();
                
                break;

            //Train network
            case 1:

                cout<<"\nTraining iterations: ";

                cin >> trainAmt;

                for(int i = 0; i < trainAmt; i++){

                    for(int k = 0; k < ITRS; k++){

                        createTrainingData(rand() % OUTPUTS);

                        cnn->train();

                    }

                    cnn->applyGradient(ITRS);

                }

                cout << "\nNetwork Cost: " << cnn->nn->cost <<"\n";

                break;
       
        }

    }

}

