
#include <vector>
#include "Linear.h"

class NN {

    public:

        std::vector<Vector*> A, aGrad, B, bGrad, bGradM;

        std::vector<Matrix*> W, wGrad, wGradM;

        Vector* Y;

        int layers;

        double costSum, learnRate, momentum, cost = 0, leakCoef;

        NN( int, int*, double, double, double );

        void run();

        void train();

        void applyGradient( int );

        void printOutput();

};