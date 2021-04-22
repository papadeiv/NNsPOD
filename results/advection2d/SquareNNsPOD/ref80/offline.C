#include "neuralAdvection.H"
#include "ITHACAstream.H"
#include "ITHACAPOD.H"
#include "forces.H"
#include "IOmanip.H"


class offline : public neuralAdvection
{
    public:
        /// Constructor
        explicit offline(int argc, char* argv[])
            :
            neuralAdvection(argc, argv)
        {}

};

vector neuralAdvection::evaluateField(scalar x, scalar y, scalar t)
{
    vector pointwiseU;

    pointwiseU.x() = 0.5*pow(y,2)*t;
    pointwiseU.y() = -2*x*pow(t,2);
    pointwiseU.z() = 0;

    return pointwiseU;
};

int main(int argc, char* argv[])
{
    
    List<scalar> X;
    offline sim12(argc, argv);
    
    sim12.startTime = 0;
    sim12.finalTime = 1.0;
    sim12.timeStep = 0.001;
    sim12.writeEvery = 0.01;

    sim12.flag = true;
    sim12.flagPrint = false;
    sim12.flagOutput = true;

    sim12.truthSolve(X);

}
