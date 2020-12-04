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

    pointwiseU.x() = pow(t, 2);
    pointwiseU.y() = -2*t;
    pointwiseU.z() = 0;

    return pointwiseU;
};

int main(int argc, char* argv[])
{
    
    List<scalar> X;
    offline sim11(argc, argv);
    
    sim11.startTime = 0;
    sim11.finalTime = 1.0;
    sim11.timeStep = 0.001;
    sim11.writeEvery = 0.01;

    sim11.flag = true;
    sim11.flagPrint = true;
    sim11.flagOutput = true;

    sim11.truthSolve(X);

}
