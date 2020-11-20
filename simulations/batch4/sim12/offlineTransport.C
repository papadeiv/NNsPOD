#include "neuralAdvection.H"
#include "ITHACAstream.H"
#include "ITHACAPOD.H"
#include "forces.H"
#include "IOmanip.H"


class offlineTransport : public neuralAdvection
{
    public:
        /// Constructor
        explicit offlineTransport(int argc, char* argv[])
            :
            neuralAdvection(argc, argv)
        {}

};

vector neuralAdvection::evaluateField(scalar x, scalar y, scalar t)
{
    vector pointwiseU;

    pointwiseU.x() = t;
    pointwiseU.y() = 0;
    pointwiseU.z() = 0;

    return pointwiseU;
};

int main(int argc, char* argv[])
{
    
    List<scalar> X;
    offlineTransport sim10(argc, argv);
    
    sim10.startTime = 0;
    sim10.finalTime = 1.0;
    sim10.timeStep = 0.001;
    sim10.writeEvery = 0.01;

    sim10.flag = false;
    sim10.flagPrint = false;
    sim10.flagOutput = false;

    sim10.truthSolve(X);

}
