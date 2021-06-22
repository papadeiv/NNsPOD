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
    offline sim(argc, argv);

    volScalarField f("f", sim._f());
    
    sim.startTime = 0;
    sim.finalTime = 1.0;
    sim.timeStep = 0.001;
    sim.writeEvery = 0.01;

    ITHACAparameters* para = ITHACAparameters::getInstance(sim._mesh(), sim._runTime());
    int NmodesOut = para->ITHACAdict->lookupOrDefault<int>("NmodesOut", 20);

    sim.flag = true;
    sim.flagPrint = false;
    sim.flagOutput = true;

    sim.truthSolve(X);
    ITHACAPOD::getModes(sim.field, sim.fmodes, f().name(), sim.podex, 0, 0, NmodesOut);

}
