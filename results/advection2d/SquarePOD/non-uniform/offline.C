#include "generalAdvection.H"
#include "ITHACAstream.H"
#include "ITHACAPOD.H"
#include "forces.H"
#include "IOmanip.H"


class offline : public generalAdvection
{
    public:
        /// Constructor
        explicit offline(int argc, char* argv[])
            :
            generalAdvection(argc, argv)
        {}

};

vector generalAdvection::evaluateField(scalar x, scalar y, scalar t)
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
    sim.startTime = 0;
    sim.finalTime = 1.0;
    sim.timeStep = 0.001;
    sim.writeEvery = 0.01;

	ITHACAparameters* para = ITHACAparameters::getInstance(sim._mesh(), sim._runTime());
    int NmodesOut = para->ITHACAdict->lookupOrDefault<int>("NmodesOut", 10);

    sim.flag = true;
    sim.flagPrint = false;
    sim.flagOutput = true;

    sim.truthSolve(X);
    ITHACAPOD::getModes(sim.field, sim.fmodes, sim._f().name(), sim.podex, 0, 0, NmodesOut);

    sim.shifting(sim.field, sim.shifted_field);
    ITHACAPOD::getModes(sim.shifted_field, sim.fmodes, sim._f().name(), sim.podex, 0, 0, NmodesOut);

}
