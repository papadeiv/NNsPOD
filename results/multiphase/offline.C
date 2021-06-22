#include "neuralMultiphase.H"
#include "ITHACAstream.H"
#include "ITHACAPOD.H"
#include "forces.H"
#include "IOmanip.H"


class offline : public neuralMultiphase
{
    public:
        /// Constructor
        explicit offline(int argc, char* argv[])
            :
            neuralMultiphase(argc, argv)
        {}

};

int main(int argc, char* argv[])
{
    
    List<scalar> X;
    offline sim(argc, argv);
    
    sim.startTime = 0;
    sim.finalTime = 5;
    sim.timeStep = 0.01;
    sim.writeEvery = 0.05;

    ITHACAparameters* para = ITHACAparameters::getInstance(sim._mesh(), sim._runTime());
    int NmodesOut = para->ITHACAdict->lookupOrDefault<int>("NmodesOut", 15);

    sim.truthSolve(X);
    ITHACAPOD::getModes(sim.field, sim.modes, "alpha.water", sim.podex, 0, 0, NmodesOut);
    ITHACAPOD::getModes(sim.Ufield, sim.Umodes, sim._U().name(), sim.podex, 0, 0, NmodesOut);

}