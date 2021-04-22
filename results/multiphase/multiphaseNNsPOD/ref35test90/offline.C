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
    offline sim16(argc, argv);
    
    sim16.startTime = 0;
    sim16.finalTime = 5;
    sim16.timeStep = 0.01;
    sim16.writeEvery = 0.05;

    ITHACAparameters* para = ITHACAparameters::getInstance(sim16._mesh(), sim16._runTime());
    int NmodesOut = para->ITHACAdict->lookupOrDefault<int>("NmodesOut", 15);

    sim16.truthSolve(X);
    ITHACAPOD::getModes(sim16.field, sim16.modes, "alpha.water", sim16.podex, 0, 0, NmodesOut);
    ITHACAPOD::getModes(sim16.Ufield, sim16.Umodes, sim16._U().name(), sim16.podex, 0, 0, NmodesOut);

}