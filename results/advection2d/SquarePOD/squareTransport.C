#include "generalAdvection.H"
#include "ITHACAstream.H"
#include "ITHACAPOD.H"
#include "forces.H"
#include "IOmanip.H"


class squareTransport : public generalAdvection
{
    public:
        /// Constructor
        explicit squareTransport(int argc, char* argv[])
            :
            generalAdvection(argc, argv)
        {}

};

vector generalAdvection::evaluateField(scalar x, scalar y, scalar t)
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
    squareTransport sim7(argc, argv);
    sim7.startTime = 0;
    sim7.finalTime = 2.5;
    sim7.timeStep = 0.001;
    sim7.writeEvery = 0.01;

	ITHACAparameters* para = ITHACAparameters::getInstance(sim7._mesh(), sim7._runTime());
    int NmodesOut = para->ITHACAdict->lookupOrDefault<int>("NmodesOut", 15);

    sim7.flag = false;
    sim7.flagPrint = false;
    sim7.flagOutput = false;

    sim7.truthSolve(X);
    ITHACAPOD::getModes(sim7.field, sim7.fmodes, sim7._f().name(), sim7.podex, 0, 0, NmodesOut);

    sim7.shifting(sim7.field, sim7.shifted_field);
    ITHACAPOD::getModes(sim7.shifted_field, sim7.fmodes, sim7._f().name(), sim7.podex, 0, 0, NmodesOut);

}
