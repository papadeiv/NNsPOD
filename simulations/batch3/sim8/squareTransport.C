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

    pointwiseU.x() = pow(t, 2);
    pointwiseU.y() = -2*t;
    pointwiseU.z() = 0;

    return pointwiseU;
};

int main(int argc, char* argv[])
{
    
    List<scalar> X;
    squareTransport sim8(argc, argv);
    sim8.startTime = 0;
    sim8.finalTime = 2.5;
    sim8.timeStep = 0.001;
    sim8.writeEvery = 0.01;

	ITHACAparameters* para = ITHACAparameters::getInstance(sim8._mesh(), sim8._runTime());
    int NmodesOut = para->ITHACAdict->lookupOrDefault<int>("NmodesOut", 15);

    sim8.flag = true;
    sim8.flagPrint = false;
    sim8.flagOutput = false;

    sim8.truthSolve(X);
    ITHACAPOD::getModes(sim8.field, sim8.fmodes, sim8._f().name(), sim8.podex, 0, 0, NmodesOut);

    sim8.shifting(sim8.field, sim8.shifted_field);
    ITHACAPOD::getModes(sim8.shifted_field, sim8.fmodes, sim8._f().name(), sim8.podex, 0, 0,NmodesOut);

}
