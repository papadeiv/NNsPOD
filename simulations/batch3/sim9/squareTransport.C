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

    pointwiseU.x() = 0.25*y;
    pointwiseU.y() = -x;
    pointwiseU.z() = 0;

    return pointwiseU;
};

int main(int argc, char* argv[])
{
    
    List<scalar> X;
    squareTransport sim9(argc, argv);
    sim9.startTime = 0;
    sim9.finalTime = 2.5;
    sim9.timeStep = 0.001;
    sim9.writeEvery = 0.01;

	ITHACAparameters* para = ITHACAparameters::getInstance(sim9._mesh(), sim9._runTime());
    int NmodesOut = para->ITHACAdict->lookupOrDefault<int>("NmodesOut", 15);

    sim9.flag = true;
    sim9.flagPrint = false;
    sim9.flagOutput = false;

    sim9.truthSolve(X);
    ITHACAPOD::getModes(sim9.field, sim9.fmodes, sim9._f().name(), sim9.podex, 0, 0,NmodesOut);

    sim9.shifting(sim9.field, sim9.shifted_field);
    ITHACAPOD::getModes(sim9.shifted_field, sim9.fmodes, sim9._f().name(), sim9.podex, 0, 0,NmodesOut);

}
