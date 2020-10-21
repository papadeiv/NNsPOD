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

	ITHACAparameters* para = ITHACAparameters::getInstance(sim10._mesh(), sim10._runTime());
    int NmodesOut = para->ITHACAdict->lookupOrDefault<int>("NmodesOut", 15);

    ITHACAPOD::getModes(sim10.field, sim10.fmodes, sim10._f().name(), sim10.podex, 0, 0,NmodesOut);

}
