#include "neuralAdvection.H"
#include "ITHACAstream.H"
#include "ITHACAPOD.H"
#include "forces.H"
#include "IOmanip.H"


class onlineTransport : public neuralAdvection
{
    public:
        /// Constructor
        explicit onlineTransport(int argc, char* argv[])
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
    
    Eigen::MatrixXd X;
    onlineTransport sim10_continued(argc, argv);

	// ITHACAparameters* para = ITHACAparameters::getInstance(sim10_continued._mesh(), sim10_continued._runTime());
    // int NmodesOut = para->ITHACAdict->lookupOrDefault<int>("NmodesOut", 15);

    X = cnpy::load(X, "shifted_snapshot_matrix.npy", "colMajor");

    int Nh = X.rows(); // X.rows() = 2500
    int Ns = X.cols(); // X.cols() = 101

    Field<scalar> f(Nh);
    Field<scalar> shifted_field;

    for (int j=0; j<=Ns-1; j++)
    {

        Eigen::MatrixXd snapp(Nh,1);

        for (int k=0; k<=Nh-1; k++)
        {
            snapp(k,0) = X(k,j);
        }

        //volScalarField f("f", Foam2Eigen::Eigen2field(f, snapp));

        f = Foam2Eigen::Eigen2field(f, snapp); // f.size() = Nh

        shifted_field.append(f); // shifted_field.size() = Nh*(j+1)
    }

    // shifted_field.size() = Nh*Ns

    ITHACAPOD::getModes(sim10_continued.shifted_field, sim10_continued.fmodes, sim10_continued._f().name(), sim10_continued.podex, 0, 0, NmodesOut);

}
