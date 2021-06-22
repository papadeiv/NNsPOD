#include "neuralAdvection.H"
#include "ITHACAstream.H"
#include "ITHACAPOD.H"
#include "forces.H"
#include "IOmanip.H"
#include "ReducedSteadyNS.H"


class online : public neuralAdvection
{
    public:
        /// Constructor
        explicit online(int argc, char* argv[])
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

    // mkDir("./ITHACAoutput/POD");
    system("ln -s ../constant ./constant");
    system("ln -s ../0 ./0");
    system("ln -s ../system ./system");

    online sim(argc, argv);

    volScalarField shifted_f("shifted_f", sim._shifted_f());
    Eigen::MatrixXd X;

    X = cnpy::load(X, "../ITHACAoutput/NUMPYsnapshots/shifted_snapshot_matrix.npy", "colMajor");

    int Nh = X.rows();
    int Ns = X.cols();

    fileName folder = "./ITHACAoutput/Offline";

    for (int j=0; j<=Ns-1; j++)
    {

        Eigen::VectorXd shifted_snapshot = X.col(j);

        shifted_f = Foam2Eigen::Eigen2field(shifted_f, shifted_snapshot); // shifted_f.size() = Nh
        ITHACAstream::exportSolution(shifted_f, name(j), folder);

        sim.shifted_field.append(shifted_f); // shifted_field.size() = Nh*(j+1)

    }// shifted_field.size() = Nh*Ns

    ITHACAPOD::getModes(sim.shifted_field, sim.fmodes, shifted_f().name(), sim.podex, 0, 0, 20);

}
