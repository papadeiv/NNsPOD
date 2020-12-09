#include "neuralMultiphase.H"
#include "ITHACAstream.H"
#include "ITHACAPOD.H"
#include "forces.H"
#include "IOmanip.H"
#include "ReducedSteadyNS.H"


class online : public neuralMultiphase
{
    public:
        /// Constructor
        explicit online(int argc, char* argv[])
            :
            neuralMultiphase(argc, argv)
        {}

};

int main(int argc, char* argv[])
{

    online sim16(argc, argv);

    volScalarField shifted_f("shifted_f", sim16._shifted_f());
    Eigen::MatrixXd X;

    X = cnpy::load(X, "./NNsPOD/NUMPYsnapshots/shifted_snapshot_matrix.npy", "colMajor");

    int Nh = X.rows();
    int Ns = X.cols();

    fileName folder = "./ITHACAoutput/Offline";

    for (int j=0; j<=Ns-1; j++)
    {

        Eigen::VectorXd shifted_snapshot = X.col(j);

        shifted_f = Foam2Eigen::Eigen2field(shifted_f, shifted_snapshot); // shifted_f.size() = Nh
        ITHACAstream::exportSolution(shifted_f, name(j), folder);

        sim16.shifted_field.append(shifted_f); // shifted_field.size() = Nh*(j+1)

    }// shifted_field.size() = Nh*Ns

    ITHACAPOD::getModes(sim16.shifted_field, sim16.fmodes, shifted_f().name(), sim16.podex, 0, 0, 10);

}
