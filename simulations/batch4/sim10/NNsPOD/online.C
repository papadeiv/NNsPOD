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

int main(int argc, char* argv[])
{

    online sim10(argc, argv);

    volScalarField shifted_f("shifted_f", sim10._shifted_f());
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

        sim10.shifted_field.append(shifted_f); // shifted_field.size() = Nh*(j+1)

    }// shifted_field.size() = Nh*Ns

    ITHACAPOD::getModes(sim10.shifted_field, sim10.fmodes, shifted_f().name(), sim10.podex, 0, 0, 10);

    // int NmodesProject = 2;
    // sim10.project(NmodesProject);
    // ReducedSteadyNS reduced(sim10);

    // // Solve the online reduced problem some new values of the parameters
    // for (int i = 0; i < 10; i++)
    // {
    //     reduced.solveOnline(sim10.mu.row(i));
    // }

    // // Reconstruct the solution and store it into Reconstruction folder
    // reduced.reconstruct("./ITHACAoutput/Reconstruction");

}
