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

	system("ln -s ../constant ./constant");
    system("ln -s ../0 ./0");
    system("ln -s ../system ./system");

    online sim(argc, argv);

    volScalarField shifted_alpha("shifted_alpha", sim._shifted_alpha());
    Eigen::MatrixXd X;

    X = cnpy::load(X, "../ITHACAoutput/NUMPYsnapshots/shifted_snapshot_matrix.npy", "colMajor");

    int Nh = X.rows();
    int Ns = X.cols();

    fileName folder = "./ITHACAoutput/Offline";

    for (int j=0; j<=Ns-1; j++)
    {

        Eigen::VectorXd shifted_snapshot = X.col(j);

        shifted_alpha = Foam2Eigen::Eigen2field(shifted_alpha, shifted_snapshot); // shifted_f.size() = Nh
        ITHACAstream::exportSolution(shifted_alpha, name(j), folder);

        sim.shifted_field.append(shifted_alpha); // shifted_field.size() = Nh*(j+1)

    }// shifted_field.size() = Nh*Ns

    ITHACAPOD::getModes(sim.shifted_field, sim.modes, shifted_alpha().name(), sim.podex, 0, 0, 20);

}
