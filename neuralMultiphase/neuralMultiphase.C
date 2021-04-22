/*---------------------------------------------------------------------------*\
     ██╗████████╗██╗  ██╗ █████╗  ██████╗ █████╗       ███████╗██╗   ██╗
     ██║╚══██╔══╝██║  ██║██╔══██╗██╔════╝██╔══██╗      ██╔════╝██║   ██║
     ██║   ██║   ███████║███████║██║     ███████║█████╗█████╗  ██║   ██║
     ██║   ██║   ██╔══██║██╔══██║██║     ██╔══██║╚════╝██╔══╝  ╚██╗ ██╔╝
     ██║   ██║   ██║  ██║██║  ██║╚██████╗██║  ██║      ██║      ╚████╔╝
     ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝      ╚═╝       ╚═══╝

 * In real Time Highly Advanced Computational Applications for Finite Volumes
 * Copyright (C) 2017 by the ITHACA-FV authors
-------------------------------------------------------------------------------

  License
  This file is part of ITHACA-FV

  ITHACA-FV is free software: you can redistribute it and/or modify
  it under the terms of the GNU Lesser General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  ITHACA-FV is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public License
  along with ITHACA-FV. If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "neuralMultiphase.H"
#include <unsupported/Eigen/MatrixFunctions>

neuralMultiphase::neuralMultiphase(int argc, char* argv[])
{
   _args = autoPtr<argList>
           (
                new argList(argc, argv)
            );

    // if (!_args->checkRootCase())
    // {
    //     Foam::FatalError.exit();
    // }

    argList& args = _args();

	#include "createTime.H"
	#include "createMesh.H"

	_pimple = autoPtr<pimpleControl>
    (
		new pimpleControl
        (
			mesh
		)
  	);

    ITHACAdict = new IOdictionary
    (
        IOobject
        (
            "ITHACAdict",
            runTime.system(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );

	#include "createFields.H"
	#include "createPhi.H"
	#include "createFvOptions.H"

    para = ITHACAparameters::getInstance(mesh, runTime);
    offline = ITHACAutilities::check_off();
    podex = ITHACAutilities::check_pod();

    //delete args;

}

/* Member functions */

void neuralMultiphase::truthSolve(List<scalar> mu_now, fileName folder)
{
    Time& runTime = _runTime();
    instantList Times = runTime.times();
    fvMesh& mesh = _mesh();
    #include "initContinuityErrs.H"
    fv::options& fvOptions = _fvOptions();
    pimpleControl& pimple = _pimple();
    volVectorField& U = _U();
    volScalarField& p_rgh = _p_rgh();
    surfaceScalarField& phi = _phi();
    IOMRFZoneList& MRF = _MRF();

    #include "createPhases.H"
    #include "createAbsPressure.H"
    #include "createAlphaFluxes.H"
    #include "createUfIfPresent.H"

    runTime.setEndTime(finalTime);
    runTime.setTime(Times[1], 1);
    runTime.setDeltaT(timeStep);
    nextWrite = startTime;

    ITHACAstream::exportSolution(alpha1, name(counter), folder);
    ITHACAstream::exportSolution(p, name(counter), folder);
    ITHACAstream::exportSolution(U, name(counter), folder);

    std::ofstream of(folder + name(counter) + "/" + runTime.timeName());

    field.append(alpha1);
    pfield.append(p);
    Ufield.append(U);

    counter++;
    nextWrite += writeEvery;

    #include <vector>
    std::vector<float> timesteps;
    timesteps.push_back(0.0);

    int step = 0;

    while(runTime.run())
    {
    	#include "readTimeControls.H"
		#include "CourantNo.H"
		#include "alphaCourantNo.H"
		#include "setDeltaT.H"

		runTime.setEndTime(finalTime);
      	runTime++;
      	Info << "Time = " << runTime.timeName() << nl << endl;

      	while(pimple.loop())
      	{
      		#include "alphaControls.H"
            #include "alphaEqnSubCycle.H"

            mixture.correct();

            #include "UEqn.H"
            while (pimple.correct())
            {
                #include "pEqn.H"
            }
      	}

      	Info << "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
             << "  ClockTime = " << runTime.elapsedClockTime() << " s"
             << nl << endl;

        if (checkWrite(runTime))
        {
        	step++;
            timesteps.push_back(runTime.value());

        	ITHACAstream::exportSolution(alpha1, name(counter), folder);
        	ITHACAstream::exportSolution(p, name(counter), folder);
            ITHACAstream::exportSolution(U, name(counter), folder);
            std::ofstream of(folder + name(counter) + "/" + runTime.timeName());

            field.append(alpha1);
            pfield.append(p);
            Ufield.append(U);

            counter++;
            nextWrite += writeEvery;
            writeMu(mu_now);

            mu_samples.conservativeResize(mu_samples.rows() + 1, mu_now.size() + 1);
            mu_samples(mu_samples.rows() - 1, 0) = atof(runTime.timeName().c_str());

            for (int i = 0; i < mu_now.size(); i++)
            {
                mu_samples(mu_samples.rows() - 1, i + 1) = mu_now[i];
            }
        }
    }

    if (system("mkdir -p ./ITHACAoutput/NUMPYsnapshots") == -1)
    {
        Info << "Error :  " << strerror(errno) << endl; 
        exit(0);
    }

    std::ofstream output_file("./ITHACAoutput/NUMPYsnapshots/timesteps.txt");
    std::ostream_iterator<float> output_iterator(output_file, "\n");
    std::copy(timesteps.begin(), timesteps.end(), output_iterator);

    int Ns = field.size();
    int Nh = field[0].size();

    Info << "Ns = " << Ns << endl << "Nh = " << Nh << endl; 

    Eigen::MatrixXd snapshot(Nh, 3);

    volScalarField x("x",mesh.C().component(vector::X));
    volScalarField y("y",mesh.C().component(vector::Y));

    for (int j=0; j<=Ns-1; j++)
    {  

      snapshot.col(0) = Foam2Eigen::field2Eigen(field[j]);
      snapshot.col(1) = Foam2Eigen::field2Eigen(x);
      snapshot.col(2) = Foam2Eigen::field2Eigen(y);

      std::string name = "./ITHACAoutput/NUMPYsnapshots/";
      
      name.append(std::to_string(j));
      name.append(".npy");

      cnpy::save(snapshot, name);

    }

    if (mu.cols() == 0)
    {
        mu.resize(1, 1);
    }

    if (mu_samples.rows() == counter * mu.cols())
    {
        ITHACAstream::exportMatrix(mu_samples, "mu_samples", "eigen",
                                   folder);
    }
}

bool neuralMultiphase::checkWrite(Time& timeObject)
{
    scalar diffnow = mag(nextWrite - atof(timeObject.timeName().c_str()));
    scalar diffnext = mag(nextWrite - atof(timeObject.timeName().c_str()) -
                          timeObject.deltaTValue());

    if ( diffnow < diffnext)
    {
        return true;
    }
    else
    {
        return false;
    }
}
