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

#include "neuralAdvection.H"
#include <unsupported/Eigen/MatrixFunctions>

/*  Constructor */

neuralAdvection::neuralAdvection(int argc, char* argv[])
{
    _args = autoPtr<argList>
            (
                new argList(argc, argv)
            );

    if (!_args->checkRootCase())
    {
        Foam::FatalError.exit();
    }

    argList& args = _args();

	  #include "createTime.H"
	  #include "createMesh.H"

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

    para = ITHACAparameters::getInstance(mesh, runTime);
    offline = ITHACAutilities::check_off();
    podex = ITHACAutilities::check_pod();
    supex = ITHACAutilities::check_sup();
}

/* Member functions */

void neuralAdvection::truthSolve(List<scalar> mu_now, fileName folder)
{

    /* Declaring variables from dummy pointers in the header file */

    Time& runTime = _runTime();
    fvMesh& mesh = _mesh();
    volVectorField& U = _U();
    volScalarField& f = _f();
    surfaceScalarField& phi = _phi();
    
    /* Setting time parameters */

    instantList Times = runTime.times();
    runTime.setEndTime(finalTime);
    runTime.setTime(Times[1], 1);
    runTime.setDeltaT(timeStep);
    nextWrite = startTime;

    /* Export and store the initial conditions for the scalar field */

    ITHACAstream::exportSolution(f, name(counter), folder);

    // Creating a copy of the uniform, constant field at each timestep that gets updated by evaluateField

    volVectorField U_new = U;
    U_new.rename("U_new");

    ITHACAstream::exportSolution(U_new, name(counter), folder);

    std::ofstream of(folder + name(counter) + "/" + runTime.timeName());

    field.append(f);

    counter++;
    nextWrite += writeEvery;

    #include <vector>
    std::vector<float> timesteps;
    timesteps.push_back(0.0);

    int step = 0;

    while (runTime.run())
    {

      #include "readTimeControls.H"
    	#include "CourantNo.H"
    	#include "setDeltaT.H"
        
      runTime.setEndTime(finalTime);
      runTime++;
      Info << "Time = " << runTime.timeName() << nl << endl;

      U_new = U;

      if(flag==true)
      {

      	for(label l=0; l < mesh.C().size(); l++)
      	{
      		
          if(flagPrint==true)
          {
            // Use std::cerr instead of Info to avoid buffering at compile runtime
            std::cerr << "Before evaluateField gets called: U = " << U.internalField()[l][0] << std::endl;
          }

          U_new[l] = evaluateField(mesh.C()[l][0], mesh.C()[l][1], runTime.value());

      		if(flagPrint==true)
          {
            std::cerr << "After evaluateField gets called: U_new = " << U_new.internalField()[l][0] << std::endl;
          }
      	}
      }

      // Calculating the updated flux with the new, corrected velocity field

      _phi() = fvc::flux(U_new);

      // Assembling and solving the full-order system of LAEs

		  #include "createEqn.H"

      Info << "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
          << "  ClockTime = " << runTime.elapsedClockTime() << " s"
          << nl << endl;

      if (checkWrite(runTime))
      {

          step++;
          timesteps.push_back(runTime.value());

          ITHACAstream::exportSolution(f, name(counter), folder);
          ITHACAstream::exportSolution(U_new, name(counter), folder);

          std::ofstream of(folder + name(counter) + "/" + runTime.timeName());

          field.append(f);

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

    // Resize to Unitary if not initialized by user (i.e. non-parametric problem)
    
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

bool neuralAdvection::checkWrite(Time& timeObject)
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

/*

void neuralAdvection::projection(label Nr)
{
	volVectorField& U = _U();
	surfaceScalarField& phi = _phi();
    _phi() = fvc::flux(U);
    A_matrices.resize(Nr,Nr);

    for (int j = 0; j < Nr; j++)
    {
        for (int k = 0; k < Nr; k++)
        {
            A_matrices[j, k] = fvc::domainIntegrate( fmodes[j] * (fvm::ddt(fmodes[k]) + fvm::div(phi, fmodes[k])).value());
        }
    }

    /// Export the A matrices
    ITHACAstream::exportMatrix(A_matrices, "A", "python",
                               "./ITHACAoutput/Matrices/");
    ITHACAstream::exportMatrix(A_matrices, "A", "matlab",
                               "./ITHACAoutput/Matrices/");
    ITHACAstream::exportMatrix(A_matrices, "A", "eigen",
                               "./ITHACAoutput/Matrices/A_matrices");
}

*/