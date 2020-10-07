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

#include "linearWave.H"

/*  Constructor */

linearWave::linearWave(int argc, char* argv[]) 
{
  _args = autoPtr<argList> (new argList(argc, argv));

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

  para = ITHACAparameters::getInstance(mesh, runTime);

}

/* Member functions */

void linearWave::truthSolve(List<scalar> X, fileName folder)
{
  Info << " Solving the full-order ALE " << endl;

  /* Declaring variables from dummy pointers in the header file*/

  Time& runTime = _runTime();
  fvMesh& mesh = _mesh();
  volScalarField& u = _u();
  volVectorField& b = _b();
  surfaceScalarField& phi = _phi();

  /* Setting time parameters */

  instantList Times = runTime.times();
  runTime.setEndTime(t_fin);
  runTime.setTime(Times[1], 1);
  runTime.setDeltaT(delta_t);
  new_t = t_0;
  new_t += write_t;

  /* Transient-loop solver */

  while (runTime.run())
  {
    runTime++;
    Info << "Simulation time step = " << runTime.timeName() << " s" << nl << endl;

    fvScalarMatrix eqn
    (
         fvm::ddt(u) + fvm::div(phi, u)
    );

    solve(eqn);

    counter++;

    Info << "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
         << "  ClockTime = " << runTime.elapsedClockTime() << " s"
         << nl << endl;

    ITHACAstream::exportSolution(u, name(counter), folder);
    std::ofstream of(folder + name(counter) + "/" + runTime.timeName());
    field.append(u);
    new_t += write_t;
    writeMu(X);
  }
}

void linearWave::project(label Nmodes)
{
  Info << "Galerkin-projection onto the extracted modes basis" << endl; 

  Info << "Number of modes selected: " << Nmodes << endl;
}


