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

#include "generalAdvection.H"
#include <unsupported/Eigen/MatrixFunctions>

/*  Constructor */

generalAdvection::generalAdvection(int argc, char* argv[])
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

void generalAdvection::truthSolve(List<scalar> mu_now, fileName folder)
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


bool generalAdvection::checkWrite(Time& timeObject)
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

vector generalAdvection::findShift(int timestep_coeff, int centroid_index, vector U_now)
{

	Time& runTime = _runTime();
	fvMesh& mesh = _mesh();

	volVectorField centroids = mesh.C();
  volScalarField centroids_x = centroids.component(vector::X);
  volScalarField centroids_y = centroids.component(vector::Y);

  if(flagPrint==true)
  {
    Info << "We are considering the " << timestep_coeff << "-th snapshot" << nl << endl;

    Info << "Of the full-order solution at time " << timestep_coeff <<
    " we are now considering its value at the " << centroid_index << "-th cell centre" << nl << endl;
  }

	vector centr = centroids[centroid_index];
	scalar x = centroids_x[centroid_index];
	scalar y = centroids_y[centroid_index];
	
	if(flagPrint==true)
  {
    Info << "The centroid's coordinates are: " << nl << "x = " << x << nl
      << "y = " << y << nl << endl;
  }

	vector pointwiseU;

	if(flag==true)
	{
		pointwiseU = evaluateField(x, y, timestep_coeff*timeStep);
	}
	else
	{
		pointwiseU = U_now;
	}

	scalar Ux = pointwiseU.x();
	scalar Uy = pointwiseU.y();

	x = x - timestep_coeff*timeStep*Ux;
	y = y - timestep_coeff*timeStep*Uy;

	if(flagPrint==true)
  {
    Info << "Following the shift the new coordinates are: " << nl << "x = " << x << nl
		  << "y = " << y << nl << endl;
  }

	vector shift;

	shift.x() = x;
	shift.y() = y;
	shift.z() = 0;

	return shift;
}

void generalAdvection::shifting(PtrList<volScalarField>& field, 
  PtrList<volScalarField>& shifted_field)
{
    
    Time& runTime = _runTime();
    fvMesh& mesh = _mesh();
    volScalarField& shifted_f = _shifted_f();

    /* The following operations reads a Foam::vector type from ITHACADict and creates a 
       referenced volVectorField from it; IT IS REDUNDANT since the velocity field is
       already read in from the transportProperties dictionary however here we use this
       operation to write the uniform field out in a file so that to compare with the 
       possible non-uniform variation create by the evaluateField method */

    ITHACAparameters* para = ITHACAparameters::getInstance(mesh, runTime);
    vector U(para->ITHACAdict->lookup("U"));

    autoPtr<volVectorField> Ufield
    (
	    new volVectorField
	    (
	        IOobject
	        (
	            "U",
	            runTime.timeName(),
	            mesh,
	            IOobject::NO_READ,
	            IOobject::NO_WRITE
	        ),
	        mesh, 
	        dimensionedVector("U", dimVelocity, U)
	    )
	  );

    /* Geometric parameters regarding the centroids are read in and write on separate files */

    volVectorField centroids = mesh.C();
    volScalarField centroids_x = centroids.component(vector::X);
    volScalarField centroids_y = centroids.component(vector::Y);

    if(flagOutput==true)
    {

      word Centroids;
      word Mesh;

      ITHACAstream::exportSolution(centroids, Centroids, "./constant/Mesh/Centroids");
      ITHACAstream::exportSolution(centroids_x, Mesh, "./constant/Mesh/x-component");
      ITHACAstream::exportSolution(centroids_y, Mesh, "./constant/Mesh/y-component");

    }

    for (label j = 0; j < field.size(); j++)
    {

        /* Here we parametrize the runTime timstep with a multiplicative integer
           i.e. we create int k s.t. runTime.value() = k*runTime.deltaT() */

        int k = j+1;

        /* Here we simply save the full-order solution's dimension in an int 
           variable N for ease of use in the following for-loop */

        int N = mesh.C().size();

        ITHACAstream::exportSolution(field[j], name(j), "./constant/Snapshot");
        ITHACAstream::exportSolution(*Ufield, name(j), "./constant/Snapshot/");

	      vector U_now;
	      vector shift;

	      dictionary interpolationDict = mesh.solutionDict().subDict("interpolationSchemes");
        autoPtr<interpolation<scalar>> fieldInterp = interpolation<scalar>::New(interpolationDict, field[j]);

        label closest_centroid;
        scalar shift_f;

        for (label l=0; l < N; l++)
        {
       		 U_now = (*Ufield)[l];
       		 shift = findShift(k, l, U_now);

           if(shift[0]>4)
           {
            shift[0]=4;
           }

           if(shift[1]>4)
           {
            shift[1]=4;
           }

           if(shift[0]<0)
           {
            shift[0]=0;
           }

           if(shift[1]<0)
           {
            shift[1]=0;
           }

           /* Here the interpolation coordinates of the centroid are found giving as input
              the shifted coordinates that follow as output of the method findShift */

    		   closest_centroid = mesh.findCell(shift);

    		   if(flagPrint==true)
           {
              Info << "For which the closest cell centroid label is the " << closest_centroid 
                << "-th cell with coordinates: " << nl << "x = " << mesh.C()[closest_centroid].x()
                << nl << "y = " << mesh.C()[closest_centroid].y() << nl << endl;
           }

    		    /* Here the value of the new, shifted field is found through interpolation with
                  the closest centroid following the previous space-shift */

           shift_f = fieldInterp->interpolate(shift, closest_centroid);

    		   shifted_f[l] = shift_f;
        }


        shifted_field.append(shifted_f);

	      ITHACAstream::exportSolution(shifted_f, name(j), "./ITHACAoutput/Offline/");

	  }
	
}

/*

********** HOW TO ACCESS GeometricField<Type> VALUES **********

					volVectorField

<autoPtr>volVectorField Ufield;

Info << Ufield() << nl << endl;

Info << Ufield()[73] << nl << endl;

Info << (*Ufield)[73][0] << " = "<< Ufield()[73][0] << nl << endl;
Info << (*Ufield)[73][1] << " = "<< Ufield()[73][1] << nl << endl;

					volScalarField

volScalarField xComponent = mesh.C().component(vector::X)

Info << xComponent << nl << endl;

Info << xComponent[73] << nl << endl;

*/

// autoPtr<volVectorField> _centroids;
// autoPtr<volScalarField> _centroids_x;
// autoPtr<volScalarField> _centroids_y;

// volVectorField& centroids = _centroids();
// centroids = mesh.C();

// Info << "OK" << endl;

// volScalarField& centroids_x = _centroids_x();
// centroids_x = centroids.component(vector::X);

// volScalarField& centroids_y = _centroids_y();
// centroids_y = centroids.component(vector::Y);



