/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*-------------------------------------------------------------------------------\*

  This is a simple solver for a 1D transient, convection-diffusion equation for
 a passive scalar quantity 'r' subject to constant and uniform velocity field 'U'

/*-------------------------------------------------------------------------------*/


#include "fvCFD.H"
// #include "fvOptions.H"

int main(int argc, char *argv[])
{
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
    // * * * * * * * * * Include directive of header files * * * * * * * * * //
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    #include "addCheckCaseOptions.H"
    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createMesh.H"

    #include "createFields.H"

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
    // * * * * * * * * * * * Reports message on screen * * * * * * * * * * * //
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info<< "\n Computing transported (scalar) field \n" << endl;

    #include "CourantNo.H"

    Info<<"\n Mesh is collocated and homogeneous and timesteps equispaced hence Courant's Number is constant for the whole simulation \n"<< endl;

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
    // * * * * * Start the time loop over the discretized equation * * * * * //
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //


    while (runTime.loop())
    {
    	Info<< "Time = " << runTime.timeName() << nl << endl;

    	fvScalarMatrix eqn
	    (
	        fvm::ddt(r)
	      + fvm::div(phi, r)
	      - fvm::laplacian(nu, r)
	    );

	    solve(eqn);

	    runTime.write();
    }

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
