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

#ifndef linearWave_H
#define linearWave_H
#include "fvCFD.H"
#include "IOmanip.H"
#include "Time.H"
#include "reductionProblem.H"
#include <iostream>
#include "ITHACAPOD.H"
#include "ITHACAutilities.H"
#include "ITHACAstream.H"
#include "Modes.H"
#include "fvOptions.H"

class linearWave: public reductionProblem
{

    public:
        /* Constructor */
        linearWave(int argc, char* argv[]);
        /* Destructor  */
        ~linearWave() {};

        /* Dummy variables for the class declaration*/

        autoPtr<volScalarField> _u;
        autoPtr<volVectorField> _b;
        autoPtr<surfaceScalarField> _phi;
        mutable autoPtr<fvMesh> _mesh;
        autoPtr<Time> _runTime;

        /* Attributes */

        ITHACAparameters* para;

        label counter = 0; // Counter to be used in the full-order solution

        scalar t_0; // Initial time instant for collecting snapshots

        scalar t_fin; // Final time for collecting snapshots

        scalar delta_t; // Timestep in-between two snapshots

        scalar write_t; // Timestep in-between writing in output

        scalar new_t; // Auxiliary variable

        PtrList<volScalarField> field; // List of pointes to assemble the solution field snapshots matrix

        Modes<scalar> LSV; // List of pointers to assemble the reduced basis modes matrix (Left Singular Vectors)

        /* Member functions */

        void truthSolve(List<scalar> X, fileName folder = "./ITHACAoutput/Offline");

        void project(label Nmodes);
};

#endif
