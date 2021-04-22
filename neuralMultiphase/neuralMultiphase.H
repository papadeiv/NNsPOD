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

#ifndef neuralMultiphase_H
#define neuralMultiphase_H
#include "fvCFD.H"
#include "fvOptions.H"
#include "CMULES.H"
#include "pimpleControl.H"
#include "interpolation.H"
#include "singlePhaseTransportModel.H"
#include "immiscibleIncompressibleTwoPhaseMixture.H"
#include "fixedFluxPressureFvPatchScalarField.H"
#include "IOMRFZoneList.H"
#include "subCycle.H"
#include "unsteadyNS.H"
#include "EulerDdtScheme.H"
#include "localEulerDdtScheme.H"
#include "CrankNicolsonDdtScheme.H"
#include "cnpy.H"
#include <stdlib.h>
#include <bits/stdc++.h> 
#include <iostream> 
#include <sys/stat.h> 
#include <sys/types.h>

class neuralMultiphase: public unsteadyNS
{
    public:

        /******************** CONSTRUCTORS ********************/

        neuralMultiphase();
        neuralMultiphase(int argc, char* argv[]);

        scalar startTime;

        scalar finalTime;

        scalar timeStep;

        scalar writeEvery = timeStep;

        scalar nextWrite;

        autoPtr<volScalarField> _p_rgh;

        PtrList<volScalarField> field;

        PtrList<volScalarField> shifted_field;

        PtrList<volScalarField> pfield;

        PtrList<volVectorField> Ufield;

        autoPtr<volScalarField> _shifted_alpha;

        volScalarModes modes;

        volScalarModes pmodes;

        volVectorModes Umodes;

        /******************** MEMBER FUNCTIONS ********************/

        void truthSolve(List<scalar> mu_now, fileName folder = "./ITHACAoutput/Offline/");

        bool checkWrite(Time& timeObject);
};
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif










