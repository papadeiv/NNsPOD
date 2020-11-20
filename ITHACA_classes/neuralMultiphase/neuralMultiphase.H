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
#include "CMULES.H"
#include "pimpleControl.H"
#include "interpolation.H"
#include "singlePhaseTransportModel.H"
#include "fixedFluxPressureFvPatchScalarField.H"
#include "steadyNS.H"
#include "cnpy.H"
#include <iostream>


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

        bool adjustTimeStep;

        scalar maxCo;

        scalar maxDeltaT;

        label counter2 = 1;

        word timedepbcMethod;

        Eigen::MatrixXd timeBCoff;

        word timeDerivativeSchemeOrder;

        PtrList<volScalarField> field;

        PtrList<volScalarField> shifted_field;

        autoPtr<volScalarField> _f;

        autoPtr<volScalarField> _shifted_f;

        volScalarModes fmodes;

        bool flag;

        bool flagPrint;

        bool flagOutput;

        /******************** MEMBER FUNCTIONS ********************/

        void truthSolve(List<scalar> mu_now, fileName folder = "./ITHACAoutput/Offline/");

        bool checkWrite(Time& timeObject);
};
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif










