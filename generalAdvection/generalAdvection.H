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
Class
    generalAdvection
Description
    Reduction class for a non-linear transport problem
SourceFiles
    generalAdvection.C
\*---------------------------------------------------------------------------*/

#ifndef generalAdvection_H
#define generalAdvection_H
#include "fvCFD.H"
#include "interpolation.H"
#include "singlePhaseTransportModel.H"
#include "IOMRFZoneList.H"
#include "fixedFluxPressureFvPatchScalarField.H"
#include "steadyNS.H"
#include <iostream>


class generalAdvection: public steadyNS
{
    public:

        /******************** CONSTRUCTORS ********************/

        generalAdvection();
        generalAdvection(int argc, char* argv[]);

        /******************** ATTRIBUTES ********************/

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

        virtual vector evaluateField(scalar x, scalar y, scalar t);

        void shifting(PtrList<volScalarField>& field, PtrList<volScalarField>& shifted_field);

        vector findShift(int timestep, int centroid, vector U_now);
};
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif










