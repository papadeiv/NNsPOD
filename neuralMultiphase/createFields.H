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

_shifted_alpha = autoPtr<volScalarField>
     (
         new volScalarField
         (
             IOobject
             (
                 "shifted_alpha",
                 runTime.timeName(),
                 mesh,
                 IOobject::MUST_READ,
                 IOobject::AUTO_WRITE
             ),
             mesh
         )
     );
volScalarField& shifted_alpha = _shifted_alpha();

Info << "Reading field pressure field (p)\n" << endl;
_p_rgh = autoPtr<volScalarField>
     (
         new volScalarField
         (
             IOobject
             (
                 "p_rgh",
                 runTime.timeName(),
                 mesh,
                 IOobject::MUST_READ,
                 IOobject::AUTO_WRITE
             ),
             mesh
         )
     );
volScalarField& p_rgh = _p_rgh();
volScalarField p0(p_rgh);
_p0 = autoPtr<volScalarField>
      (
          new volScalarField(p0)
      );


Info << "Reading field transport field (U)\n" << endl;
_U = autoPtr<volVectorField>
     (
         new volVectorField
         (
             IOobject
             (
                 "U",
                 runTime.timeName(),
                 mesh,
                 IOobject::MUST_READ,
                 IOobject::AUTO_WRITE
             ),
             mesh
         )
     );
volVectorField& U = _U();
volVectorField U0(U);
_U0 = autoPtr<volVectorField>
      (
          new volVectorField(U0)
      );

_MRF = autoPtr<IOMRFZoneList>
 (
     new IOMRFZoneList(mesh)
 );
