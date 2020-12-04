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

// Solve pressure equation
volScalarField rAU("rAU", 1.0 / UEqn.A());
surfaceScalarField rAUf("rAUf", fvc::interpolate(rAU));
volVectorField HbyA(constrainHbyA(rAU*UEqn.H(), U, p_rgh));
surfaceScalarField phiHbyA
(
    "phiHbyA",
    fvc::flux(HbyA)
    + fvc::interpolate(rho*rAU)*fvc::ddtCorr(U, phi, Uf)
);
MRF.makeRelative(phiHbyA);

if (p_rgh.needReference())
{
    fvc::makeRelative(phiHbyA, U);
    adjustPhi(phiHbyA, U, p_rgh);
    fvc::makeAbsolute(phiHbyA, U);
}

surfaceScalarField phig
(
    (
        mixture.surfaceTensionForce()
      - ghf*fvc::snGrad(rho)
    )*rAUf*mesh.magSf()
);

phiHbyA += phig;
constrainPressure(p_rgh, U, phiHbyA, rAUf, MRF);

while (pimple.correctNonOrthogonal())
{
    fvScalarMatrix p_rghEqn
    (
        fvm::laplacian(rAUf, p_rgh) == fvc::div(phiHbyA)
    );

    p_rghEqn.setReference(pRefCell, getRefCellValue(p_rgh, pRefCell));

    p_rghEqn.solve(mesh.solver(p_rgh.select(pimple.finalInnerIter())));

    if (pimple.finalNonOrthogonalIter())
    {
        phi = phiHbyA - p_rghEqn.flux();

        p_rgh.relax();

        U = HbyA + rAU*fvc::reconstruct((phig - p_rghEqn.flux())/rAUf);
        U.correctBoundaryConditions();
        fvOptions.correct(U);
    }
}

#include "continuityErrs.H"

p == p_rgh + rho*gh;
if (p_rgh.needReference())
{
    p += dimensionedScalar
    (
        "p",
        p.dimensions(),
        pRefValue - getRefCellValue(p, pRefCell)
    );
    p_rgh = p - rho*gh;
}
