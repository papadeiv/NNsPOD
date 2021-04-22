//#include "createFvOptions.H"

Info<< "Reading-in field r\n" << endl;

volScalarField r
(
    IOobject
    (
        "r",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);


Info<< "Reading-in field U\n" << endl;

volVectorField U
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
);

Info<< "Reading-in file createPhi.H"<< endl;

#include "createPhi.H"

Info<< "Reading-in constant/transportProperties\n" << endl;

IOdictionary transportProperties
(
    IOobject
    (
        "transportProperties",
        runTime.constant(),
        mesh,
        IOobject::MUST_READ_IF_MODIFIED,
        IOobject::NO_WRITE
    )
);


Info<< "Reading-in entries of keyword nu\n" << endl;

dimensionedScalar nu("nu", dimViscosity, transportProperties);
