# DISCLAIMER: The repository is a work in progress and made public only to ease the access to other members of my team for consultation and fast upload. It is by no means complete nor standalone and it is linked with the forked repository of ITHACA-FV on my profile.

# NNsPOD

NNsPOD is the name of a family of C++ classes implemented for the reduced order modelling (ROM) of hyperbolic PDEs of interest in computational fluid dynamics (CFD).
The repository is based on ITHACA-FV, a C++ object-oriented library for the implementation of ROM techniques of computational models in OpenFOAM.

What follows is the directory structure with a breif description of the subdirectories content regarding both the solvers and the simulations performed with the aforementioned libraries.

## Contents

The repository is organised as follows:
- [x] an OpenFOAM solver called __transpDomFoam__  for the full-order solution of linear transport problems with constant and uniform velocity field;
- [x] an ITHACA-FV class called __linearWave__ that implements a traditional POD-Galerkin reduction of a scalar advected field along 1-dimension without pre-processing;
- [x] an ITHACA-FV class called __generalAdvection__ that implements the most generic method for shifting and reducing any linear transport equation with standard detection;
     - [x] 2-dimensional scalar field transported with uniform and constant speed;
     - [x] 2-dimensional scalar field transported with uniform but unsteady velocity field;
     - [x] 2-dimensional scalar field transported with non-uniform and unsteady linear velocity field;
- [ ] an ITHACA-FV class called __neuralAdvection__ that performs the same shift-reduction technique but with a ML-based detecting method;

## Installation

Once downloaded, the content of the present repository must be moved into different directories in order for the code to compile and execute. Specifically, once both OpenFOAM and ITHACA-FV have been installed, one must
* move __transpDomFoam__ into the proper sub-directory among the other solvers in OpenFOAM
```bash
cd NNsPOD
mv transpDomFoam $FOAM_SOLVERS/
``` 
;
* move __generalAdvection__ and __linearWave__ into the proper sub-directory in ITHACA-FV and amend the make file accordingly
```bash
mv generalAdvection ~/ITHACA-FV/src/ITHACA_FOMPROBLEMS
```
;
