# DISCLAIMER: The repository is a work in progress and made public only to ease the access to other members of my team for consultation and fast upload. It is by no means complete nor standalone and it is linked with the forked repository of ITHACA-FV on my profile.

# NNsPOD

NNsPOD is the name of a family of C++ classes implemented for the reduced order modelling (ROM) of hyperbolic PDEs of interest in computational fluid dynamics (CFD).
The repository is based on ITHACA-FV, a C++ object-oriented library for the implementation of ROM techniques of computational models in OpenFOAM.

What follows are simply the directories containing the solvers and simulations performed with the aforementioned libraries and, as such, they do not compile on their own.

## Installation

Once downloaded, the content of the present repository must be moved into different directories in order for the code to compile and execute. Specifically, once both OpenFOAM and ITHACA-FV have been installed, one must
* move __linearWave__ into the proper sub-directory among the other solvers in OpenFOAM
```bash
cd NNsPOD
mv linearWave $FOAM_SOLVERS/
``` 
;
* move __generalAdvection__ into the proper sub-directory in ITHACA-FV and amend the make file accordingly
```bash
mv generalAdvection ~/ITHACA-FV/src/ITHACA_FOMPROBLEMS
```
;

## Contents

The repository is organised as follows:
- [ ] an OpenFOAM solver called __linearWave__  for the full-order solution of linear transport problems with constant and uniform velocity field;
- [x] an ITHACA-FV class called __generalAdvection__ that implements the most generic method for shifting and reducing any linear transport equation with standard detection;
     - [x] 2-dimensional scalar field transported with uniform and constant speed;
     - [x] 2-dimensional scalar field transported with uniform but unsteady velocity field;
     - [x] 2-dimensional scalar field transported with non-uniform and unsteady linear velocity field;
- [ ] an ITHACA-FV class called __neuralAdvection__ that performs the same shift-reduction technique but with a ML-based detecting method;
