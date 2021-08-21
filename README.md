# NNsPOD - Neural Network shifted-Proper Orthogonal Decomposition

NNsPOD is a machine-learning based pre-processing algorithm for the model order reduction of non-linear hyperbolic equations with unknown transport fields. It consists of a split architecture of two neural networks __InterpNet__ and __ShiftNet__ with a continuous data-flow in between (i.e. the output of the former is cascaded into the input layer of the former whilst their training is performed separately).

The project has been developed within mathLab's research group at SISSA - Scuola Internazionale degli Studi Avanzati by myself, Nicola Demo, Michele Girfoglio and Giovanni Stabile under the supervision of group's head Prof. Gianluigi Rozza. The results of this algorithm, as well as its detailed mathematical derivation and implementation, can be found in [our paper](https://arxiv.org/abs/2108.06558).
The algorithm is written in __Python__ using __PyTorch__ framework and it is located in the directory `NNsPOD` of this repository.

The software can be interfaced with any other library through suitable wrappers: in this repository NNsPOD has be used to decompose fluid fields derived numerically in C++ as oulined below.

## Content
The repository, is linked with [ITHACA-FV](https://github.com/mathLab/ITHACA-FV), an open-source C++ suite, developed and maintaned by mathLab's group, that implements various ROM techniques (including POD on which this algorithm is based) of full-order simulations performed in [OpenFOAM-v2006](https://www.openfoam.com/news/main-news/openfoam-v20-06). 

Despite that NNsPOD consists of Python scripts that can be adapted to interface with other outputs depending on the needs of the user. At the moment we are not planning to introduce an automated interface, or wrapping feature for the software. All NNsPOD's scripts can be found in the `NNsPOD` directory.

The repository is organised as follows:
- [x] `generalAdvection` is an ITHACA-FV class that allows to solve for an advection equation with non-uniform, non-constant transport field and performing a shifted-Proper Orthogonal Decompostion (the algorithm on which such technique is based is derived from a [2018 article by Reiss et al](https://arxiv.org/pdf/1512.01985.pdf));
- [x] `neuralAdvection` reduces the same advection equation but using NNsPOD automatic shift detection;
- [x] `neuralMultiphase` applies NNsPOD reduction to a multiphase fluid flow with specified densities and viscosities for the two fluids;
- [x] `results` contains the numerical experiments performed and reported in [our paper](https://arxiv.org/abs/2108.06558). Specifically:
	- [x] `advection2d/SquarePOD` uses the solver __generalAdvection__ and it refers to the simulation in Section 2 of the article;
	- [x] `advection2d/SquareNNsPOD` uses the solver __neuralAdvection__ and it refers to the simulation in Section 3 of the article;
	- [x] `multiphase` uses the solver __neuralMultiphase__ and it refers to the simulation in Section 4 of the article.

## Installation
Once downloaded, the content of the present repository must be moved into different directories in order for the code to compile and execute. Specifically, once both OpenFOAM and ITHACA-FV have been installed and compiled correctly, one must:
* copy the chosen ITHACA-FV class directory into the `src` subdirectory of ITHACA-FV
```bash
cp -r neuralMultiphase ~/ITHACA-FV/src/ITHACA_FOMPROBLEMS
``` 
* write on `ITHACA_FOMPROBLEMS/Make/files` the name of ITHACA-FV class among the list of the files to be compiled and then compile it
```bash
cd ~/ITHACA-FV/src/ITHACA_FOMPROBLEMS
wclean
wmake
```
* for __neuralMultiphase__ class only the file `ITHACA_FOMPROBLEMS/Make/options` must also be amended by inserting the following lines before compiling:
```bash
-I$(LIB_SRC)/transportModels/immiscibleIncompressibleTwoPhaseMixture \
-I$(LIB_SRC)/transportModels/incompressible/singlePhaseTransportModel \
-I$(LIB_SRC)/transportModels/incompressible/incompressibleTwoPhaseMixture \
```

Now the solver should be available among the compiled executables.

## Launching simulation
Once the binary file for the class is created one can either create an ITHACA-FV class of its own or simply amend one among those in the `results` sub-directory. To execute the simulation one must:
* create a case and, following the OpenFOAM proper setup for the simulation to be run, copy the directory NNsPOD into the desired case for the simulation (only for __neuralAdvection__ and __neuralMultiphase__)
```bash
mkdir sim
cp -r NNsPOD ./sim/
``` 
* compile the C++ script associated to the full-order simulation and snapshot collection `offline.C`
```bash
cd sim
wclean
wmake
``` 
* once the binary is created one must perform the full-order simulation as per OpenFOAM workflow and execute the binary of the solver
```bash
blockMesh
setExpreFields
offline
``` 
* at the end of the full-order computation the `ITHACAoutput/NUMPYsnapshots` subdirectory will contain the snapshots collected into numpy arrays. Then move to the `NNsPOD` subdirectory and, once setting the neural networks architecture, the `main.py` script can be executed
```bash
cd NNsPOD
python main.py
``` 
At this point, to conclude the simulation one must:
* compile the `online.C` script that converts `ITHACAoutput/NUMPYsnapshots/shifted_snapshot_matrix.npy` into OpenFOAM objects and execute it to perform the POD
```bash
cd online
wclean
wmake
online
``` 
Now in `online/ITHACAoutput/Offline` the pre-processed shifted snaphots will be found whereas in `online/ITHACAoutput/POD` the modes and the singular values associated to the POD will be found.


## NNsPOD scripts
In the directory `NNsPOD` there are 4 scripts:
- [x] `main.py` is the script to be launched for running NNsPOD. In it the reference configuration for the training is specified alongside the test snapshot to be plotted for the shift monitoring (line 8);
- [x] `interp_net.py` implements the __InterpNet__ part of NNsPOD. The network's parameters can be specified (lines 15-27 and 42);
- [x] `shift_net.py` implements the __ShiftNet__ part of NNsPOD. The network's paramaters can be specified (lines 16-38 and 46);
- [x] `plot_test.py` is the script that creates the plots for the training stages of __InterpNet__ and __ShiftNet__. The size of the 2-dimensional grid has to be specified (lines 19-20)
Once ShiftNet's training is completed `main.py` will generate in `ITHACAoutput/NUMPYsnapshots` a (numpy) shifted snapshot's matrix accoring to the autodetected best bijective mapping. Also in NNsPOD directory three folders `Plots`, `Results` and `TrainedModels` will contain, respectively, the matplotlib plots of various training stages of InterpNet and ShiftNet, the .csv files containing the loss values at each epoch of boths networks' optimisers and the fully trained networks models (weights and biases).