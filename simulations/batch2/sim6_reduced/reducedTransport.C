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

#include <iostream>
#include "fvCFD.H"
#include "IOmanip.H"
#include "Time.H"

#include "linearWave.H"
#include "ITHACAPOD.H"
#include "ITHACAutilities.H"
#include "ITHACAstream.H"

#include <Eigen/Dense>
#define _USE_MATH_DEFINES
#include <cmath>

class reducedTransport: public linearWave
{
    public:
        explicit reducedTransport(int argc, char* argv[])
            :
            linearWave(argc, argv)
        {};      

        List<scalar> X;

        void offline()
        {
            Info << "Initiating the offline phase" << endl;

            for (label j = 0; j< mu.cols(); j++)
            {
                X = mu(0, j);
                truthSolve(X);
            }
        }  
};


int main(int argc, char* argv[])
{
    /* Instatiating a new object of class reducedTransport */
    
    reducedTransport sim6_reduced(argc, argv);

    /* Reading in geometrical and time parameters from ITHACAdict */

    ITHACAparameters* para = ITHACAparameters::getInstance(sim6_reduced._mesh(), sim6_reduced._runTime());

    /* Reading in reduction parameters from ITHACAdict */

    int NmodesOut = para->ITHACAdict->lookupOrDefault<int>("NmodesOut", 20);
    int NmodesProj = para->ITHACAdict->lookupOrDefault<int>("NmodesProj", 20);

    /* Setting the ROM parameteric dependence */

    sim6_reduced.Pnumber = 1; /* Dimensions of the parameter vector mu */
    sim6_reduced.Tnumber = 1; /* Number of realizations in the parameter space (T:=Training samples) */

    sim6_reduced.setParameters(); /* Initialize mu and mu_range in reductionProblem.H */

    sim6_reduced.mu_range(0, 0) = 0;
    sim6_reduced.mu_range(0, 1) = 0;

    /* Setting simulation transient parameters */

    sim6_reduced.t_0 = 0;
    sim6_reduced.t_fin = 5;
    sim6_reduced.delta_t = 0.1;
    sim6_reduced.write_t = 0.5;
    
    /* Performing offline phase */

    sim6_reduced.podex = ITHACAutilities::check_pod(); // checking if POD has already been computed 
    sim6_reduced.offline();

    /* Performing the SVD and extracting the specified number of modes (NmodesOut) */

    ITHACAPOD::getModes(sim6_reduced.field, sim6_reduced.LSV, sim6_reduced._u().name(), sim6_reduced.podex, 0, 0,
                        NmodesOut);

    /* Performing the Galerkin projection */

    sim6_reduced.project(NmodesProj);


    exit(0);
}