/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2009 Stanford University and the Authors.           *
 * Authors: Scott Le Grand, Peter Eastman                                     *
 * Contributors:                                                              *
 *                                                                            *
 * This program is free software: you can redistribute it and/or modify       *
 * it under the terms of the GNU Lesser General Public License as published   *
 * by the Free Software Foundation, either version 3 of the License, or       *
 * (at your option) any later version.                                        *
 *                                                                            *
 * This program is distributed in the hope that it will be useful,            *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of             *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
 * GNU Lesser General Public License for more details.                        *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public License   *
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.      *
 * -------------------------------------------------------------------------- */

//#include "kernels/gputypes.h"
#include "CudaLTMDKernelSources.h"
#include "CudaIntegrationUtilities.h"
#include "CudaContext.h"
#include "CudaArray.h"
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <vector_functions.h>
#include <cstdlib>
#include <string>
#include <iostream>
#include <stdlib.h>
using namespace std;
using namespace OpenMM;

#ifdef USE_MIXED_PRECISION
typedef double mixed;
#else
typedef float mixed;
#endif

// CPU code
void kNMLUpdate(CUmodule* module, CudaContext* cu, mixed deltaT, mixed tau, mixed kT, int numModes, int& iterations, CudaArray& modes, CudaArray& modeWeights, CudaArray& noiseVal, CudaArray& randomIndex ) {

	int atoms = cu->getNumAtoms();
	int paddednumatoms = cu->getPaddedNumAtoms();
	int numrand = cu->getIntegrationUtilities().getRandom().getSize();

	void* update1Args[] = {&atoms, &paddednumatoms, &tau, &deltaT, &kT, 
                            &cu->getVelm().getDevicePointer(), &cu->getForce().getDevicePointer(), &cu->getIntegrationUtilities().getRandom().getDevicePointer(), &randomIndex.getDevicePointer(), &numrand }; // # of random numbers equal to the number of atoms? TMC
	CUfunction update1Kernel, update2Kernel, update3Kernel;
        update1Kernel = cu->getKernel(*module, "kNMLUpdate1_kernel");
	cu->executeKernel(update1Kernel, update1Args, cu->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize);

        void* update2Args[] = {&atoms, &numModes, &cu->getVelm().getDevicePointer(), &modes.getDevicePointer(), &modeWeights.getDevicePointer()};
	update2Kernel = cu->getKernel(*module, "kNMLUpdate2_kernel");
	cu->executeKernel(update2Kernel, update2Args, cu->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize, cu->ThreadBlockSize*sizeof(mixed)); 

        void* update3Args[] = {&atoms, &numModes, &deltaT, &cu->getPosq().getDevicePointer(), &cu->getPosqCorrection().getDevicePointer(), &cu->getVelm().getDevicePointer(), &modes.getDevicePointer(), &modeWeights.getDevicePointer(), &noiseVal.getDevicePointer()};
	update3Kernel = cu->getKernel(*module, "kNMLUpdate3_kernel");
	cu->executeKernel(update3Kernel, update3Args, cu->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize, numModes*sizeof(mixed)); 
}

#ifdef FAST_NOISE
void kFastNoise(CUmodule* module, CudaContext* cu, int numModes, mixed kT, int& iterations, CudaArray& modes, CudaArray& modeWeights, mixed maxEigenvalue, CudaArray& noiseVal, CudaArray& randomIndex, CudaArray& oldpos, mixed stepSize ) {
	int atoms = cu->getNumAtoms();
	int paddednumatoms = cu->getPaddedNumAtoms();
	int numrand = cu->getIntegrationUtilities().getRandom().getSize();
	void* fastnoise1Args[] = {&atoms, &paddednumatoms, &numModes, &kT, &oldpos.getDevicePointer(), &cu->getVelm().getDevicePointer(), &modes.getDevicePointer(), &modeWeights.getDevicePointer(),&cu->getIntegrationUtilities().getRandom().getDevicePointer(), &randomIndex.getDevicePointer(), &numrand, &maxEigenvalue, &stepSize};
	CUfunction fastnoise1Kernel, fastnoise2Kernel;
	
	
	fastnoise1Kernel = cu->getKernel(*module, "kFastNoise1_kernel");
	cu->executeKernel(fastnoise1Kernel, fastnoise1Args, cu->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize, cu->ThreadBlockSize*sizeof(mixed));

        void* fastnoise2Args[] = {&atoms, &numModes, &cu->getPosq().getDevicePointer(), &cu->getPosqCorrection().getDevicePointer(), &noiseVal.getDevicePointer(), &cu->getVelm().getDevicePointer(), &modes.getDevicePointer(), &modeWeights.getDevicePointer()};

        fastnoise2Kernel = cu->getKernel(*module, "kFastNoise2_kernel");
	cu->executeKernel(fastnoise2Kernel, fastnoise2Args, cu->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize, numModes*sizeof(mixed));
}
#endif 

void kNMLRejectMinimizationStep(CUmodule* module, CudaContext* cu, CudaArray& oldpos, CudaArray& oldposCorrection ) {
	CUfunction rejectKernel = cu->getKernel(*module, "kRejectMinimizationStep_kernel");
	int atoms = cu->getNumAtoms();
	void* rejectArgs[] = {&atoms, &cu->getPosq().getDevicePointer(), &cu->getPosqCorrection().getDevicePointer(), &oldpos.getDevicePointer(), &oldposCorrection.getDevicePointer() };
	cu->executeKernel(rejectKernel, rejectArgs, cu->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize);
}

void kNMLAcceptMinimizationStep(CUmodule* module, CudaContext* cu, CudaArray& oldpos, CudaArray& oldposCorrection ) {
	CUfunction acceptKernel = cu->getKernel(*module, "kAcceptMinimizationStep_kernel");
	int atoms = cu->getNumAtoms();
	void* acceptArgs[] = {&atoms, &cu->getPosq().getDevicePointer(), &cu->getPosqCorrection().getDevicePointer(), &oldpos.getDevicePointer(), &oldposCorrection.getDevicePointer() };
	cu->executeKernel(acceptKernel, acceptArgs, cu->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize);
}

void kNMLLinearMinimize(CUmodule* module, CudaContext* cu, int numModes, mixed maxEigenvalue, CudaArray& oldpos, CudaArray& oldposCorrection, CudaArray& modes, CudaArray& modeWeights ) {
	int atoms = cu->getNumAtoms();
	int paddedatoms = cu->getPaddedNumAtoms();
	void* linmin1Args[] = {&atoms, &paddedatoms, &numModes, &cu->getVelm().getDevicePointer(), &cu->getForce().getDevicePointer(), &modes.getDevicePointer(), &modeWeights.getDevicePointer()};
	CUfunction linmin1Kernel, linmin2Kernel;
        linmin1Kernel = cu->getKernel(*module, "kNMLLinearMinimize1_kernel");
	cu->executeKernel(linmin1Kernel, linmin1Args, cu->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize, cu->ThreadBlockSize*sizeof(mixed));
	linmin2Kernel = cu->getKernel(*module, "kNMLLinearMinimize2_kernel");
	mixed oneoverEig = (mixed) 1.0f/maxEigenvalue;
	void* linmin2Args[] = {&atoms, &paddedatoms, &numModes, &oneoverEig, &cu->getPosq().getDevicePointer(), &cu->getPosqCorrection().getDevicePointer(), &oldpos.getDevicePointer(), &oldposCorrection.getDevicePointer(), &cu->getVelm().getDevicePointer(), &cu->getForce().getDevicePointer(), &modes.getDevicePointer(), &modeWeights.getDevicePointer()};
        cu->executeKernel(linmin2Kernel, linmin2Args, cu->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize, numModes*sizeof(mixed));
}


void kNMLQuadraticMinimize(CUmodule* module, CudaContext* cu, mixed maxEigenvalue, mixed currentPE, mixed lastPE, CudaArray& oldpos, CudaArray& oldposCorrection, CudaArray& slopeBuffer, CudaArray& lambdaval ) {
	int atoms = cu->getNumAtoms();
	int paddedatoms = cu->getPaddedNumAtoms();
	void* quadmin1Args[] = {&atoms, &paddedatoms, 
				&oldpos.getDevicePointer(), 
				&oldposCorrection.getDevicePointer(),
				&cu->getVelm().getDevicePointer(), 
				&cu->getForce().getDevicePointer(), 
				&slopeBuffer.getDevicePointer()};
	CUfunction quadmin1Kernel, quadmin2Kernel;
        quadmin1Kernel = cu->getKernel(*module, "kNMLQuadraticMinimize1_kernel");
	cu->executeKernel(quadmin1Kernel, quadmin1Args, cu->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize, cu->ThreadBlockSize*sizeof(mixed));
	mixed oneoverEig = (mixed)1.0f/maxEigenvalue;
	void* quadmin2Args[] = {&atoms, &currentPE, &lastPE, &oneoverEig, &cu->getPosq().getDevicePointer(), &cu->getPosqCorrection().getDevicePointer(), &oldpos.getDevicePointer(), &oldposCorrection.getDevicePointer(), &cu->getVelm().getDevicePointer(), &slopeBuffer.getDevicePointer(), &lambdaval.getDevicePointer()}; 
        quadmin2Kernel = cu->getKernel(*module, "kNMLQuadraticMinimize2_kernel");
        cu->executeKernel(quadmin2Kernel, quadmin2Args, cu->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize, cu->getNumThreadBlocks()*cu->ThreadBlockSize*sizeof(mixed)); 
	
}
