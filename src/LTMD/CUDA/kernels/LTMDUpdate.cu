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

#include "kernels/gputypes.h"
#include <stdio.h>
#include <cuda.h>
#include <vector_functions.h>
#include <cstdlib>
#include <string>
#include <iostream>
using namespace std;

typedef float Real;

__global__ void kNMLUpdate1_kernel( int numAtoms, int paddedNumAtoms, float tau, float dt, float kT, float4 *posq, float4 *posqP, float4 *velm, float4 *force,
									float4 *random, int *randomPosition, int totalRandoms ) {
	// Update the velocity.
	const Real vscale = exp( -dt / tau );
	const Real fscale = ( 1.0f - vscale ) * tau;
	const Real noisescale = sqrt( kT * ( 1 - vscale * vscale ) );

	int rpos = randomPosition[blockIdx.x];
	for( int atom = threadIdx.x + blockIdx.x * blockDim.x; atom < numAtoms; atom += blockDim.x * gridDim.x ) {
		const float4 n = random[rpos + atom];
		const float4 randomNoise = make_float4( n.x * noisescale, n.y * noisescale, n.z * noisescale, n.w * noisescale );

		const Real sqrtInvMass = sqrt( velm[atom].w );

		float4 v = velm[atom];
		float4 f = force[atom];

		v.x = ( vscale * v.x ) + ( fscale * f.x * v.w ) + ( randomNoise.x * sqrtInvMass );
		v.y = ( vscale * v.y ) + ( fscale * f.y * v.w ) + ( randomNoise.y * sqrtInvMass );
		v.z = ( vscale * v.z ) + ( fscale * f.z * v.w ) + ( randomNoise.z * sqrtInvMass );

		velm[atom] = v;
	}

	if( threadIdx.x == 0 ) {
		rpos += paddedNumAtoms;
		if( rpos > totalRandoms ) {
			rpos -= totalRandoms;
		}
		randomPosition[blockIdx.x] = rpos;
	}
}

__global__ void kNMLUpdate2_kernel( int numAtoms, int numModes, float4 *velm, float4 *modes, float *modeWeights ) {
	extern __shared__ float dotBuffer[];
	for( int mode = blockIdx.x; mode < numModes; mode += gridDim.x ) {
		// Compute the projection of the mass weighted velocity onto one normal mode vector.
		Real dot = 0.0f;

		for( int atom = threadIdx.x; atom < numAtoms; atom += blockDim.x ) {
			const int modePos = mode * numAtoms + atom;
			const Real scale = 1.0f / sqrt( velm[atom].w );

			float4 v = velm[atom];
			float4 m = modes[modePos];

			dot += scale * ( v.x * m.x + v.y * m.y + v.z * m.z );
		}

		dotBuffer[threadIdx.x] = dot;

		__syncthreads();
		if( threadIdx.x == 0 ) {
			Real sum = 0;
			for( int i = 0; i < blockDim.x; i++ ) {
				sum += dotBuffer[i];
			}
			modeWeights[mode] = sum;
		}
	}
}

__global__ void kNMLUpdate3_kernel( int numAtoms, int numModes, float dt, float4 *posq, float4 *velm, float4 *modes, float *modeWeights, float4 *noiseVal ) {
	// Load the weights into shared memory.
	extern __shared__ float weightBuffer[];
	for( int mode = threadIdx.x; mode < numModes; mode += blockDim.x ) {
		weightBuffer[mode] = modeWeights[mode];
	}
	__syncthreads();

	// Compute the projected velocities and update the atom positions.
	for( int atom = threadIdx.x + blockIdx.x * blockDim.x; atom < numAtoms; atom += blockDim.x * gridDim.x ) {
		const Real invMass = velm[atom].w, scale = sqrt( invMass );

		float3 v = make_float3( 0.0f, 0.0f, 0.0f );
		for( int mode = 0; mode < numModes; mode++ ) {
			float4 m = modes[mode * numAtoms + atom];
			float weight = weightBuffer[mode];
			v.x += m.x * weight;
			v.y += m.y * weight;
			v.z += m.z * weight;
		}

		v.x *= scale;
		v.y *= scale;
		v.z *= scale;
		velm[atom] = make_float4( v.x, v.y, v.z, invMass );

		float4 pos = posq[atom];

		// Add Step
		pos.x += dt * v.x;
		pos.y += dt * v.y;
		pos.z += dt * v.z;

#ifdef FAST_NOISE
		// Remove Noise
		pos.x -= noiseVal[atom].x;
		pos.y -= noiseVal[atom].y;
		pos.z -= noiseVal[atom].z;
#endif

		posq[atom] = pos;
	}
}

extern void kGenerateRandoms( gpuContext gpu );
void kNMLUpdate( gpuContext gpu, int numModes, CUDAStream<float4>& modes, CUDAStream<float>& modeWeights, CUDAStream<float4>& noiseVal ) {
	kNMLUpdate1_kernel <<< gpu->sim.blocks, gpu->sim.update_threads_per_block >>> ( gpu->natoms, gpu->sim.paddedNumberOfAtoms,
		    gpu->sim.tau, gpu->sim.deltaT, gpu->sim.kT, gpu->sim.pPosq, noiseVal._pDevData, gpu->sim.pVelm4, gpu->sim.pForce4, gpu->sim.pRandom4, gpu->sim.pRandomPosition, gpu->sim.randoms );
	LAUNCHERROR( "kNMLUpdate1" );
	kNMLUpdate2_kernel <<< gpu->sim.blocks, gpu->sim.update_threads_per_block, gpu->sim.update_threads_per_block *sizeof( float ) >>> ( gpu->natoms,
			numModes, gpu->sim.pVelm4, modes._pDevData, modeWeights._pDevData );
	LAUNCHERROR( "kNMLUpdate2" );
	kNMLUpdate3_kernel <<< gpu->sim.blocks, gpu->sim.update_threads_per_block, numModes *sizeof( float ) >>> ( gpu->natoms, numModes,
			gpu->sim.deltaT, gpu->sim.pPosq, gpu->sim.pVelm4, modes._pDevData, modeWeights._pDevData, noiseVal._pDevData );
	LAUNCHERROR( "kNMLUpdate3" );

	// Update randoms if necessary
	gpu->iterations++;
	if( gpu->iterations == gpu->sim.randomIterations ) {
		kGenerateRandoms( gpu );
		gpu->iterations = 0;
	}
}

#ifdef FAST_NOISE
__global__ void kFastNoise1_kernel( int numAtoms, int paddedNumAtoms, int numModes, float kT, float4 *noiseVal, float4 *velm, float4 *modes, float *modeWeights, float4 *random, int *randomPosition, int totalRandoms, float maxEigenvalue, float stepSize ) {
	extern __shared__ float dotBuffer[];
	const Real val = stepSize / 0.002;
	const Real noisescale = sqrt( 2 * kT * 1.0f / maxEigenvalue );

	int rpos = randomPosition[blockIdx.x];
	for( int mode = blockIdx.x; mode < numModes; mode += gridDim.x ) {
		Real dot = 0.0f;

		for( int atom = threadIdx.x; atom < numAtoms; atom += blockDim.x ) {
			const float4 n = random[rpos + atom];
			const float4 randomNoise = make_float4( n.x * noisescale, n.y * noisescale, n.z * noisescale, n.w * noisescale );

			noiseVal[atom] = randomNoise;

			float4 m = modes[mode * numAtoms + atom];
			dot += randomNoise.x * m.x + randomNoise.y * m.y + randomNoise.z * m.z;
		}

		dotBuffer[threadIdx.x] = dot;

		__syncthreads();
		if( threadIdx.x == 0 ) {
			Real sum = 0;
			for( int i = 0; i < blockDim.x; i++ ) {
				sum += dotBuffer[i];
			}
			modeWeights[mode] = sum;

			rpos += paddedNumAtoms;
			if( rpos > totalRandoms ) {
				rpos -= totalRandoms;
			}
			randomPosition[blockIdx.x] = rpos;
		}
	}
}

__global__ void kFastNoise2_kernel( int numAtoms, int numModes, float4 *posq, float4 *noiseVal, float4 *velm, float4 *modes, float *modeWeights ) {
	// Load the weights into shared memory.
	extern __shared__ float weightBuffer[];
	for( int mode = threadIdx.x; mode < numModes; mode += blockDim.x ) {
		weightBuffer[mode] = modeWeights[mode];
	}
	__syncthreads();

	// Compute the projected forces and update the atom positions.
	for( int atom = threadIdx.x + blockIdx.x * blockDim.x; atom < numAtoms; atom += blockDim.x * gridDim.x ) {
		const Real invMass = velm[atom].w, sqrtInvMass = sqrt( invMass );

		float3 r = make_float3( 0.0f, 0.0f, 0.0f );
		for( int mode = 0; mode < numModes; mode++ ) {
			float4 m = modes[mode * numAtoms + atom];
			float weight = weightBuffer[mode];
			r.x += m.x * weight;
			r.y += m.y * weight;
			r.z += m.z * weight;
		}

		/*r.x *= sqrtInvMass;
		r.y *= sqrtInvMass;
		r.z *= sqrtInvMass;*/
		noiseVal[atom] = make_float4( noiseVal[atom].x - r.x, noiseVal[atom].y - r.y, noiseVal[atom].z - r.z, 0.0f );
		noiseVal[atom].x *= sqrtInvMass;
		noiseVal[atom].y *= sqrtInvMass;
		noiseVal[atom].z *= sqrtInvMass;

		float4 pos = posq[atom];
		pos.x += noiseVal[atom].x;
		pos.y += noiseVal[atom].y;
		pos.z += noiseVal[atom].z;
		posq[atom] = pos;
	}
}

void kFastNoise( gpuContext gpu, int numModes, CUDAStream<float4>& modes, CUDAStream<float>& modeWeights, float maxEigenvalue, CUDAStream<float4>& noiseVal, float stepSize ) {
	kFastNoise1_kernel <<< gpu->sim.blocks, gpu->sim.update_threads_per_block, gpu->sim.update_threads_per_block *sizeof( float ) >>> (
		gpu->natoms, gpu->sim.paddedNumberOfAtoms, numModes, gpu->sim.kT, gpu->sim.pPosqP, gpu->sim.pVelm4, modes._pDevData, modeWeights._pDevData, gpu->sim.pRandom4, gpu->sim.pRandomPosition, gpu->sim.randoms, maxEigenvalue, stepSize
	);
	LAUNCHERROR( "kFastNoise1" );
	kFastNoise2_kernel <<< gpu->sim.blocks, gpu->sim.update_threads_per_block, numModes *sizeof( float ) >>> (
		gpu->natoms, numModes, gpu->sim.pPosq, noiseVal._pDevData, gpu->sim.pVelm4, modes._pDevData, modeWeights._pDevData
	);
	LAUNCHERROR( "kFastNoise2" );

	// Update randoms if necessary
	gpu->iterations++;
	if( gpu->iterations == gpu->sim.randomIterations ) {
		kGenerateRandoms( gpu );
		gpu->iterations = 0;
	}
}
#endif 

__global__ void kRejectMinimizationStep_kernel( int numAtoms, float4 *posq, float4 *oldPosq ) {
	for( int atom = threadIdx.x + blockIdx.x * blockDim.x; atom < numAtoms; atom += blockDim.x * gridDim.x ) {
		posq[atom] = oldPosq[atom];
	}
}

void kNMLRejectMinimizationStep( gpuContext gpu ) {
	kRejectMinimizationStep_kernel <<< gpu->sim.blocks, gpu->sim.update_threads_per_block >>> ( gpu->natoms, gpu->sim.pPosq, gpu->sim.pOldPosq );
	LAUNCHERROR( "kRejectMinimizationStep" );
}

__global__ void kAcceptMinimizationStep_kernel( int numAtoms, float4 *posq, float4 *oldPosq ) {
	for( int atom = threadIdx.x + blockIdx.x * blockDim.x; atom < numAtoms; atom += blockDim.x * gridDim.x ) {
		oldPosq[atom] = posq[atom];
	}
}

void kNMLAcceptMinimizationStep( gpuContext gpu ) {
	kAcceptMinimizationStep_kernel <<< gpu->sim.blocks, gpu->sim.update_threads_per_block >>> ( gpu->natoms, gpu->sim.pPosq, gpu->sim.pOldPosq );
	LAUNCHERROR( "kAcceptMinimizationStep" );
}

__global__ void kNMLLinearMinimize1_kernel( int numAtoms, int numModes, float4 *velm, float4 *force, float4 *modes, float *modeWeights ) {
	extern __shared__ float dotBuffer[];
	for( int mode = blockIdx.x; mode < numModes; mode += gridDim.x ) {
		// Compute the projection of the mass weighted force onto one normal mode vector.
		Real dot = 0.0f;
		for( int atom = threadIdx.x; atom < numAtoms; atom += blockDim.x ) {
			const Real scale = sqrt( velm[atom].w );
			const int modePos = mode * numAtoms + atom;

			float4 f = force[atom];
			float4 m = modes[modePos];

			dot += scale * ( f.x * m.x + f.y * m.y + f.z * m.z );
		}
		dotBuffer[threadIdx.x] = dot;

		__syncthreads();
		if( threadIdx.x == 0 ) {
			Real sum = 0;
			for( int i = 0; i < blockDim.x; i++ ) {
				sum += dotBuffer[i];
			}
			modeWeights[mode] = sum;
		}
	}
}

__global__ void kNMLLinearMinimize2_kernel( int numAtoms, int numModes, float invMaxEigen, float4 *posq, float4 *posqP, float4 *velm, float4 *force, float4 *modes, float *modeWeights ) {
	// Load the weights into shared memory.
	extern __shared__ float weightBuffer[];
	for( int mode = threadIdx.x; mode < numModes; mode += blockDim.x ) {
		weightBuffer[mode] = modeWeights[mode];
	}
	__syncthreads();

	// Compute the projected forces and update the atom positions.
	for( int atom = threadIdx.x + blockIdx.x * blockDim.x; atom < numAtoms; atom += blockDim.x * gridDim.x ) {
		const Real invMass = velm[atom].w, sqrtInvMass = sqrt( invMass ), factor = invMass * invMaxEigen;

		float3 f = make_float3( 0.0f, 0.0f, 0.0f );
		for( int mode = 0; mode < numModes; mode++ ) {
			float4 m = modes[mode * numAtoms + atom];
			float weight = weightBuffer[mode];
			f.x += m.x * weight;
			f.y += m.y * weight;
			f.z += m.z * weight;
		}

		f.x *= sqrtInvMass;
		f.y *= sqrtInvMass;
		f.z *= sqrtInvMass;
		posqP[atom] = make_float4( force[atom].x - f.x, force[atom].y - f.y, force[atom].z - f.z, 0.0f );

		float4 pos = posq[atom];
		pos.x += factor * posqP[atom].x;
		pos.y += factor * posqP[atom].y;
		pos.z += factor * posqP[atom].z;
		posq[atom] = pos;
	}
}

void kNMLLinearMinimize( gpuContext gpu, int numModes, float maxEigenvalue, CUDAStream<float4>& modes, CUDAStream<float>& modeWeights ) {
	kNMLLinearMinimize1_kernel <<< gpu->sim.blocks, gpu->sim.update_threads_per_block, gpu->sim.update_threads_per_block *sizeof( float ) >>> ( gpu->natoms,
			numModes, gpu->sim.pVelm4, gpu->sim.pForce4, modes._pDevData, modeWeights._pDevData );
	LAUNCHERROR( "kNMLLinearMinimize1" );
	kNMLLinearMinimize2_kernel <<< gpu->sim.blocks, gpu->sim.update_threads_per_block, numModes *sizeof( float ) >>> ( gpu->natoms, numModes,
			1.0f / maxEigenvalue, gpu->sim.pPosq, gpu->sim.pPosqP, gpu->sim.pVelm4, gpu->sim.pForce4, modes._pDevData, modeWeights._pDevData );
	LAUNCHERROR( "kNMLLinearMinimize2" );
}

__global__ void kNMLQuadraticMinimize1_kernel( int numAtoms, float4 *posqP, float4 *velm, float4 *force, float *blockSlope ) {
	// Compute the slope along the minimization direction.
	extern __shared__ float slopeBuffer[];

	Real slope = 0.0f;
	for( int atom = threadIdx.x + blockIdx.x * blockDim.x; atom < numAtoms; atom += blockDim.x * gridDim.x ) {
		const Real invMass = velm[atom].w;
		const float4 xp = posqP[atom];
		const float4 f = force[atom];

		slope -= invMass * ( xp.x * f.x + xp.y * f.y + xp.z * f.z );
	}
	slopeBuffer[threadIdx.x] = slope;
	__syncthreads();
	if( threadIdx.x == 0 ) {
		for( int i = 1; i <  blockDim.x; i++ ) {
			slope += slopeBuffer[i];
		}
		blockSlope[blockIdx.x] = slope;
	}
}

__global__ void kNMLQuadraticMinimize2_kernel( int numAtoms, float currentPE, float lastPE, float invMaxEigen, float4 *posq, float4 *posqP, float4 *velm, float *blockSlope, float *lambdaval ) {
	// Load the block contributions into shared memory.
	extern __shared__ float slopeBuffer[];
	for( int block = threadIdx.x; block < gridDim.x; block += blockDim.x ) {
		slopeBuffer[block] = blockSlope[block];
	}

	__syncthreads();

	// Compute the scaling coefficient.
	if( threadIdx.x == 0 ) {
		Real slope = 0.0f;
		for( int i = 0; i < gridDim.x; i++ ) {
			slope += slopeBuffer[i];
		}
		Real lambda = invMaxEigen;
		Real oldLambda = lambda;
		Real a = ( ( ( lastPE - currentPE ) / oldLambda + slope ) / oldLambda );

		if( a != 0.0f ) {
			const Real b = slope - 2.0f * a * oldLambda;
			lambda = -b / ( 2.0f * a );
		} else {
			lambda = 0.5f * oldLambda;
		}

		if( lambda <= 0.0f ) {
			lambda = 0.5f * oldLambda;
		}

		slopeBuffer[0] = lambda - oldLambda;

		// Store variables for retrival
		lambdaval[0] = lambda;
	}

	__syncthreads();

	// Remove previous position update (-oldLambda) and add new move (lambda).
	const Real dlambda = slopeBuffer[0];
	for( int atom = threadIdx.x + blockIdx.x * blockDim.x; atom < numAtoms; atom += blockDim.x * gridDim.x ) {
		const Real factor = velm[atom].w * dlambda;

		float4 pos = posq[atom];
		pos.x += factor * posqP[atom].x;
		pos.y += factor * posqP[atom].y;
		pos.z += factor * posqP[atom].z;
		posq[atom] = pos;
	}
}

void kNMLQuadraticMinimize( gpuContext gpu, float maxEigenvalue, float currentPE, float lastPE, CUDAStream<float>& slopeBuffer, CUDAStream<float>& lambdaval ) {
	kNMLQuadraticMinimize1_kernel <<< gpu->sim.blocks, gpu->sim.update_threads_per_block, gpu->sim.update_threads_per_block *sizeof( float ) >>> ( gpu->natoms,
			gpu->sim.pPosqP, gpu->sim.pVelm4, gpu->sim.pForce4, slopeBuffer._pDevData );
	LAUNCHERROR( "kNMLQuadraticMinimize1" );
	kNMLQuadraticMinimize2_kernel <<< gpu->sim.blocks, gpu->sim.update_threads_per_block, gpu->sim.blocks *sizeof( float ) >>> ( gpu->natoms, currentPE,
			lastPE, 1.0f / maxEigenvalue, gpu->sim.pPosq, gpu->sim.pPosqP, gpu->sim.pVelm4, slopeBuffer._pDevData, lambdaval._pDevData );
	LAUNCHERROR( "kNMLQuadraticMinimize2" );
}
