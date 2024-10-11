/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2015 OpenFOAM Foundation
    Copyright (C) 2019 OpenCFD Ltd.
    Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "DICPreconditioner.H"
#include <algorithm>

#ifdef USE_OMP
#include <omp.h>
    #ifndef OMP_UNIFIED_MEMORY_REQUIRED
    #define OMP_UNIFIED_MEMORY_REQUIRED
    #pragma omp requires unified_shared_memory
    #endif

#include "AtomicAccumulator.H"
#endif

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(DICPreconditioner, 0);

    lduMatrix::preconditioner::
        addsymMatrixConstructorToTable<DICPreconditioner>
        addDICPreconditionerSymMatrixConstructorToTable_;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::DICPreconditioner::DICPreconditioner
(
    const lduMatrix::solver& sol,
    const dictionary&
)
:
    lduMatrix::preconditioner(sol),
    rD_(sol.matrix().diag().size())
{
    const scalarField& diag = sol.matrix().diag();
#ifdef USE_OMP
    const label loop_len = diag.size();
    const solveScalar* __restrict__ diagPtr = diag.begin();
    solveScalar* __restrict__ rD_Ptr = rD_.begin();

    #pragma omp target teams distribute parallel for if (target:loop_len>20000)
    for (label i = 0; i < loop_len; ++i)
    {
        rD_Ptr[i] = diagPtr[i];
    }    
#else
    std::copy(diag.begin(), diag.end(), rD_.begin());
#endif
    calcReciprocalD(rD_, sol.matrix());
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::DICPreconditioner::calcReciprocalD
(
    solveScalarField& rD,
    const lduMatrix& matrix
)
{
    solveScalar* __restrict__ rDPtr = rD.begin();

    const label* const __restrict__ uPtr = matrix.lduAddr().upperAddr().begin();
    const label* const __restrict__ lPtr = matrix.lduAddr().lowerAddr().begin();
    const scalar* const __restrict__ upperPtr = matrix.upper().begin();

    // Calculate the DIC diagonal
    const label nFaces = matrix.upper().size();
    const label nCells = rD.size();
#ifdef USE_OMP
    solveScalarField rD_temp(rD.size());
    solveScalar* __restrict__ rD_temp_Ptr = rD_temp.begin();

    #pragma omp target teams distribute parallel for if (target:nCells>20000)
    for (label cell=0; cell<nCells; cell++)
    {
        rD_temp_Ptr[cell] = 0.0;
    }

    // Calculate the sum[cell] += U[cell][j]*U[j][cell]/D[j]
    #pragma omp target teams distribute parallel for if (target:nFaces>10000)
    for (label face=0; face<nFaces; face++)
    {
        atomicAccumulator(rD_temp_Ptr[uPtr[face]]) += upperPtr[face]*upperPtr[face]/rDPtr[lPtr[face]];
    }

    // Calculate the reciprocal of the preconditioned diagonal
    // inv_D [cell] = 1/(D[cell] - sum[cell])
    #pragma omp target teams distribute parallel for if (target:nCells>20000)
    for (label cell=0; cell<nCells; cell++)
    {
        rDPtr[cell] = 1.0/(rDPtr[cell] - rD_temp_Ptr[cell]);
    }
#else
    for (label face=0; face<nFaces; face++)
    {
        rDPtr[uPtr[face]] -= upperPtr[face]*upperPtr[face]/rDPtr[lPtr[face]];
    }

    // Calculate the reciprocal of the preconditioned diagonal
    for (label cell=0; cell<nCells; cell++)
    {
        rDPtr[cell] = 1.0/rDPtr[cell];
    }
#endif
}


void Foam::DICPreconditioner::precondition
(
    solveScalarField& wA,
    const solveScalarField& rA,
    const direction
) const
{
    solveScalar* __restrict__ wAPtr = wA.begin();
    const solveScalar* __restrict__ rAPtr = rA.begin();
    const solveScalar* __restrict__ rDPtr = rD_.begin();

    const label* const __restrict__ uPtr =
        solver_.matrix().lduAddr().upperAddr().begin();
    const label* const __restrict__ lPtr =
        solver_.matrix().lduAddr().lowerAddr().begin();
    const scalar* const __restrict__ upperPtr =
        solver_.matrix().upper().begin();

    const label nCells = wA.size();
    const label nFaces = solver_.matrix().upper().size();
    const label nFacesM1 = nFaces - 1;

#ifdef USE_OMP
    solveScalarField wA_temp(wA.size());
    solveScalar* __restrict__ wA_temp_Ptr = wA_temp.begin();

    #pragma omp target teams distribute parallel for if (target:nCells>20000)
    for (label cell=0; cell<nCells; cell++)
    {
        wAPtr[cell] = rDPtr[cell]*rAPtr[cell];
        wA_temp_Ptr[cell] = wAPtr[cell];
    }

    #pragma omp target teams distribute parallel for if (target:nFaces>10000)
    for (label face=0; face<nFaces; face++)
    {
        atomicAccumulator(wA_temp_Ptr[uPtr[face]]) -= rDPtr[uPtr[face]]*upperPtr[face]*wAPtr[lPtr[face]];
    }

    #pragma omp target teams distribute parallel for if (target:nFacesM1>10000)
    for (label face=nFacesM1; face>=0; face--)
    {
        atomicAccumulator(wAPtr[lPtr[face]]) -= rDPtr[lPtr[face]]*upperPtr[face]*wA_temp_Ptr[uPtr[face]];
    }
#else
    for (label cell=0; cell<nCells; cell++)
    {
        wAPtr[cell] = rDPtr[cell]*rAPtr[cell];
    }

    for (label face=0; face<nFaces; face++)
    {
        wAPtr[uPtr[face]] -= rDPtr[uPtr[face]]*upperPtr[face]*wAPtr[lPtr[face]];
    }

    for (label face=nFacesM1; face>=0; face--)
    {
        wAPtr[lPtr[face]] -= rDPtr[lPtr[face]]*upperPtr[face]*wAPtr[uPtr[face]];
    }
#endif
}


// ************************************************************************* //
