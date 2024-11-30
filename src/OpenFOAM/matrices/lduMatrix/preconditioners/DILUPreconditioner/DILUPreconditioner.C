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

#include "DILUPreconditioner.H"
#include <algorithm>

#ifdef USE_OMP
#include <omp.h>
    #ifndef OMP_UNIFIED_MEMORY_REQUIRED
    #define OMP_UNIFIED_MEMORY_REQUIRED
    #pragma omp requires unified_shared_memory
    #endif

#include "AtomicAccumulator.H"
#include "macros.H"
#endif

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(DILUPreconditioner, 0);

    lduMatrix::preconditioner::
        addasymMatrixConstructorToTable<DILUPreconditioner>
        addDILUPreconditionerAsymMatrixConstructorToTable_;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::DILUPreconditioner::DILUPreconditioner
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

    #pragma omp target teams distribute parallel for if (loop_len > THRESHOLD_HIGH)
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

void Foam::DILUPreconditioner::calcReciprocalD
(
    solveScalarField& rD,
    const lduMatrix& matrix
)
{
    solveScalar* __restrict__ rDPtr = rD.begin();

    const label* const __restrict__ uPtr = matrix.lduAddr().upperAddr().begin();
    const label* const __restrict__ lPtr = matrix.lduAddr().lowerAddr().begin();

    const scalar* const __restrict__ upperPtr = matrix.upper().begin();
    const scalar* const __restrict__ lowerPtr = matrix.lower().begin();

    const label nFaces = matrix.upper().size();
    const label nCells = rD.size();
#ifdef USE_OMP
    solveScalarField rD_temp(rD.size());
    solveScalar* __restrict__ rD_temp_Ptr = rD_temp.begin();

    #pragma omp target teams distribute parallel for if (nCells > THRESHOLD_HIGH)
    for (label cell=0; cell<nCells; cell++)
    {
        rD_temp_Ptr[cell] = 0.0;
    }
    // Calculate the sum[cell] += U[cell][j]*L[j][cell]/D[j]
    #pragma omp target teams distribute parallel for if (nFaces > THRESHOLD_LOW)
    for (label face=0; face<nFaces; face++)
    {
        atomicAccumulator(rD_temp_Ptr[uPtr[face]]) += upperPtr[face]*lowerPtr[face]/rDPtr[lPtr[face]];
    }

    // Calculate the reciprocal of the preconditioned diagonal
    // inv_D [cell] = 1/(D[cell] - sum[cell])
    #pragma omp target teams distribute parallel for if (nCells > THRESHOLD_HIGH)
    for (label cell=0; cell<nCells; cell++)
    {
        rDPtr[cell] = 1.0/(rDPtr[cell] - rD_temp_Ptr[cell]);
    }
#else
    for (label face=0; face<nFaces; face++)
    {
        rDPtr[uPtr[face]] -= upperPtr[face]*lowerPtr[face]/rDPtr[lPtr[face]];
    }

    // Calculate the reciprocal of the preconditioned diagonal
    for (label cell=0; cell<nCells; cell++)
    {
        rDPtr[cell] = 1.0/rDPtr[cell];
    }
#endif
}


void Foam::DILUPreconditioner::precondition
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
    const label* const __restrict__ losortPtr =
        solver_.matrix().lduAddr().losortAddr().begin();

    const scalar* const __restrict__ upperPtr =
        solver_.matrix().upper().begin();
    const scalar* const __restrict__ lowerPtr =
        solver_.matrix().lower().begin();

    const label nCells = wA.size();
    const label nFaces = solver_.matrix().upper().size();
    const label nFacesM1 = nFaces - 1;

#ifdef USE_OMP
    solveScalarField wA_temp(wA.size());
    solveScalar* __restrict__ wA_temp_Ptr = wA_temp.begin();

    #pragma omp target teams distribute parallel for if (nCells > THRESHOLD_HIGH)
    for (label cell=0; cell<nCells; cell++)
    {
        wAPtr[cell] = rDPtr[cell]*rAPtr[cell];
        wA_temp_Ptr[cell] = wAPtr[cell];
    }

    #pragma omp target teams distribute parallel for if (nFaces > THRESHOLD_LOW)
    for (label face=0; face<nFaces; face++)
    {
        const label sface = losortPtr[face];
        atomicAccumulator(wA_temp_Ptr[uPtr[sface]]) -=
            rDPtr[uPtr[sface]]*lowerPtr[sface]*wAPtr[lPtr[sface]];
    }

    #pragma omp target teams distribute parallel for if (nFacesM1 > THRESHOLD_LOW)
    for (label face=nFacesM1; face>=0; face--)
    {
        const label lptr_index = lPtr[face];
        atomicAccumulator(wAPtr[lptr_index]) -=
            rDPtr[lptr_index]*upperPtr[face]*wA_temp_Ptr[uPtr[face]];
    }
#else
    for (label cell=0; cell<nCells; cell++)
    {
        wAPtr[cell] = rDPtr[cell]*rAPtr[cell];
    }

    for (label face=0; face<nFaces; face++)
    {
        const label sface = losortPtr[face];
        wAPtr[uPtr[sface]] -=
            rDPtr[uPtr[sface]]*lowerPtr[sface]*wAPtr[lPtr[sface]];
    }

    for (label face=nFacesM1; face>=0; face--)
    {
        wAPtr[lPtr[face]] -=
            rDPtr[lPtr[face]]*upperPtr[face]*wAPtr[uPtr[face]];
    }
#endif
}


void Foam::DILUPreconditioner::preconditionT
(
    solveScalarField& wT,
    const solveScalarField& rT,
    const direction
) const
{
    solveScalar* __restrict__ wTPtr = wT.begin();
    const solveScalar* __restrict__ rTPtr = rT.begin();
    const solveScalar* __restrict__ rDPtr = rD_.begin();

    const label* const __restrict__ uPtr =
        solver_.matrix().lduAddr().upperAddr().begin();
    const label* const __restrict__ lPtr =
        solver_.matrix().lduAddr().lowerAddr().begin();
    const label* const __restrict__ losortPtr =
        solver_.matrix().lduAddr().losortAddr().begin();

    const scalar* const __restrict__ upperPtr =
        solver_.matrix().upper().begin();
    const scalar* const __restrict__ lowerPtr =
        solver_.matrix().lower().begin();

    const label nCells = wT.size();
    const label nFaces = solver_.matrix().upper().size();
    const label nFacesM1 = nFaces - 1;

#ifdef USE_OMP
    solveScalarField wT_temp(wT.size());
    solveScalar* __restrict__ wT_temp_Ptr = wT_temp.begin();
    
    #pragma omp target teams distribute parallel for if (nCells > THRESHOLD_HIGH)
    for (label cell=0; cell<nCells; cell++)
    {
        wTPtr[cell] = rDPtr[cell]*rTPtr[cell];
        wT_temp_Ptr[cell] = wTPtr[cell];
    }

    #pragma omp target teams distribute parallel for if (nFaces > THRESHOLD_LOW)
    for (label face=0; face<nFaces; face++)
    {
        atomicAccumulator(wT_temp_Ptr[uPtr[face]]) -=
            rDPtr[uPtr[face]]*upperPtr[face]*wTPtr[lPtr[face]];
    }

    #pragma omp target teams distribute parallel for if (nFacesM1 > THRESHOLD_LOW)
    for (label face=nFacesM1; face>=0; face--)
    {
        const label sface = losortPtr[face];
        atomicAccumulator(wTPtr[lPtr[sface]]) -=
            rDPtr[lPtr[sface]]*lowerPtr[sface]*wT_temp_Ptr[uPtr[sface]];
    }
#else
    for (label cell=0; cell<nCells; cell++)
    {
        wTPtr[cell] = rDPtr[cell]*rTPtr[cell];
    }

    for (label face=0; face<nFaces; face++)
    {
        wTPtr[uPtr[face]] -=
            rDPtr[uPtr[face]]*upperPtr[face]*wTPtr[lPtr[face]];
    }


    for (label face=nFacesM1; face>=0; face--)
    {
        const label sface = losortPtr[face];
        wTPtr[lPtr[sface]] -=
            rDPtr[lPtr[sface]]*lowerPtr[sface]*wTPtr[uPtr[sface]];
    }
#endif
}


// ************************************************************************* //
