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

#include "DILUSmoother.H"
#include "DILUPreconditioner.H"
#include "PrecisionAdaptor.H"
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
    defineTypeNameAndDebug(DILUSmoother, 0);

    lduMatrix::smoother::addasymMatrixConstructorToTable<DILUSmoother>
        addDILUSmootherAsymMatrixConstructorToTable_;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::DILUSmoother::DILUSmoother
(
    const word& fieldName,
    const lduMatrix& matrix,
    const FieldField<Field, scalar>& interfaceBouCoeffs,
    const FieldField<Field, scalar>& interfaceIntCoeffs,
    const lduInterfaceFieldPtrsList& interfaces
)
:
    lduMatrix::smoother
    (
        fieldName,
        matrix,
        interfaceBouCoeffs,
        interfaceIntCoeffs,
        interfaces
    ),
    rD_(matrix_.diag().size())
{
    const scalarField& diag = matrix_.diag();
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
    DILUPreconditioner::calcReciprocalD(rD_, matrix_);
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::DILUSmoother::smooth
(
    solveScalarField& psi,
    const scalarField& source,
    const direction cmpt,
    const label nSweeps
) const
{
    const solveScalar* const __restrict__ rDPtr = rD_.begin();

    const label* const __restrict__ uPtr =
        matrix_.lduAddr().upperAddr().begin();
    const label* const __restrict__ lPtr =
        matrix_.lduAddr().lowerAddr().begin();

    const scalar* const __restrict__ upperPtr = matrix_.upper().begin();
    const scalar* const __restrict__ lowerPtr = matrix_.lower().begin();

    // Temporary storage for the residual
    solveScalarField rA(rD_.size());
    solveScalar* __restrict__ rAPtr = rA.begin();

    for (label sweep=0; sweep<nSweeps; sweep++)
    {
        matrix_.residual
        (
            rA,
            psi,
            source,
            interfaceBouCoeffs_,
            interfaces_,
            cmpt
        );

        const label nCells=rA.size();
        const label nFaces = matrix_.upper().size();
        const label nFacesM1 = nFaces - 1;
    #ifdef USE_OMP
        solveScalarField rA_temp(rA.size());
        solveScalar* __restrict__ rA_temp_Ptr = rA_temp.begin();

        #pragma omp target teams distribute parallel for if (nCells > THRESHOLD_HIGH)
        for (label cell=0; cell<nCells; cell++)
        {
            rAPtr[cell] *= rDPtr[cell];
            rA_temp_Ptr[cell] = rAPtr[cell];
        }
        
        #pragma omp target teams distribute parallel for if (nFaces > THRESHOLD_LOW)
        for (label face=0; face<nFaces; face++)
        {
            const label u = uPtr[face];
            atomicAccumulator(rA_temp_Ptr[u]) -= rDPtr[u]*lowerPtr[face]*rAPtr[lPtr[face]];
        }
        
        #pragma omp target teams distribute parallel for if (nFacesM1 > THRESHOLD_LOW)
        for (label face=nFacesM1; face>=0; face--)
        {
            const label l = lPtr[face];
            atomicAccumulator(rAPtr[l]) -= rDPtr[l]*upperPtr[face]*rA_temp_Ptr[uPtr[face]];
        }
    #else
        for (label cell=0; cell<nCells; cell++)
        {
            rAPtr[cell] *= rDPtr[cell];
        }
        
        for (label face=0; face<nFaces; face++)
        {
            const label u = uPtr[face];
            rAPtr[u] -= rDPtr[u]*lowerPtr[face]*rAPtr[lPtr[face]];
        }
        
        for (label face=nFacesM1; face>=0; face--)
        {
            const label l = lPtr[face];
            rAPtr[l] -= rDPtr[l]*upperPtr[face]*rAPtr[uPtr[face]];
        }
    #endif
        psi += rA;
    }
}


void Foam::DILUSmoother::scalarSmooth
(
    solveScalarField& psi,
    const solveScalarField& source,
    const direction cmpt,
    const label nSweeps
) const
{
    smooth
    (
        psi,
        ConstPrecisionAdaptor<scalar, solveScalar>(source),
        cmpt,
        nSweeps
    );
}


// ************************************************************************* //
