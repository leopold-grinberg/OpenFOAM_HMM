/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2015 OpenFOAM Foundation
    Copyright (C) 2017-2019 OpenCFD Ltd.
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

#include "GaussSeidelSmoother.H"
#include "PrecisionAdaptor.H"

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
    defineTypeNameAndDebug(GaussSeidelSmoother, 0);

    lduMatrix::smoother::addsymMatrixConstructorToTable<GaussSeidelSmoother>
        addGaussSeidelSmootherSymMatrixConstructorToTable_;

    lduMatrix::smoother::addasymMatrixConstructorToTable<GaussSeidelSmoother>
        addGaussSeidelSmootherAsymMatrixConstructorToTable_;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::GaussSeidelSmoother::GaussSeidelSmoother
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
    )
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::GaussSeidelSmoother::smooth
(
    const word& fieldName_,
    solveScalarField& psi,
    const lduMatrix& matrix_,
    const solveScalarField& source,
    const FieldField<Field, scalar>& interfaceBouCoeffs_,
    const lduInterfaceFieldPtrsList& interfaces_,
    const direction cmpt,
    const label nSweeps
)
{
    solveScalar* __restrict__ psiPtr = psi.begin();

    const label nCells = psi.size();

    solveScalarField bPrime(nCells);
    solveScalar* __restrict__ bPrimePtr = bPrime.begin();

    const scalar* const __restrict__ diagPtr = matrix_.diag().begin();
    const scalar* const __restrict__ upperPtr =
        matrix_.upper().begin();
    const scalar* const __restrict__ lowerPtr =
        matrix_.lower().begin();

    const label* const __restrict__ uPtr =
        matrix_.lduAddr().upperAddr().begin();

    const label* const __restrict__ lPtr =
        matrix_.lduAddr().lowerAddr().begin();

    const label* const __restrict__ ownStartPtr =
        matrix_.lduAddr().ownerStartAddr().begin();


    // Parallel boundary initialisation.  The parallel boundary is treated
    // as an effective jacobi interface in the boundary.
    // Note: there is a change of sign in the coupled
    // interface update.  The reason for this is that the
    // internal coefficients are all located at the l.h.s. of
    // the matrix whereas the "implicit" coefficients on the
    // coupled boundaries are all created as if the
    // coefficient contribution is of a source-kind (i.e. they
    // have a sign as if they are on the r.h.s. of the matrix.
    // To compensate for this, it is necessary to turn the
    // sign of the contribution.

    for (label sweep=0; sweep<nSweeps; sweep++)
    {
        bPrime = source;

        const label startRequest = UPstream::nRequests();

        matrix_.initMatrixInterfaces
        (
            false,
            interfaceBouCoeffs_,
            interfaces_,
            psi,
            bPrime,
            cmpt
        );

        matrix_.updateMatrixInterfaces
        (
            false,
            interfaceBouCoeffs_,
            interfaces_,
            psi,
            bPrime,
            cmpt,
            startRequest
        );

        solveScalar psii;
        label fStart;
        label fEnd = ownStartPtr[0];
    
    #ifdef USE_OMP
        solveScalarField Z(psi.size());
        solveScalarField R(psi.size());

        solveScalar* __restrict__ rhsPtr = bPrime.begin();
        solveScalar* __restrict__ rPtr = R.begin();
        solveScalar* __restrict__ zPtr = Z.begin();

        const label nFaces = matrix_.upper().size();

        #pragma omp target teams distribute parallel for if (nCells > THRESHOLD_HIGH)
        for (label celli=0; celli<nCells; celli++)
        {
            rPtr[celli] = diagPtr[celli]*psiPtr[celli];
        }

        #pragma omp target teams distribute parallel for if (nFaces > THRESHOLD_LOW) thread_limit(256)
        for (label face=0; face<nFaces; face+=2)
        {
            const label nf = (nFaces - face) > 1 ? 2 : 1;
            #pragma unroll 2
            for (label i=0; i<nf; i++)
            {
                const label l_val = lPtr[face+i];
                const label u_val = uPtr[face+i];
                atomicAccumulator(rPtr[u_val]) += lowerPtr[face+i]*psiPtr[l_val];
                atomicAccumulator(rPtr[l_val]) += upperPtr[face+i]*psiPtr[u_val];
            }
        }

        #pragma omp target teams distribute parallel for if (nCells > THRESHOLD_HIGH)
        for (label celli=0; celli<nCells; celli++)
        {
            scalar r = rhsPtr[celli] - rPtr[celli];
            zPtr[celli] = r/diagPtr[celli];
            psiPtr[celli] += zPtr[celli];
        }

        scalar multiplier = -1.0;

        #pragma omp target teams distribute parallel for if(nCells > THRESHOLD_HIGH)
        for(label celli=0; celli<nCells; celli++)
        {
            fStart  = ownStartPtr[celli];
            fEnd    = ownStartPtr[celli+1];

            scalar tmp = 0.0;
            #pragma unroll 4
            for (label facei=fStart; facei<fEnd; facei++)
            {
                tmp += upperPtr[facei]*zPtr[uPtr[facei]];
            }
            
            rPtr[celli] = tmp;
        }

        #pragma omp target teams distribute parallel for if (nCells > THRESHOLD_HIGH)
        for (label celli=0; celli<nCells; celli++)
        {
            zPtr[celli] = rPtr[celli]/diagPtr[celli];
            psiPtr[celli] += multiplier * zPtr[celli];
        }
    #else
        for (label celli=0; celli<nCells; celli++)
        {
            // Start and end of this row
            fStart = fEnd;
            fEnd = ownStartPtr[celli + 1];

            // Get the accumulated neighbour side
            psii = bPrimePtr[celli];

            // Accumulate the owner product side
            for (label facei=fStart; facei<fEnd; facei++)
            {
                psii -= upperPtr[facei]*psiPtr[uPtr[facei]];
            }

            // Finish psi for this cell
            psii /= diagPtr[celli];

            // Distribute the neighbour side using psi for this cell
            for (label facei=fStart; facei<fEnd; facei++)
            {
                bPrimePtr[uPtr[facei]] -= lowerPtr[facei]*psii;
            }

            psiPtr[celli] = psii;
        }
    #endif    
    }
}


void Foam::GaussSeidelSmoother::smooth
(
    solveScalarField& psi,
    const scalarField& source,
    const direction cmpt,
    const label nSweeps
) const
{
    smooth
    (
        fieldName_,
        psi,
        matrix_,
        ConstPrecisionAdaptor<solveScalar, scalar>(source),
        interfaceBouCoeffs_,
        interfaces_,
        cmpt,
        nSweeps
    );
}


void Foam::GaussSeidelSmoother::scalarSmooth
(
    solveScalarField& psi,
    const solveScalarField& source,
    const direction cmpt,
    const label nSweeps
) const
{
    smooth
    (
        fieldName_,
        psi,
        matrix_,
        source,
        interfaceBouCoeffs_,
        interfaces_,
        cmpt,
        nSweeps
    );
}


// ************************************************************************* //
