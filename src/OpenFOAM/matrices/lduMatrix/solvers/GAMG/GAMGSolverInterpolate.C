/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2013-2015 OpenFOAM Foundation
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

#include "GAMGSolver.H"

#ifdef USE_OMP
#include <omp.h>
    #ifndef OMP_UNIFIED_MEMORY_REQUIRED
    #define OMP_UNIFIED_MEMORY_REQUIRED
    #pragma omp requires unified_shared_memory
    #endif

#include "AtomicAccumulator.H"
#endif

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

void Foam::GAMGSolver::interpolate
(
    solveScalarField& psi,
    solveScalarField& Apsi,
    const lduMatrix& m,
    const FieldField<Field, scalar>& interfaceBouCoeffs,
    const lduInterfaceFieldPtrsList& interfaces,
    const direction cmpt
) const
{
    solveScalar* __restrict__ psiPtr = psi.begin();

    const label* const __restrict__ uPtr = m.lduAddr().upperAddr().begin();
    const label* const __restrict__ lPtr = m.lduAddr().lowerAddr().begin();

    const scalar* const __restrict__ diagPtr = m.diag().begin();
    const scalar* const __restrict__ upperPtr = m.upper().begin();
    const scalar* const __restrict__ lowerPtr = m.lower().begin();

    Apsi = 0;
    solveScalar* __restrict__ ApsiPtr = Apsi.begin();

    const label startRequest = UPstream::nRequests();

    m.initMatrixInterfaces
    (
        true,
        interfaceBouCoeffs,
        interfaces,
        psi,
        Apsi,
        cmpt
    );

    const label nFaces = m.upper().size();
#ifdef USE_OMP
    #pragma omp target teams distribute parallel for if (target:nFaces>10000)
    for (label face=0; face<nFaces; face++)
    {
        atomicAccumulator(ApsiPtr[uPtr[face]]) += lowerPtr[face]*psiPtr[lPtr[face]];
        atomicAccumulator(ApsiPtr[lPtr[face]]) += upperPtr[face]*psiPtr[uPtr[face]];
    }
#else
    for (label face=0; face<nFaces; face++)
    {
        ApsiPtr[uPtr[face]] += lowerPtr[face]*psiPtr[lPtr[face]];
        ApsiPtr[lPtr[face]] += upperPtr[face]*psiPtr[uPtr[face]];
    }
#endif

    m.updateMatrixInterfaces
    (
        true,
        interfaceBouCoeffs,
        interfaces,
        psi,
        Apsi,
        cmpt,
        startRequest
    );

    const label nCells = m.diag().size();
#ifdef USE_OMP
    #pragma omp target teams distribute parallel for if (target:nCells>20000)
#endif
    for (label celli=0; celli<nCells; celli++)
    {
        psiPtr[celli] = -ApsiPtr[celli]/(diagPtr[celli]);
    }
}


void Foam::GAMGSolver::interpolate
(
    solveScalarField& psi,
    solveScalarField& Apsi,
    const lduMatrix& m,
    const FieldField<Field, scalar>& interfaceBouCoeffs,
    const lduInterfaceFieldPtrsList& interfaces,
    const labelList& restrictAddressing,
    const solveScalarField& psiC,
    const direction cmpt
) const
{
    interpolate
    (
        psi,
        Apsi,
        m,
        interfaceBouCoeffs,
        interfaces,
        cmpt
    );

    const label nCells = m.diag().size();
    solveScalar* __restrict__ psiPtr = psi.begin();
    const scalar* const __restrict__ diagPtr = m.diag().begin();
    const solveScalar* const __restrict__ psiCPtr = psiC.begin();


    const label nCCells = psiC.size();
    solveScalarField corrC(nCCells, 0);
    solveScalar* __restrict__ corrCPtr = corrC.begin();

    solveScalarField diagC(nCCells, 0);
    solveScalar* __restrict__ diagCPtr = diagC.begin();

#ifdef USE_OMP
    #pragma omp target teams distribute parallel for if (target:nCells>20000)
#endif
    for (label celli=0; celli<nCells; celli++)
    {
        corrCPtr[restrictAddressing[celli]] += diagPtr[celli]*psiPtr[celli];
        diagCPtr[restrictAddressing[celli]] += diagPtr[celli];
    }

#ifdef USE_OMP
    #pragma omp target teams distribute parallel for if (target:nCCells>20000)
#endif
    for (label ccelli=0; ccelli<nCCells; ccelli++)
    {
        corrCPtr[ccelli] = psiCPtr[ccelli] - corrCPtr[ccelli]/diagCPtr[ccelli];
    }

#ifdef USE_OMP
    #pragma omp target teams distribute parallel for if (target:nCells>20000)
#endif
    for (label celli=0; celli<nCells; celli++)
    {
        psiPtr[celli] += corrCPtr[restrictAddressing[celli]];
    }
}


// ************************************************************************* //
