/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
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

Class
    Foam::AtomicAccumulator

Description
    Atomic operations of scalars

SourceFiles
    AtomicAccumulatorScalarI.H
    
\*---------------------------------------------------------------------------*/

#ifndef AtomicAccumulatorScalar_H
#define AtomicAccumulatorScalar_H

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                         Class AtomicAccumulator Declaration
\*---------------------------------------------------------------------------*/

template<typename T>
class AtomicAccumulator
{
    // Private Data

        T* val;

public:

    // Constructors

        //- Construct from components
        AtomicAccumulator(T& ref);

    // Member Operators

        inline void operator+=(const T& rhs);

        inline void operator-=(const T& rhs);
};

// * * * * * * * * * * * * * * * * * Traits  * * * * * * * * * * * * * * * * //

template <typename T>
inline typename std::enable_if<std::is_scalar<T>::value, AtomicAccumulator<T>>::type atomicAccumulator(T& ref)
{
    return AtomicAccumulator<T>(ref);
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#include "AtomicAccumulatorScalarI.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
