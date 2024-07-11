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
    Atomic operations of vectors

SourceFiles
    AtomicAccumulatorVectorI.H
    
\*---------------------------------------------------------------------------*/

#ifndef AtomicAccumulatorVector_H
#define AtomicAccumulatorVector_H

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
            Class AtomicAccumulator<VectorSpace<...>> Declaration
\*---------------------------------------------------------------------------*/

template<class Form, class Cmpt, direction Ncmpts>
class AtomicAccumulator<VectorSpace<Form, Cmpt, Ncmpts>>
{
    // Private Data
        VectorSpace<Form, Cmpt, Ncmpts>* val;

public:
    
    // Constructors

        //- Construct from components
        AtomicAccumulator(VectorSpace<Form, Cmpt, Ncmpts> &ref);

    // Member Operators

        inline void operator+=(const VectorSpace<Form, Cmpt, Ncmpts>& rhs);

        inline void operator-=(const VectorSpace<Form, Cmpt, Ncmpts>& rhs);
};


// * * * * * * * * * * * * * * * * * Traits  * * * * * * * * * * * * * * * * //

template <class Form, class Cmpt, direction Ncmpts>
inline AtomicAccumulator<VectorSpace<Form, Cmpt, Ncmpts>> atomicAccumulator(VectorSpace<Form, Cmpt, Ncmpts>& ref)
{
    return AtomicAccumulator<VectorSpace<Form, Cmpt, Ncmpts>>(ref);
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#include "AtomicAccumulatorVectorI.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
