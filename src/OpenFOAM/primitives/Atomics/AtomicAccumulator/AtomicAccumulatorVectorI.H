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

\*---------------------------------------------------------------------------*/

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //
template <class Form, class Cmpt, direction Ncmpts>
AtomicAccumulator<VectorSpace<Form, Cmpt, Ncmpts>>::AtomicAccumulator (VectorSpace<Form, Cmpt, Ncmpts> &ref) 
: 
    val(&ref) 
{}

// * * * * * * * * * * * * * * * Member Operators  * * * * * * * * * * * * * //
template <class Form, class Cmpt, direction Ncmpts>
inline void AtomicAccumulator<VectorSpace<Form, Cmpt, Ncmpts>>::operator+=(const VectorSpace<Form, Cmpt, Ncmpts>& rhs)
{
    VectorSpaceOps<Ncmpts,0>::eqOp(*val, rhs, atomicPlusEqOp<Cmpt>());
}

template <class Form, class Cmpt, direction Ncmpts>
inline void AtomicAccumulator<VectorSpace<Form, Cmpt, Ncmpts>>::operator-=(const VectorSpace<Form, Cmpt, Ncmpts>& rhs)
{
    VectorSpaceOps<Ncmpts,0>::eqOp(*val, rhs, atomicMinusEqOp<Cmpt>());
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //



// ************************************************************************* //
