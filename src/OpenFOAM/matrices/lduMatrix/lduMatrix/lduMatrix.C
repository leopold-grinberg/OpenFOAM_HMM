/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2019-2021 OpenCFD Ltd.
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

#include "lduMatrix.H"
#include "IOstreams.H"
#include "Switch.H"
#include "objectRegistry.H"
#include "scalarIOField.H"
#include "Time.H"

#ifdef USE_ROCTX
#include <roctracer/roctx.h>
#endif

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(lduMatrix, 1);
}


const Foam::label Foam::lduMatrix::solver::defaultMaxIter_ = 1000;


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

Foam::lduMatrix::lduMatrix(const lduMesh& mesh)
:
    lduMesh_(mesh),
    lowerPtr_(nullptr),
    diagPtr_(nullptr),
    #ifndef WITH_CSR
    upperPtr_(nullptr)
    #else 
    upperPtr_(nullptr),
    lower_CSR_Ptr_(nullptr)
    #endif
{}


Foam::lduMatrix::lduMatrix(const lduMatrix& A)
:
    lduMesh_(A.lduMesh_),
    lowerPtr_(nullptr),
    diagPtr_(nullptr),
    #ifndef WITH_CSR 
    upperPtr_(nullptr)
    #else
    upperPtr_(nullptr),
    lower_CSR_Ptr_(nullptr)
    #endif

{
    if (A.lowerPtr_)
    {
        lowerPtr_ = new scalarField(*(A.lowerPtr_));
    }

    if (A.diagPtr_)
    {
        diagPtr_ = new scalarField(*(A.diagPtr_));
    }

    if (A.upperPtr_)
    {
        upperPtr_ = new scalarField(*(A.upperPtr_));
    }
    #ifdef WITH_CSR   
    if (A.lower_CSR_Ptr_)
    {
        lower_CSR_Ptr_ = new scalarField(*(A.lower_CSR_Ptr_));
    }
    #endif
}


Foam::lduMatrix::lduMatrix(lduMatrix& A, bool reuse)
:
    lduMesh_(A.lduMesh_),
    lowerPtr_(nullptr),
    diagPtr_(nullptr),
    #ifndef WITH_CSR
    upperPtr_(nullptr)
    #else
    upperPtr_(nullptr),
    lower_CSR_Ptr_(nullptr)
    #endif
{
    if (reuse)
    {
        if (A.lowerPtr_)
        {
            lowerPtr_ = A.lowerPtr_;
            A.lowerPtr_ = nullptr;
        }

        if (A.diagPtr_)
        {
            diagPtr_ = A.diagPtr_;
            A.diagPtr_ = nullptr;
        }

        if (A.upperPtr_)
        {
            upperPtr_ = A.upperPtr_;
            A.upperPtr_ = nullptr;
        }
        #ifdef WITH_CSR
        if (A.lower_CSR_Ptr_)
        {
            lower_CSR_Ptr_ = A.lower_CSR_Ptr_;
            A.lower_CSR_Ptr_ = nullptr;
        }
        #endif

    }
    else
    {
        if (A.lowerPtr_)
        {
            lowerPtr_ = new scalarField(*(A.lowerPtr_));
        }

        if (A.diagPtr_)
        {
            diagPtr_ = new scalarField(*(A.diagPtr_));
        }

        if (A.upperPtr_)
        {
            upperPtr_ = new scalarField(*(A.upperPtr_));
        }
        #ifdef WITH_CSR    
        if (A.lower_CSR_Ptr_)
        {
            lower_CSR_Ptr_ = new scalarField(*(A.lower_CSR_Ptr_));
        }
        #endif

    }
}


Foam::lduMatrix::lduMatrix(const lduMesh& mesh, Istream& is)
:
    lduMesh_(mesh),
    lowerPtr_(nullptr),
    diagPtr_(nullptr),
    #ifndef WITH_CSR
    upperPtr_(nullptr)
    #else
    upperPtr_(nullptr),
    lower_CSR_Ptr_(nullptr)
    #endif
{
    Switch hasLow(is);
    Switch hasDiag(is);
    Switch hasUp(is);
    #ifdef WITH_CSR
    Switch hasLowCSR(is);
    #endif

    if (hasLow)
    {
        lowerPtr_ = new scalarField(is);
    }
    if (hasDiag)
    {
        diagPtr_ = new scalarField(is);
    }
    if (hasUp)
    {
        upperPtr_ = new scalarField(is);
    }
    #ifdef WITH_CSR
    if (hasLowCSR)
    {
        lower_CSR_Ptr_ = new scalarField(is);
    }
    #endif
}


Foam::lduMatrix::~lduMatrix()
{
    if (lowerPtr_)
    {
        delete lowerPtr_;
    }

    if (diagPtr_)
    {
        delete diagPtr_;
    }

    if (upperPtr_)
    {
        delete upperPtr_;
    }

    #ifdef WITH_CSR
    if (lower_CSR_Ptr_)
    {
        delete lower_CSR_Ptr_;
    }
    #endif
}


Foam::scalarField& Foam::lduMatrix::lower()
{
    if (!lowerPtr_)
    {
        if (upperPtr_)
        {
            lowerPtr_ = new scalarField(*upperPtr_);
        }
        else
        {
            lowerPtr_ = new scalarField(lduAddr().lowerAddr().size(), Zero);
        }
    }

    return *lowerPtr_;
}

//#ifdef WITH_CSR
//Foam::scalarField& Foam::lduMatrix::lowerCSR()
//{
//    if (!lower_CSR_Ptr_)
//    {
//        {
//            //sieze of lower_CSR_Ptr is the same as lowerPtr_
//            fprintf(stderr,"in Foam::lduMatrix::lowerCSR line=%d\n",__LINE__);
//            lower_CSR_Ptr_ = new scalarField(lduAddr().lowerAddr().size(), Zero);
//            calc_lowerCSR();
//        }
//    }
//
//    return *lower_CSR_Ptr_;
//}
//#endif

Foam::scalarField& Foam::lduMatrix::diag()
{
    if (!diagPtr_)
    {
        diagPtr_ = new scalarField(lduAddr().size(), Zero);
    }

    return *diagPtr_;
}


Foam::scalarField& Foam::lduMatrix::upper()
{
    if (!upperPtr_)
    {
        if (lowerPtr_)
        {
            upperPtr_ = new scalarField(*lowerPtr_);
        }
        else
        {
            upperPtr_ = new scalarField(lduAddr().lowerAddr().size(), Zero);
        }
    }

    return *upperPtr_;
}


Foam::scalarField& Foam::lduMatrix::lower(const label nCoeffs)
{
    if (!lowerPtr_)
    {
        if (upperPtr_)
        {
            lowerPtr_ = new scalarField(*upperPtr_);
        }
        else
        {
            lowerPtr_ = new scalarField(nCoeffs, Zero);
        }
    }

    return *lowerPtr_;
}

//#ifdef WITH_CSR
//Foam::scalarField& Foam::lduMatrix::lowerCSR(const label nCoeffs)
//{
//    if (!lower_CSR_Ptr_)
//    {
//        {
//            lower_CSR_Ptr_ = new scalarField(nCoeffs, Zero);
//            fprintf(stderr,"lower_CSR_Ptr_  unallocated: creating line=%d\n",__LINE__);
//            calc_lowerCSR();
//        }
//    }
//
//    return *lower_CSR_Ptr_;
//}
//#endif


Foam::scalarField& Foam::lduMatrix::diag(const label size)
{
    if (!diagPtr_)
    {
        diagPtr_ = new scalarField(size, Zero);
    }

    return *diagPtr_;
}


Foam::scalarField& Foam::lduMatrix::upper(const label nCoeffs)
{
    if (!upperPtr_)
    {
        if (lowerPtr_)
        {
            upperPtr_ = new scalarField(*lowerPtr_);
        }
        else
        {
            upperPtr_ = new scalarField(nCoeffs, Zero);
        }
    }

    return *upperPtr_;
}


const Foam::scalarField& Foam::lduMatrix::lower() const
{
    if (!lowerPtr_ && !upperPtr_)
    {
        FatalErrorInFunction
            << "lowerPtr_ or upperPtr_ unallocated"
            << abort(FatalError);
    }

    if (lowerPtr_)
    {
        return *lowerPtr_;
    }
    else
    {
        return *upperPtr_;
    }
}

#ifdef WITH_CSR
const Foam::scalarField& Foam::lduMatrix::lowerCSR() const
{
    if (!lower_CSR_Ptr_)
    {
       #ifdef USE_ROCTX
       roctxRangePush("lduMatrix::lowerCSR");
       #endif
       //fprintf(stderr,"in lowerCSR line=%d lduAddr().lowerAddr().size() = %d\n",__LINE__,lduAddr().lowerAddr().size());
//        fprintf(stderr,"lduAddr().lowerAddr().size() = %d\n",lduAddr().lowerAddr().size());
        lower_CSR_Ptr_ = new scalarField(lduAddr().lowerAddr().size(), Zero);
        calc_lowerCSR();

        if (!lower_CSR_Ptr_) 
	      fprintf(stderr,"!lower_CSR_Ptr_ == true\n");

        #ifdef USE_ROCTX
        roctxRangePop();
        #endif
        return *lower_CSR_Ptr_;
        //FatalErrorInFunction
        //    << "lower_CSR_Ptr_  unallocated"
        //    << abort(FatalError);
    }
    else{
       return *lower_CSR_Ptr_;
    }
    
}
#endif

const Foam::scalarField& Foam::lduMatrix::diag() const
{
    if (!diagPtr_)
    {
        FatalErrorInFunction
            << "diagPtr_ unallocated"
            << abort(FatalError);
    }

    return *diagPtr_;
}


const Foam::scalarField& Foam::lduMatrix::upper() const
{
    if (!lowerPtr_ && !upperPtr_)
    {
        FatalErrorInFunction
            << "lowerPtr_ or upperPtr_ unallocated"
            << abort(FatalError);
    }

    if (upperPtr_)
    {
        return *upperPtr_;
    }
    else
    {
        return *lowerPtr_;
    }
}


void Foam::lduMatrix::setResidualField
(
    const scalarField& residual,
    const word& fieldName,
    const bool initial
) const
{
    if (!mesh().hasDb())
    {
        return;
    }

    scalarIOField* residualPtr =
        mesh().thisDb().getObjectPtr<scalarIOField>
        (
            initial
          ? IOobject::scopedName("initialResidual", fieldName)
          : IOobject::scopedName("residual", fieldName)
        );

    if (residualPtr)
    {
        const IOdictionary* dataPtr =
            mesh().thisDb().findObject<IOdictionary>("data");

        if (dataPtr)
        {
            if (initial && dataPtr->found("firstIteration"))
            {
                *residualPtr = residual;
                DebugInfo
                    << "Setting residual field for first solver iteration "
                    << "for solver field: " << fieldName << endl;
            }
        }
        else
        {
            *residualPtr = residual;
            DebugInfo
                << "Setting residual field for solver field "
                << fieldName << endl;
        }
    }
}


// * * * * * * * * * * * * * * * Friend Operators  * * * * * * * * * * * * * //

Foam::Ostream& Foam::operator<<(Ostream& os, const lduMatrix& ldum)
{
    Switch hasLow = ldum.hasLower();
    Switch hasDiag = ldum.hasDiag();
    Switch hasUp = ldum.hasUpper();
    #ifdef WITH_CSR
    Switch hasLowCSR = ldum.hasLowerCSR();
    #endif


    os  << hasLow << token::SPACE << hasDiag << token::SPACE
        << hasUp << token::SPACE;

    if (hasLow)
    {
        os  << ldum.lower();
    }

    if (hasDiag)
    {
        os  << ldum.diag();
    }

    if (hasUp)
    {
        os  << ldum.upper();
    }
    #ifdef WITH_CSR
    if (hasLowCSR)
    {
        os  << ldum.lowerCSR();
    }
    #endif


    os.check(FUNCTION_NAME);

    return os;
}


Foam::Ostream& Foam::operator<<(Ostream& os, const InfoProxy<lduMatrix>& ip)
{
    const lduMatrix& ldum = ip.t_;

    Switch hasLow = ldum.hasLower();
    Switch hasDiag = ldum.hasDiag();
    Switch hasUp = ldum.hasUpper();

    #ifdef WITH_CSR
    Switch hasLowCSR = ldum.hasLowerCSR();; 
    #endif

    os  << "Lower:" << hasLow
        << " Diag:" << hasDiag
        << " Upper:" << hasUp << endl;

    if (hasLow)
    {
        os  << "lower:" << ldum.lower().size() << endl;
    }
    if (hasDiag)
    {
        os  << "diag :" << ldum.diag().size() << endl;
    }
    if (hasUp)
    {
        os  << "upper:" << ldum.upper().size() << endl;
    }

    #ifdef WITH_CSR
    if (hasLowCSR)
    {
        os  << "lowerCSR:" << ldum.lowerCSR().size() << endl;
    }
    #endif

    //if (hasLow)
    //{
    //    os  << "lower contents:" << endl;
    //    forAll(ldum.lower(), i)
    //    {
    //        os  << "i:" << i << "\t" << ldum.lower()[i] << endl;
    //    }
    //    os  << endl;
    //}
    //if (hasDiag)
    //{
    //    os  << "diag contents:" << endl;
    //    forAll(ldum.diag(), i)
    //    {
    //        os  << "i:" << i << "\t" << ldum.diag()[i] << endl;
    //    }
    //    os  << endl;
    //}
    //if (hasUp)
    //{
    //    os  << "upper contents:" << endl;
    //    forAll(ldum.upper(), i)
    //    {
    //        os  << "i:" << i << "\t" << ldum.upper()[i] << endl;
    //    }
    //    os  << endl;
    //}

    os.check(FUNCTION_NAME);

    return os;
}

#ifdef WITH_CSR
void Foam::lduMatrix::calc_lowerCSR() const
{

    const label* __restrict__ uPtr = lduAddr().upperAddr().begin();	
    const label* __restrict__ lPtr = lduAddr().lowerAddr().begin();
    const label nFaces = lduAddr().lowerAddr().size(); //check if this is correct
    const scalar* const __restrict__ lowerPtr = lower().begin();
    const label* const __restrict__ offsets = lduAddr().L_offsets_CSR().begin();
    const label* const __restrict__ inrow_offset = lduAddr().L_inrow_offsets_CSR().begin(); 

    const label nCellsL = lduAddr().L_offsets_CSR().size()-1; 

    auto& lower_csr_coeffs_Ptr = *lower_CSR_Ptr_;

    //save coeficients  in CSR format
    #pragma omp target teams distribute parallel for if(nFaces > 3000)
    for (label face=0; face < nFaces; face++){

        label cell = uPtr[face];
        lower_csr_coeffs_Ptr[ offsets[ cell] + inrow_offset[face] ] = lowerPtr[face];
   }
}
#endif

// ************************************************************************* //
