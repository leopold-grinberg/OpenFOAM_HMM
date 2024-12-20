/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2016 OpenFOAM Foundation
    Copyright (C) 2016 OpenCFD Ltd.
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

#include "lduAddressing.H"
#include "demandDrivenData.H"
#include "scalarField.H"

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

void Foam::lduAddressing::calcLosort() const
{
    if (losortPtr_)
    {
        FatalErrorInFunction
            << "losort already calculated"
            << abort(FatalError);
    }

    // Scan the neighbour list to find out how many times the cell
    // appears as a neighbour of the face. Done this way to avoid guessing
    // and resizing list
    labelList nNbrOfFace(size(), Zero);

    const labelUList& nbr = upperAddr();

    forAll(nbr, nbrI)
    {
        nNbrOfFace[nbr[nbrI]]++;
    }

    // Create temporary neighbour addressing
    labelListList cellNbrFaces(size());

    forAll(cellNbrFaces, celli)
    {
        cellNbrFaces[celli].setSize(nNbrOfFace[celli]);
    }

    // Reset the list of number of neighbours to zero
    nNbrOfFace = 0;

    // Scatter the neighbour faces
    forAll(nbr, nbrI)
    {
        cellNbrFaces[nbr[nbrI]][nNbrOfFace[nbr[nbrI]]] = nbrI;

        nNbrOfFace[nbr[nbrI]]++;
    }

    // Gather the neighbours into the losort array
    losortPtr_ = new labelList(nbr.size(), -1);

    labelList& lst = *losortPtr_;

    // Set counter for losort
    label lstI = 0;

    forAll(cellNbrFaces, celli)
    {
        const labelList& curNbr = cellNbrFaces[celli];

        forAll(curNbr, curNbrI)
        {
            lst[lstI] = curNbr[curNbrI];
            lstI++;
        }
    }
}


void Foam::lduAddressing::calcOwnerStart() const
{
    if (ownerStartPtr_)
    {
        FatalErrorInFunction
            << "owner start already calculated"
            << abort(FatalError);
    }

    const labelList& own = lowerAddr();

    ownerStartPtr_ = new labelList(size() + 1, own.size());

    labelList& ownStart = *ownerStartPtr_;

    // Set up first lookup by hand
    ownStart[0] = 0;
    label nOwnStart = 0;
    label i = 1;

    forAll(own, facei)
    {
        label curOwn = own[facei];

        if (curOwn > nOwnStart)
        {
            while (i <= curOwn)
            {
                ownStart[i++] = facei;
            }

            nOwnStart = curOwn;
        }
    }
}


void Foam::lduAddressing::calcLosortStart() const
{
    if (losortStartPtr_)
    {
        FatalErrorInFunction
            << "losort start already calculated"
            << abort(FatalError);
    }

    losortStartPtr_ = new labelList(size() + 1, Zero);

    labelList& lsrtStart = *losortStartPtr_;

    const labelList& nbr = upperAddr();

    const labelList& lsrt = losortAddr();

    // Set up first lookup by hand
    lsrtStart[0] = 0;
    label nLsrtStart = 0;
    label i = 0;

    forAll(lsrt, facei)
    {
        // Get neighbour
        const label curNbr = nbr[lsrt[facei]];

        if (curNbr > nLsrtStart)
        {
            while (i <= curNbr)
            {
                lsrtStart[i++] = facei;
            }

            nLsrtStart = curNbr;
        }
    }

    // Set up last lookup by hand
    lsrtStart[size()] = nbr.size();
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::lduAddressing::~lduAddressing()
{
    deleteDemandDrivenData(losortPtr_);
    deleteDemandDrivenData(ownerStartPtr_);
    deleteDemandDrivenData(losortStartPtr_);
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

const Foam::labelUList& Foam::lduAddressing::losortAddr() const
{
    if (!losortPtr_)
    {
        calcLosort();
    }

    return *losortPtr_;
}


const Foam::labelUList& Foam::lduAddressing::ownerStartAddr() const
{
    if (!ownerStartPtr_)
    {
        calcOwnerStart();
    }

    return *ownerStartPtr_;
}


const Foam::labelUList& Foam::lduAddressing::losortStartAddr() const
{
    if (!losortStartPtr_)
    {
        calcLosortStart();
    }

    return *losortStartPtr_;
}

#ifdef WITH_CSR
const Foam::labelUList& Foam::lduAddressing::L_J_CSR() const
{
    if (!L_J_CSR_Ptr_)
    {
	fprintf(stderr, " in duAddressing::L_J_CSR calling calc_L_CSR\n");
        calc_L_CSR(); //L_offsets_CSR_Ptr_ and L_J_CSR_Ptr_ are calculated in the same function
    }

    return *L_J_CSR_Ptr_;
}

const Foam::labelUList& Foam::lduAddressing::L_offsets_CSR() const
{
    if (!L_offsets_CSR_Ptr_)
    {
	fprintf(stderr, " in duAddressing::L_offsets_CSR calling calc_L_CSR\n");     
        calc_L_CSR(); //L_offsets_CSR_Ptr_ and L_J_CSR_Ptr_ are calculated in the same function
    }

    return *L_offsets_CSR_Ptr_;
}

const Foam::labelUList& Foam::lduAddressing::L_inrow_offsets_CSR() const
{
    if (!L_offsets_CSR_Ptr_)
    {
 	    fprintf(stderr, " in duAddressing::L_inrow_offsets_CSR calling calc_L_CSR\n");     
        calc_L_CSR(); //L_offsets_CSR_Ptr_ and L_J_CSR_Ptr_ are calculated in the same function
    }

    return *L_inrow_offsets_CSR_Ptr_;
}

#endif



void Foam::lduAddressing::clearOut()
{
    deleteDemandDrivenData(losortPtr_);
    deleteDemandDrivenData(ownerStartPtr_);
    deleteDemandDrivenData(losortStartPtr_);
    #ifdef WITH_CSR
    deleteDemandDrivenData(L_offsets_CSR_Ptr_);
    deleteDemandDrivenData(L_J_CSR_Ptr_);
    deleteDemandDrivenData(L_inrow_offsets_CSR_Ptr_);
    #endif
}


Foam::label Foam::lduAddressing::triIndex(const label a, const label b) const
{
    label own = min(a, b);

    label nbr = max(a, b);

    label startLabel = ownerStartAddr()[own];

    label endLabel = ownerStartAddr()[own + 1];

    const labelUList& neighbour = upperAddr();

    for (label i=startLabel; i<endLabel; i++)
    {
        if (neighbour[i] == nbr)
        {
            return i;
        }
    }

    // If neighbour has not been found, something has gone seriously
    // wrong with the addressing mechanism
    FatalErrorInFunction
        << "neighbour " << nbr << " not found for owner " << own << ". "
        << "Problem with addressing"
        << abort(FatalError);

    return -1;
}


Foam::Tuple2<Foam::label, Foam::scalar> Foam::lduAddressing::band() const
{
    const labelUList& owner = lowerAddr();
    const labelUList& neighbour = upperAddr();

    labelList cellBandwidth(size(), Zero);

    forAll(neighbour, facei)
    {
        label own = owner[facei];
        label nei = neighbour[facei];

        // Note: mag not necessary for correct (upper-triangular) ordering.
        label diff = nei-own;
        cellBandwidth[nei] = max(cellBandwidth[nei], diff);
    }

    label bandwidth = max(cellBandwidth);

    // Do not use field algebra because of conversion label to scalar
    scalar profile = 0.0;
    forAll(cellBandwidth, celli)
    {
        profile += 1.0*cellBandwidth[celli];
    }

    return Tuple2<label, scalar>(bandwidth, profile);
}


#ifdef WITH_CSR
void Foam::lduAddressing::calc_L_CSR() const
{

    fprintf(stderr,"in calc_L_CSR line=%d\n",__LINE__);

    const label* __restrict__ uPtr = upperAddr().begin();	
    const label* __restrict__ lPtr = lowerAddr().begin();
    const label nFaces = lowerAddr().size(); //check if this is correct

    fprintf(stderr,"in calc_L_CSR line=%d\n",__LINE__);

    //temporary array	
    labelList nnz_per_row_L(nFaces, 0);

    //label *nnz_per_row_L_Ptr = nnz_per_row_L.begin();


    for (label face=0; face < nFaces; face++) nnz_per_row_L[face] = 0; //likely not needed due to List creation

    for (label face=0; face < nFaces; face++) nnz_per_row_L [ uPtr[face] ]++;

    //find MAX nnz_per_row
    label max_nnz_per_row=0;
    for (label cell=0; cell < size(); ++cell){
       max_nnz_per_row = nnz_per_row_L[cell] >  max_nnz_per_row  ?  nnz_per_row_L[cell] : max_nnz_per_row ;  
    }

    fprintf(stderr," max_nnz_per_row = %d\n",max_nnz_per_row);

    #if 0
    //number of sparse rows:
    label ncells_L = 0; //should we keep ncells_L as a parameter or retrive it from L_offsets_CSR_Ptr_.size()-1   ?

    //find last row with non zero values
    for (label face=nFaces-1; face >=0; face--){
         if (nnz_per_row_L[face] > 0) {
            ncells_L = face+1;
            break;
         }
    }
    if (ncells_L != size() ){
       fprintf(stderr,"in calc_L_CSR line=%d ncells_L = %d size() = %d  \n",__LINE__, ncells_L, size());
    }
    #else
    label ncells_L = size();

    #endif

    L_offsets_CSR_Ptr_ = new labelList(ncells_L+1, Zero); //no need to initialize , but can we skip initialization ?
    labelList& offsets = *L_offsets_CSR_Ptr_;

    L_inrow_offsets_CSR_Ptr_ = new labelList(nFaces, Zero); //no need to initialize , but can we skip initialization ?
    labelList& inrow_offsets = *L_inrow_offsets_CSR_Ptr_;
    


    //calculate offsets to sparse rows
    offsets[0] = 0;
    for (label cell=1; cell <= ncells_L; ++cell){
        offsets[cell] =  offsets[cell-1] +  nnz_per_row_L[cell-1];
    }

    //reset nnz 
    for (label face=0; face < nFaces; face++) nnz_per_row_L[face] = 0;

    L_J_CSR_Ptr_ = new labelList(nFaces, Zero); //no need to initialize , but can we skip initialization ?
    labelList& ljcsr = *L_J_CSR_Ptr_;

    //save column indices  in CSR format
    for (label face=0; face < nFaces; face++){

       label cell = uPtr[face];
       ljcsr[ offsets[ cell] + nnz_per_row_L[cell] ] = lPtr[face];
       inrow_offsets[face] = nnz_per_row_L[cell]; 
       nnz_per_row_L [ cell ]++;
   }
  
}
#endif





// ************************************************************************* //
