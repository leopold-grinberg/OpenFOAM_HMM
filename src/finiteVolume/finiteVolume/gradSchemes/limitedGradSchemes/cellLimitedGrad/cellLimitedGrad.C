/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2018 OpenFOAM Foundation
    Copyright (C) 2021 OpenCFD Ltd.
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

#include "cellLimitedGrad.H"
#include "gaussGrad.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

template<class Type, class Limiter>
void Foam::fv::cellLimitedGrad<Type, Limiter>::limitGradient
(
    const Field<scalar>& limiter,
    Field<vector>& gIf
) const
{
    gIf *= limiter;
}


template<class Type, class Limiter>
void Foam::fv::cellLimitedGrad<Type, Limiter>::limitGradient
(
    const Field<vector>& limiter,
    Field<tensor>& gIf
) const
{
#ifdef USE_OMP
    #pragma omp target teams distribute parallel for if(gIf.size() > THRESHOLD_HIGH)
#endif
    for (label celli=0; celli < gIf.size(); ++celli)
    {
        gIf[celli] = tensor
        (
            cmptMultiply(limiter[celli], gIf[celli].x()),
            cmptMultiply(limiter[celli], gIf[celli].y()),
            cmptMultiply(limiter[celli], gIf[celli].z())
        );
    }
}


template<class Type, class Limiter>
Foam::tmp
<
    Foam::GeometricField
    <
        typename Foam::outerProduct<Foam::vector, Type>::type,
        Foam::fvPatchField,
        Foam::volMesh
    >
>
Foam::fv::cellLimitedGrad<Type, Limiter>::calcGrad
(
    const GeometricField<Type, fvPatchField, volMesh>& vsf,
    const word& name
) const
{
    const fvMesh& mesh = vsf.mesh();

    tmp
    <
        GeometricField
        <typename outerProduct<vector, Type>::type, fvPatchField, volMesh>
    > tGrad = basicGradScheme_().calcGrad(vsf, name);

    if (k_ < SMALL)
    {
        return tGrad;
    }

    GeometricField
    <
        typename outerProduct<vector, Type>::type,
        fvPatchField,
        volMesh
    >& g = tGrad.ref();

    const labelUList& owner = mesh.owner();
    const labelUList& neighbour = mesh.neighbour();

    const volVectorField& C = mesh.C();
    const surfaceVectorField& Cf = mesh.Cf();

    Field<Type> maxVsf(vsf.primitiveField());
    Field<Type> minVsf(vsf.primitiveField());

#ifdef USE_OMP
    static label *offsets = NULL;
    static label *neighbour_list = NULL;
    
    if (neighbour_list == NULL){
        offsets = new label[maxVsf.size()+1];
        label *count = new label[maxVsf.size()];

        for (label i = 0; i < maxVsf.size(); ++i ) { count[i] = 0; }

        //count neighbours for each owner
        for (label facei=0; facei < owner.size(); ++facei)
        {
            const label own = owner[facei];
            count[own]++;
        }

        offsets[0] = 0;
        for (label i = 0; i < maxVsf.size(); ++i )
        {
            offsets[i+1] = offsets[i]+count[i];
        }
        neighbour_list = new label[offsets[maxVsf.size()]];

        //list faces for each cell
        for (label i = 0; i < maxVsf.size(); ++i ) { count[i] = 0; }

        label *ptr_to_neighbour_list;

        for (label facei=0; facei < owner.size(); ++facei)
        {
            const label own = owner[facei];
            const label nei = neighbour[facei];
            ptr_to_neighbour_list = &neighbour_list[offsets[own] + count[own]];
            ptr_to_neighbour_list[0] = nei;
            count[own]++;
        }
        delete[] count;
    }

    label loop_len = maxVsf.size();
    #pragma omp target teams distribute parallel for if(loop_len > THRESHOLD_LOW) thread_limit(256)
    for (label celli = 0; celli < loop_len; celli+=1)
    {
        const label *ptr_to_neighbour_list = &neighbour_list[offsets[celli]];
        const label nFaces = offsets[celli+1] - offsets[celli];

        /*Foam::Vector<scalar>*/ Type maxVsf_celli = maxVsf[celli];
        /*Foam::Vector<scalar>*/ Type minVsf_celli = minVsf[celli];
        #pragma unroll 2
        for ( label f = 0; f < nFaces; ++f){
            label nei = ptr_to_neighbour_list[f];
            maxVsf_celli = Foam::max(maxVsf_celli, vsf[nei]);
            minVsf_celli = Foam::min(minVsf_celli, vsf[nei]);
        }
        maxVsf[celli] = maxVsf_celli;
        minVsf[celli] = minVsf_celli;
    }

    #pragma omp target teams distribute parallel for if(owner.size() > THRESHOLD_LOW) thread_limit(256)
    for (label facei = 0; facei < owner.size(); facei += 2)
    {
        const label nf = (owner.size() - facei) > 1 ? 2 : 1;
        #pragma unroll 2
        for ( label i = 0; i < nf; ++i)
        {
            const label own = owner[facei+i];
            const label nei = neighbour[facei+i];
        
            const Type& vsfOwn = vsf[own];
            const Type& vsfNei = vsf[nei];

            //maxVsf[nei] = Foam::max(maxVsf[nei], vsfOwn);
            for (direction cmpt = 0; 
                cmpt < pTraits<Foam::Vector<scalar>>::nComponents; 
                ++cmpt)
            {
                scalar& var = setComponent(maxVsf[nei],cmpt);
                #pragma omp atomic compare
                if (var < (scalar) component(vsfOwn,cmpt)) var = (scalar) component(vsfOwn,cmpt);
            }

            //minVsf[nei] = Foam::min(minVsf[nei], vsfOwn);
            for (direction cmpt = 0; 
                cmpt < pTraits<Foam::Vector<scalar>>::nComponents; 
                ++cmpt)
            {
                scalar& var = setComponent(minVsf[nei],cmpt);
                #pragma omp atomic compare
                if (var > (scalar) component(vsfOwn,cmpt)) var = (scalar) component(vsfOwn,cmpt);
            }
        }
    }

    const auto& bsf = vsf.boundaryField();

    forAll(bsf, patchi)
    {
        const fvPatchField<Type>& psf = bsf[patchi];
        const labelUList& pOwner = mesh.boundary()[patchi].faceCells();

        if (psf.coupled())
        {
            const Field<Type> psfNei(psf.patchNeighbourField());

            #pragma omp target teams distribute parallel for if (pOwner.size() > THRESHOLD_LOW)
            for (label pFacei = 0; pFacei < pOwner.size(); ++pFacei)
            {
                const label own = pOwner[pFacei];
                const Type& vsfNei = psfNei[pFacei];

                for (direction cmpt = 0; 
                    cmpt < pTraits<Foam::Vector<scalar>>::nComponents; 
                    ++cmpt)
                {
                    scalar& maxVar = setComponent(maxVsf[own],cmpt);
                    scalar& minVar = setComponent(minVsf[own],cmpt);
                    #pragma omp atomic compare
                    if (maxVar < (scalar) component(vsfNei,cmpt)) maxVar = (scalar) component(vsfNei,cmpt);

                    #pragma omp atomic compare
                    if (minVar > (scalar) component(vsfNei,cmpt)) minVar = (scalar) component(vsfNei,cmpt);
                }
            }
        }
        else
        {
            #pragma omp target teams distribute parallel for if (pOwner.size() > THRESHOLD_LOW)
            for (label pFacei = 0; pFacei < pOwner.size(); ++pFacei)
            {
                const label own = pOwner[pFacei];
                const Type& vsfNei = psf[pFacei];

                for (direction cmpt = 0; 
                    cmpt < pTraits<Foam::Vector<scalar>>::nComponents; 
                    ++cmpt)
                {
                    scalar& maxVar = setComponent(maxVsf[own],cmpt);
                    scalar& minVar = setComponent(minVsf[own],cmpt);
                    #pragma omp atomic compare
                    if (maxVar < (scalar) component(vsfNei,cmpt)) maxVar = (scalar) component(vsfNei,cmpt);

                    #pragma omp atomic compare
                    if (minVar > (scalar) component(vsfNei,cmpt)) minVar = (scalar) component(vsfNei,cmpt);
                }
            }
        }
    }
#else
    forAll(owner, facei)
    {
        const label own = owner[facei];
        const label nei = neighbour[facei];

        const Type& vsfOwn = vsf[own];
        const Type& vsfNei = vsf[nei];

        maxVsf[own] = max(maxVsf[own], vsfNei);
        minVsf[own] = min(minVsf[own], vsfNei);

        maxVsf[nei] = max(maxVsf[nei], vsfOwn);
        minVsf[nei] = min(minVsf[nei], vsfOwn);
    }


    const auto& bsf = vsf.boundaryField();

    forAll(bsf, patchi)
    {
        const fvPatchField<Type>& psf = bsf[patchi];
        const labelUList& pOwner = mesh.boundary()[patchi].faceCells();

        if (psf.coupled())
        {
            const Field<Type> psfNei(psf.patchNeighbourField());

            forAll(pOwner, pFacei)
            {
                const label own = pOwner[pFacei];
                const Type& vsfNei = psfNei[pFacei];

                maxVsf[own] = max(maxVsf[own], vsfNei);
                minVsf[own] = min(minVsf[own], vsfNei);
            }
        }
        else
        {
            forAll(pOwner, pFacei)
            {
                const label own = pOwner[pFacei];
                const Type& vsfNei = psf[pFacei];

                maxVsf[own] = max(maxVsf[own], vsfNei);
                minVsf[own] = min(minVsf[own], vsfNei);
            }
        }
    }
#endif

    maxVsf -= vsf;
    minVsf -= vsf;

    if (k_ < 1.0)
    {
        const Field<Type> maxMinVsf((1.0/k_ - 1.0)*(maxVsf - minVsf));
        maxVsf += maxMinVsf;
        minVsf -= maxMinVsf;
    }


    // Create limiter initialized to 1
    // Note: the limiter is not permitted to be > 1
    Field<Type> limiter(vsf.primitiveField().size(), pTraits<Type>::one);

#ifdef USE_OMP
    #pragma omp target teams distribute parallel for if (owner.size() > THRESHOLD_HIGH)
#endif
    for (label facei=0; facei<owner.size(); ++facei)
    {
        const label own = owner[facei];
        const label nei = neighbour[facei];

        // owner side
        limitFace
        (
            limiter[own],
            maxVsf[own],
            minVsf[own],
            (Cf[facei] - C[own]) & g[own]
        );

        // neighbour side
        limitFace
        (
            limiter[nei],
            maxVsf[nei],
            minVsf[nei],
            (Cf[facei] - C[nei]) & g[nei]
        );
    }

    forAll(bsf, patchi)
    {
        const labelUList& pOwner = mesh.boundary()[patchi].faceCells();
        const vectorField& pCf = Cf.boundaryField()[patchi];

    #ifdef USE_OMP
        #pragma omp target teams distribute parallel for if (pOwner.size() > THRESHOLD_HIGH)
    #endif
        for (label pFacei=0; pFacei<pOwner.size(); ++pFacei)
        {
            const label own = pOwner[pFacei];

            limitFace
            (
                limiter[own],
                maxVsf[own],
                minVsf[own],
                ((pCf[pFacei] - C[own]) & g[own])
            );
        }
    }

    if (fv::debug)
    {
        Info<< "gradient limiter for: " << vsf.name()
            << " max = " << gMax(limiter)
            << " min = " << gMin(limiter)
            << " average: " << gAverage(limiter) << endl;
    }

    limitGradient(limiter, g);
    g.correctBoundaryConditions();
    gaussGrad<Type>::correctBoundaryConditions(vsf, g);

    return tGrad;
}


// ************************************************************************* //
