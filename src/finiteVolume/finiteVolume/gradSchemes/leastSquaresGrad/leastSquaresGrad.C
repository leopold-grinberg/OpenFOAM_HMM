/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2016 OpenFOAM Foundation
    Copyright (C) 2018-2021 OpenCFD Ltd.
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

#include "leastSquaresGrad.H"
#include "leastSquaresVectors.H"
#include "gaussGrad.H"
#include "fvMesh.H"
#include "volMesh.H"
#include "surfaceMesh.H"
#include "GeometricField.H"
#include "extrapolatedCalculatedFvPatchField.H"

#ifdef USE_ROCTX
#include <roctracer/roctx.h>
#endif


#ifdef USE_OMP
  #include <omp.h>
  #ifndef OMP_UNIFIED_MEMORY_REQUIRED
  #pragma omp requires unified_shared_memory
  #define OMP_UNIFIED_MEMORY_REQUIRED
  #endif
#endif


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

template<class Type>
Foam::tmp
<
    Foam::GeometricField
    <
        typename Foam::outerProduct<Foam::vector, Type>::type,
        Foam::fvPatchField,
        Foam::volMesh
    >
>
Foam::fv::leastSquaresGrad<Type>::calcGrad
(
    const GeometricField<Type, fvPatchField, volMesh>& vsf,
    const word& name
) const
{

    #ifdef USE_ROCTX
    roctxRangePush("leastSquaresGrad::calcGrad");
    #endif

    typedef typename outerProduct<vector, Type>::type GradType;
    typedef GeometricField<GradType, fvPatchField, volMesh> GradFieldType;

    const fvMesh& mesh = vsf.mesh();

    tmp<GradFieldType> tlsGrad
    (
        new GradFieldType
        (
            IOobject
            (
                name,
                vsf.instance(),
                mesh,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            mesh,
            dimensioned<GradType>(vsf.dimensions()/dimLength, Zero),
            extrapolatedCalculatedFvPatchField<GradType>::typeName
        )
    );
    GradFieldType& lsGrad = tlsGrad.ref();

    // Get reference to least square vectors
    const leastSquaresVectors& lsv = leastSquaresVectors::New(mesh);

    const surfaceVectorField& ownLs = lsv.pVectors();
    const surfaceVectorField& neiLs = lsv.nVectors();

    const labelUList& own = mesh.owner();
    const labelUList& nei = mesh.neighbour();


    #if 0 
    //forAll(own, facei)
    label loop_len = own.size();
    #pragma omp target teams distribute parallel for if(loop_len > 5000)
    for (label facei = 0; facei < loop_len; ++facei)
    {
        const label ownFacei = own[facei];
        const label neiFacei = nei[facei];

        const Type deltaVsf(vsf[neiFacei] - vsf[ownFacei]);
        atomicAccumulator(lsGrad[ownFacei]) += ownLs[facei]*deltaVsf;
        atomicAccumulator(lsGrad[neiFacei]) -= neiLs[facei]*deltaVsf;
    }
    #else
    static label *offsets = NULL;
    static label *face_list = NULL;

    if (face_list == NULL){
       fprintf(stderr, " GAUSS GRAD: setting up\n");

       offsets = new  (std::align_val_t(256))  label[lsGrad.size()+1];
       label *count = new (std::align_val_t(256))  label[lsGrad.size()];
       
       for (label i = 0; i < lsGrad.size(); ++i ) count[i] = 0;

       for (label facei=0; facei < own.size(); ++facei)
       {
           const label ownFacei = own[facei];
           const label neiFacei = nei[facei];
           count[ownFacei]++;
           count[neiFacei]++;
       }

       offsets[0] = 0;
       for (label i = 0; i < lsGrad.size(); ++i ){
         offsets[i+1] = offsets[i]+count[i];
       }
       face_list = new label[offsets[lsGrad.size()]];

       for (label i = 0; i < lsGrad.size(); ++i ) count[i] = 0;

        label *ptr_to_face_list, *ptr_to_face_sign;

       for (label facei=0; facei < own.size(); ++facei){

          const label ownFacei = own[facei];
          const label neiFacei = nei[facei];

          ptr_to_face_list = &face_list[ offsets[ownFacei] + count[ownFacei] ];
          ptr_to_face_list[0] = facei;
          count[ownFacei]++;

          ptr_to_face_list = &face_list[ offsets[neiFacei] + count[neiFacei] ];
          ptr_to_face_list[0] = facei;
          count[neiFacei]++;
       }
       delete[] count;
    }


    #if 1
    const auto& cells = mesh.cells();
    const label nCells = cells.size();
    #pragma omp target teams distribute parallel for thread_limit(256) if(nCells>5000 )
    for (label celli = 0; celli < nCells; ++celli)
        {
         
            const auto& cFaces = cells[celli];
            #pragma unroll 4
            for (label f = 0; f < cFaces.size(); ++f) 
            {
                const label facei = cFaces[f];
                const label ownFacei = own[facei];
                const label neiFacei = nei[facei];
                lsGrad[celli] += (ownLs[facei] - neiLs[facei])*(vsf[neiFacei] - vsf[ownFacei]);
            }
        }
    #else

    const label nCells = lsGrad.size();
    #pragma omp target teams distribute parallel for thread_limit(256) if(nCells>5000 )
    for (label celli = 0; celli < nCells; ++celli){

        const label *ptr_to_face_list = &face_list[offsets[celli]];
        const label nFaces = offsets[celli+1] - offsets[celli];

        #pragma unroll 4
        for ( label f = 0; f < nFaces; ++f){
          const label facei = ptr_to_face_list[f];
          const label ownFacei = own[facei];
          const label neiFacei = nei[facei];
          lsGrad[celli] += (ownLs[facei] - neiLs[facei])*(vsf[neiFacei] - vsf[ownFacei]);
        }
	}
    #endif

    #endif 
    // Boundary faces
    forAll(vsf.boundaryField(), patchi)
    {
        const fvsPatchVectorField& patchOwnLs = ownLs.boundaryField()[patchi];

        const labelUList& faceCells =
            vsf.boundaryField()[patchi].patch().faceCells();

        if (vsf.boundaryField()[patchi].coupled())
        {
            const Field<Type> neiVsf
            (
                vsf.boundaryField()[patchi].patchNeighbourField()
            );

            //forAll(neiVsf, patchFacei)
	    const label loop_len = neiVsf.size();
            #pragma omp target teams distribute parallel for if(loop_len > 5000)
            for (label patchFacei = 0; patchFacei < loop_len; ++patchFacei)	    
            {
                 atomicAccumulator(lsGrad[faceCells[patchFacei]]) +=
                    patchOwnLs[patchFacei]
                   *(neiVsf[patchFacei] - vsf[faceCells[patchFacei]]);
            }
        }
        else
        {
            const fvPatchField<Type>& patchVsf = vsf.boundaryField()[patchi];

            //forAll(patchVsf, patchFacei)
            const label loop_len = patchVsf.size();
            #pragma omp target teams distribute parallel for if(loop_len > 5000)
            for (label patchFacei = 0; patchFacei < loop_len; ++patchFacei)
            {
                 atomicAccumulator(lsGrad[faceCells[patchFacei]]) +=
                     patchOwnLs[patchFacei]
                    *(patchVsf[patchFacei] - vsf[faceCells[patchFacei]]);
            }
        }
    }


    lsGrad.correctBoundaryConditions();
    gaussGrad<Type>::correctBoundaryConditions(vsf, lsGrad);

    #ifdef USE_ROCTX
    roctxRangePop();
    #endif

    return tlsGrad;
}


// ************************************************************************* //
