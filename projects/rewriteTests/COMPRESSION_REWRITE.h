#ifndef COMPRESSION_REWRITE_H
#define COMPRESSION_REWRITE_H

#include <iostream>
#include <fftw3.h>

#include "EIGEN.h"
#include "VECTOR3_FIELD_3D.h"
#include "INTEGER_FIELD_3D.h"
#include "MATRIX_COMPRESSION_DATA.h"

//////////////////////////////////////////////////////// 
// Function signatures
////////////////////////////////////////////////////////

// round a FIELD_3D to an INTEGER_FIELD_3D
void RoundFieldToInt(const FIELD_3D& F, INTEGER_FIELD_3D* castedField); 

// cast an INTEGER_FIELD_3D to a FIELD_3D
void CastIntFieldToDouble(const INTEGER_FIELD_3D& F, FIELD_3D* castedField); 

// extract the three scalar fields from a vector field
void GetScalarFields(const VECTOR3_FIELD_3D& V, FIELD_3D* X, FIELD_3D* Y, FIELD_3D* Z); 

// make an fftw 3d dct plan. direction 1 is forward, -1 is inverse
void Create_DCT_Plan(double* in, int direction, fftw_plan* plan); 

// perform a unitary normalization for the forward 3d dct
void DCT_Unitary_Normalize(double* buffer);

// perform a unitary normalization in preparation for the inverse 3d dct
void UndoNormalize(FIELD_3D* F);

// do the corresponding dct based on the plan and direction
void DCT_Smart_Unitary(const fftw_plan& plan, int direction, double* in, FIELD_3D* F); 

#endif
