/*
 This file is part of SSFR (Zephyr).
 
 Zephyr is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 
 Zephyr is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with Zephyr.  If not, see <http://www.gnu.org/licenses/>.
 
 Copyright 2013 Theodore Kim
 */

/*
 * Multi-dimensional DCT testing
 * Aaron Demby Jones
 * Fall 2014
 */

#include <iostream>
#include <time.h>
#include "COMPRESSION.h"


///////////////////////////////////////////////////////
// Globals
////////////////////////////////////////////////////////
const int xRes = 8;
const int yRes = 8;
const int zRes = 8;
const int numRows = 3 * xRes * yRes * zRes;
const int numCols = 8;
const VEC3I dims(xRes, yRes, zRes);
MATRIX U(numRows, numCols);
VECTOR3_FIELD_3D V(xRes, yRes, zRes);

////////////////////////////////////////////////////////
// Function Declarations
////////////////////////////////////////////////////////
VectorXd ProjectTest(const VECTOR3_FIELD_3D& V, const MATRIX& U);
VectorXd ProjectTransformTest(const VECTOR3_FIELD_3D& V, const MATRIX& U);
void UnprojectTest(const MATRIX& U, const VectorXd& q, VectorXd* V); 
void UnprojectTransformTest(const MATRIX& U, const VectorXd& q, VECTOR* V_x, VECTOR* V_y, VECTOR* V_z, VectorXd* result); 
void FormVectorField(const MATRIX& U, const VEC3I& dims, int col, VECTOR3_FIELD_3D* field);
void BuildRandomU(MATRIX* U);
void BuildRandomV(VECTOR3_FIELD_3D* V);
vector<VectorXd> CastFieldToVecXd(const vector<FIELD_3D>& V);
VECTOR FlattenedVecOfFields(const vector<FIELD_3D>& V);

////////////////////////////////////////////////////////
// Main
////////////////////////////////////////////////////////

int main(int argc, char* argv[]) 
{
  VECTOR::printVertical = false;
  srand(time(NULL));
  BuildRandomU(&U);
  BuildRandomV(&V);
  VectorXd result = ProjectTest(V, U);
  VECTOR resultVec = EIGEN::convert(result);
  cout << "Projection result: " << resultVec << '\n';

  VectorXd resultTransform = ProjectTransformTest(V, U);
  VECTOR resultTransformVec = EIGEN::convert(resultTransform);
  cout << "Projection transform result: " << resultTransformVec << '\n';

  double diffProject = (result - resultTransform).norm();
  cout << "Magnitude difference of projection methods: " << diffProject << '\n';

  VectorXd u;
  UnprojectTest(U, result, &u);
  VECTOR uVec = EIGEN::convert(u);
  // cout << "Unprojection result: " << uVec << '\n';

  VECTOR X, Y, Z;
  VectorXd vecFinal;
  UnprojectTransformTest(U, result, &X, &Y, &Z, &vecFinal);
  VECTOR vecFinalVec = EIGEN::convert(vecFinal);
  // cout << "Unprojection transform result: " << vecFinalVec << '\n';

  double diffUnproject = (u - vecFinal).norm();
  cout << "Magnitude difference of unprojection methods: " << diffUnproject << '\n';

  return 0;
}

///////////////////////////////////////////////////////
// Function implementations
///////////////////////////////////////////////////////

VectorXd ProjectTest(const VECTOR3_FIELD_3D& V, const MATRIX& U)
{
  const int totalColumns = U.cols(); 
  VectorXd result(totalColumns);

  FIELD_3D V_X, V_Y, V_Z;
  GetScalarFields(V, V_X, V_Y, V_Z);
  
  // GetBlocksEigen zero-pads as not to disturb the integrity of the matrix-vector multiply
  vector<VectorXd> Xpart = GetBlocksEigen(V_X);
  vector<VectorXd> Ypart = GetBlocksEigen(V_Y);
  vector<VectorXd> Zpart = GetBlocksEigen(V_Z);
  
  VECTOR3_FIELD_3D U_field;
  FIELD_3D U_X, U_Y, U_Z;
  int col = 0;
  for (int col = 0; col < totalColumns; col++) {
    FormVectorField(U, dims, col, &U_field);
    GetScalarFields(U_field, U_X, U_Y, U_Z);
    vector<VectorXd> XpartU = GetBlocksEigen(U_X);
    vector<VectorXd> YpartU = GetBlocksEigen(U_Y);
    vector<VectorXd> ZpartU = GetBlocksEigen(U_Z);

    double totalSum = 0.0;
    totalSum += GetDotProductSum(XpartU, Xpart);
   
    totalSum += GetDotProductSum(YpartU, Ypart); 
 
    totalSum += GetDotProductSum(ZpartU, Zpart);

    result[col] = totalSum;

  }

  return result;

}
VectorXd ProjectTransformTest(const VECTOR3_FIELD_3D& V, const MATRIX& U)
{
  const int totalColumns = U.cols(); 
  VectorXd result(totalColumns);

  FIELD_3D V_X, V_Y, V_Z;
  GetScalarFields(V, V_X, V_Y, V_Z);
  
  vector<FIELD_3D> Xpart = GetBlocks(V_X);
  vector<FIELD_3D> Ypart = GetBlocks(V_Y);
  vector<FIELD_3D> Zpart = GetBlocks(V_Z);

  DoSmartUnitaryBlockDCT(Xpart, 1);
  DoSmartUnitaryBlockDCT(Ypart, 1);
  DoSmartUnitaryBlockDCT(Zpart, 1);

  vector<VectorXd> XpartEigen = CastFieldToVecXd(Xpart);
  vector<VectorXd> YpartEigen = CastFieldToVecXd(Ypart);
  vector<VectorXd> ZpartEigen = CastFieldToVecXd(Zpart);

  
  VECTOR3_FIELD_3D U_field;
  FIELD_3D U_X, U_Y, U_Z;
  int col = 0;
  for (int col = 0; col < totalColumns; col++) {
    FormVectorField(U, dims, col, &U_field);
    GetScalarFields(U_field, U_X, U_Y, U_Z);

    vector<FIELD_3D> XpartU = GetBlocks(U_X);
    vector<FIELD_3D> YpartU = GetBlocks(U_Y);
    vector<FIELD_3D> ZpartU = GetBlocks(U_Z);

    DoSmartUnitaryBlockDCT(XpartU, 1);
    DoSmartUnitaryBlockDCT(YpartU, 1);
    DoSmartUnitaryBlockDCT(ZpartU, 1);

    vector<VectorXd> XpartUEigen = CastFieldToVecXd(XpartU);
    vector<VectorXd> YpartUEigen = CastFieldToVecXd(YpartU);
    vector<VectorXd> ZpartUEigen = CastFieldToVecXd(ZpartU);
 
    double totalSum = 0.0;
    totalSum += GetDotProductSum(XpartUEigen, XpartEigen);
   
    totalSum += GetDotProductSum(YpartUEigen, YpartEigen); 
 
    totalSum += GetDotProductSum(ZpartUEigen, ZpartEigen);

    result[col] = totalSum;

  }

  return result;

}


void UnprojectTest(const MATRIX& U, const VectorXd& q, VectorXd* V) 
{
  *V = VectorXd(U.rows());
  V->setZero();
  
  for (int col = 0; col < U.cols(); col++) {
    VECTOR U_col = U.getColumn(col);
    VectorXd U_col_eigen = EIGEN::convert(U_col);
    double coeff = q[col];
    (*V) += (coeff * U_col_eigen);
  }
}

void UnprojectTransformTest(const MATRIX& U, const VectorXd& q, VECTOR* Vx, VECTOR* Vy, VECTOR* Vz, VectorXd* result)
{
  *Vx = VECTOR(U.rows()/3);
  *Vy = VECTOR(U.rows()/3);
  *Vz = VECTOR(U.rows()/3);

  for (int col = 0; col < U.cols(); col++) {
    VECTOR U_col = U.getColumn(col);
    VECTOR3_FIELD_3D U_vecfield(U_col, xRes, yRes, zRes);
    FIELD_3D U_x, U_y, U_z;
    GetScalarFields(U_vecfield, U_x, U_y, U_z);

    vector<FIELD_3D> blocks_x = GetBlocks(U_x);
    vector<FIELD_3D> blocks_y = GetBlocks(U_y);
    vector<FIELD_3D> blocks_z = GetBlocks(U_z);
    
    DoSmartBlockDCT(blocks_x, 1);
    DoSmartBlockDCT(blocks_y, 1);
    DoSmartBlockDCT(blocks_z, 1);

    VECTOR blocks_x_vec = FlattenedVecOfFields(blocks_x);
    VECTOR blocks_y_vec = FlattenedVecOfFields(blocks_y);
    VECTOR blocks_z_vec = FlattenedVecOfFields(blocks_z);

    double coeff = q[col];
    (*Vx) += (coeff * blocks_x_vec);
    (*Vy) += (coeff * blocks_y_vec);
    (*Vz) += (coeff * blocks_z_vec);

    FIELD_3D reassembled_x(Vx->data(), xRes, yRes, zRes);
    FIELD_3D reassembled_y(Vy->data(), xRes, yRes, zRes);
    FIELD_3D reassembled_z(Vz->data(), xRes, yRes, zRes);

    vector<FIELD_3D> blocks_reassembled_x = GetBlocks(reassembled_x);
    vector<FIELD_3D> blocks_reassembled_y = GetBlocks(reassembled_y);
    vector<FIELD_3D> blocks_reassembled_z = GetBlocks(reassembled_z);

    DoSmartBlockDCT(blocks_reassembled_x, -1);
    DoSmartBlockDCT(blocks_reassembled_y, -1);
    DoSmartBlockDCT(blocks_reassembled_z, -1);
    
    FIELD_3D assimilated_x(xRes, yRes, zRes);
    FIELD_3D assimilated_y(xRes, yRes, zRes);
    FIELD_3D assimilated_z(xRes, yRes, zRes);
    AssimilateBlocks(dims, blocks_reassembled_x, assimilated_x);
    AssimilateBlocks(dims, blocks_reassembled_y, assimilated_y);
    AssimilateBlocks(dims, blocks_reassembled_z, assimilated_z);

    VECTOR3_FIELD_3D reassembled_V(assimilated_x.data(), assimilated_y.data(), assimilated_z.data(), xRes, yRes, zRes);
    *result = reassembled_V.flattenedEigen();

  }
} 

void FormVectorField(const MATRIX& U, const VEC3I& dims, int col, VECTOR3_FIELD_3D* field)
{
  VECTOR column = U.getColumn(col);
  *field = VECTOR3_FIELD_3D(column, dims[0], dims[1], dims[2]); 
}

void BuildRandomU(MATRIX* U)
{
  for (int i = 0; i < U->rows(); i++) {
    for (int j = 0; j < U->cols(); j++) {
      (*U)(i, j) = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }
  }
}

void BuildRandomV(VECTOR3_FIELD_3D* V)
{
  for (int z = 0; z < V->zRes(); z++) {
    for (int y = 0; y < V->yRes(); y++) {
      for (int x = 0; x < V->xRes(); x++) {
        for (int i = 0; i < 3; i++) {
          (*V)(x, y, z)[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        }
      }
    }
  }
}

vector<VectorXd> CastFieldToVecXd(const vector<FIELD_3D>& V)
{
  vector<VectorXd> result(V.size());

  for (int i = 0; i < V.size(); i++) {
    VectorXd flat_i = V[i].flattenedEigen();
    result[i] = flat_i;
  }
  return result;
}
 
VECTOR FlattenedVecOfFields(const vector<FIELD_3D>& V)
{
  // assume each FIELD_3D is of dimensions 8 x 8 x 8!
  int totalLength = 8 * 8 * 8 * V.size();
  VECTOR result(totalLength);
  int index = 0;

  for (int i = 0; i < V.size(); i++) {
    for (int j = 0; j < 8 * 8 * 8; j++, index++) {
      result[index] = V[i][j];
    }
  }
  return result;
}
