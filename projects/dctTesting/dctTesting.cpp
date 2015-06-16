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
#include "COMPRESSION_REWRITE.h"


///////////////////////////////////////////////////////
// Globals
////////////////////////////////////////////////////////
int xRes = 8;
int yRes = 8;
int zRes = 8;
int numRows = 3 * xRes * yRes * zRes;
int numCols = 8;
VEC3I dims(xRes, yRes, zRes);
MATRIX U(numRows, numCols);
VECTOR3_FIELD_3D V(xRes, yRes, zRes);
MATRIX_COMPRESSION_DATA U_final_data;
MatrixXd svdV;

////////////////////////////////////////////////////////
// Function Declarations
////////////////////////////////////////////////////////
void TestProjectionAndUnprojection();

void PrintMatrixDims(const MatrixXd& M);

void InitGlobals();
void InitCompressionData();

void TestCompressedProjections();
VectorXd ProjectTest(const VECTOR3_FIELD_3D& V, const MATRIX& U);
VectorXd ProjectTransformTest(const VECTOR3_FIELD_3D& V, const MATRIX& U);
VectorXd ProjectTransformTestEigen(const VECTOR3_FIELD_3D& V, const MATRIX& U);
void UnprojectTest(const MATRIX& U, const VectorXd& q, VectorXd* V); 
void UnprojectTransformTest(const MATRIX& U, const VectorXd& q, VECTOR* V_x, VECTOR* V_y, VECTOR* V_z, VectorXd* result); 
void TestUnitaryDCTs();
void BasicDCTTest();

void FormVectorField(const MATRIX& U, const VEC3I& dims, int col, VECTOR3_FIELD_3D* field);
void BuildRandomU(MATRIX* U);
void BuildRandomV(VECTOR3_FIELD_3D* V);
vector<VectorXd> CastFieldToVecXd(const vector<FIELD_3D>& V);
VECTOR FlattenedVecOfFields(const vector<FIELD_3D>& V);

void BuildXYZMatrix(const VECTOR3_FIELD_3D& V, MatrixXd* A);
void BlockDiagonal(const MatrixXd& A, int count, MatrixXd* B);
void TransformVectorFieldSVD(const VECTOR3_FIELD_3D& V, VectorXd* s, MatrixXd* v, VECTOR3_FIELD_3D* transformedV);
void TestEigenRawBuffering();




////////////////////////////////////////////////////////
// Main
////////////////////////////////////////////////////////

int main(int argc, char* argv[]) 
{
  InitGlobals();
  // TestProjectionAndUnprojection();
  // TestCompressedProjections();
  // TestUnitaryDCTs();
  // BasicDCTTest();



  return 0;
}

////////////////////////////////////////////////////////
// *****************************************************
////////////////////////////////////////////////////////



///////////////////////////////////////////////////////
// Function implementations
///////////////////////////////////////////////////////

// initialize the global variables
void InitGlobals()
{
  xRes = 46;
  yRes = 62;
  zRes = 46;
  dims = VEC3I(xRes, yRes, zRes);
  numCols = 10;
  numRows = 3 * xRes * yRes * zRes;
  V = VECTOR3_FIELD_3D(xRes, yRes, zRes);
  U = MATRIX(numRows, numCols);

  // initialize an svdV matrix
  VectorXd s;
  VECTOR3_FIELD_3D transformedV;
  TransformVectorFieldSVD(V, &s, &svdV, &transformedV);
  cout << " svdV: " << endl;
  cout << svdV << endl;

}

// print out the dimensions of a passed in matrix
void PrintMatrixDims(const MatrixXd& M)
{
  cout << "Dims: (" << M.rows() << ", " << M.cols() << ")\n";
}


// naive projection
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

// projection, implemented in the spatial frequency domain
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
// projection, implemented in the spatial frequency domain
// using GetBlocksEigen, etc.
VectorXd ProjectTransformTestEigen(const VECTOR3_FIELD_3D& V, const MATRIX& U)
{
  const int totalColumns = U.cols(); 
  VectorXd result(totalColumns);

  FIELD_3D V_X, V_Y, V_Z;
  GetScalarFields(V, V_X, V_Y, V_Z);
  
  vector<VectorXd> Xpart = GetBlocksEigen(V_X);
  vector<VectorXd> Ypart = GetBlocksEigen(V_Y);
  vector<VectorXd> Zpart = GetBlocksEigen(V_Z);

  DoSmartUnitaryBlockDCTEigen(Xpart, 1);
  DoSmartUnitaryBlockDCTEigen(Ypart, 1);
  DoSmartUnitaryBlockDCTEigen(Zpart, 1);

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
    totalSum += GetDotProductSum(XpartUEigen, Xpart);
    totalSum += GetDotProductSum(YpartUEigen, Ypart); 
    totalSum += GetDotProductSum(ZpartUEigen, Zpart);

    result[col] = totalSum;
  }
  return result;
}

// unprojection implemented naively
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

// unprojection, implemented in the spatial frequency domain
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

// build a vector field of specified dimensions from  a particular column of
// a passed in matrix
void FormVectorField(const MATRIX& U, const VEC3I& dims, int col, VECTOR3_FIELD_3D* field)
{
  VECTOR column = U.getColumn(col);
  *field = VECTOR3_FIELD_3D(column, dims[0], dims[1], dims[2]); 
}

// build a random matrix with entries from 0 to 1
void BuildRandomU(MATRIX* U)
{
  for (int i = 0; i < U->rows(); i++) {
    for (int j = 0; j < U->cols(); j++) {
      (*U)(i, j) = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }
  }
}

// build a random vector field with entries from 0 to 1
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

// cast a c++ vector of field3ds into a c++ vector of
// (flattened-out) VectorXds
vector<VectorXd> CastFieldToVecXd(const vector<FIELD_3D>& V)
{
  vector<VectorXd> result(V.size());

  for (int i = 0; i < V.size(); i++) {
    VectorXd flat_i = V[i].flattenedEigen();
    result[i] = flat_i;
  }
  return result;
}

// flatten out a c++ vector of field3ds into a single VECTOR
// assumes each FIELD_3D is of dimensions 8 x 8 x 8!
VECTOR FlattenedVecOfFields(const vector<FIELD_3D>& V)
{
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

// given a passed in vec3 field, build a matrix with 3 columns,
// one for each of the x, y, and z components
void BuildXYZMatrix(const VECTOR3_FIELD_3D& V, MatrixXd* A) 
{
  int N = V.xRes() * V.yRes() * V.zRes();
  *A = MatrixXd::Zero(N, 3);

  FIELD_3D Vx, Vy, Vz;
  GetScalarFields(V, Vx, Vy, Vz);

  A->col(0) = Vx.flattenedEigen();
  A->col(1) = Vy.flattenedEigen();
  A->col(2) = Vz.flattenedEigen();
}

// build a block diagonal matrix with repeating copies of the passed
// in matrix A along the diagonal
void BlockDiagonal(const MatrixXd& A, int count, MatrixXd* B)
{
  *B = MatrixXd::Zero(A.rows() * count, A.cols() * count);
  for (int i = 0; i < count; i++) {
    B->block(i * A.rows(), i * A.cols(), A.rows(), A.cols()) = A;
  }
}

// find a new 3d coordinate system for a vector field using svd and transform into it
void TransformVectorFieldSVD(const VECTOR3_FIELD_3D& V, VectorXd* s, MatrixXd* v, VECTOR3_FIELD_3D* transformedV)
{
  MatrixXd xyzMatrix;
  BuildXYZMatrix(V, &xyzMatrix);
  JacobiSVD<MatrixXd> svd(xyzMatrix, ComputeThinU | ComputeThinV);
  *s = svd.singularValues();
  *v = svd.matrixV();

  int count = V.xRes() * V.yRes() * V.zRes();
  MatrixXd B;
  BlockDiagonal(svd.matrixV(), count,  &B); 

  VectorXd transformProduct = B * V.flattenedEigen();
  *transformedV = VECTOR3_FIELD_3D(transformProduct, V.xRes(), V.yRes(), V.zRes());
}

// testing the transform versions of projection/unprojection
void TestProjectionAndUnprojection()
{
  VECTOR::printVertical = false;
  srand(time(NULL));
  BuildRandomU(&U);
  BuildRandomV(&V);
  VectorXd result = ProjectTest(V, U);
  VECTOR resultVec = EIGEN::convert(result);
  cout << "Projection result: " << resultVec << '\n';

  VectorXd resultTransform = ProjectTransformTestEigen(V, U);
  VECTOR resultTransformVec = EIGEN::convert(resultTransform);
  cout << "Projection transform eigen result: " << resultTransformVec << '\n';

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
}
  
// testing how Eigen interfaces with raw buffers
void TestEigenRawBuffering()
{
  int size = 8;
  double* X = (double*)calloc(size, sizeof(double));
  Map<VectorXd> eigenX(X, size);
  cout << "eigenX is: " << endl;
  cout << eigenX << endl;
  free(X);
  X = NULL;

  VectorXd eigenRand = VectorXd::Random(size);
  cout << "eigenRand is: " << endl;
  cout << eigenRand << endl;

  X = eigenRand.data();
  cout << "X from eigenRand is: " << endl;
  for (int i = 0; i < size; i++) {
    cout << X[i] << endl;
  } 

  int rows = 3;
  int cols = 3;
  X = (double*)calloc(rows*cols, sizeof(double));
  Map<MatrixXd> eigenM(X, rows, cols);
  eigenM.setRandom();
  cout << "eigenM: " << endl;
  cout << eigenM << endl;
  cout << "X: " << endl;
  for (int i = 0; i < rows * cols; i++) {
    cout << X[i] << endl;
  }
  free(X);
  X = NULL;
}

/*
void TestCompressedProjections()
{
  InitCompressionData();
  BuildRandomV(&V);
  VectorXd q1 = PeeledCompressedProject(V, U_final_data);
  cout << "q1: " << endl; 
  cout << q1 << endl;
  
  VectorXd q2;
  PeeledCompressedProjectTransformTest1(V, U_final_data, &q2);
  cout << "q2: " << endl;
  cout << q2 << endl;

  double diff = (q1 - q2).norm();
  cout << "diff: " << diff << endl;
     
}
*/

/*
void InitCompressionData()
{
  int* UallDataX = NULL;
  int* UallDataY = NULL;
  int* UallDataZ = NULL;

  DECOMPRESSION_DATA Udecompression_dataX;
  DECOMPRESSION_DATA Udecompression_dataY;
  DECOMPRESSION_DATA Udecompression_dataZ;

  string reducedPath("../../data/reduced.stam.64/");
  string filename = reducedPath + string("U.final.componentX");
  ReadBinaryFileToMemory(filename.c_str(), UallDataX, Udecompression_dataX);
  filename = reducedPath + string("U.final.componentY");
  ReadBinaryFileToMemory(filename.c_str(), UallDataY, Udecompression_dataY);
  filename = reducedPath + string("U.final.componentZ");
  ReadBinaryFileToMemory(filename.c_str(), UallDataZ, Udecompression_dataZ);

  U_final_data = MATRIX_COMPRESSION_DATA(UallDataX, UallDataY, UallDataZ,
      Udecompression_dataX, Udecompression_dataY, Udecompression_dataZ);
}
*/

void TestUnitaryDCTs()
{
  BuildRandomV(&V);
  FIELD_3D V_X, V_Y, V_Z;
  GetScalarFields(V, V_X, V_Y, V_Z);
  vector<FIELD_3D> blocks = GetBlocks(V_X);
  // make a copy to compare with
  vector<FIELD_3D> truth = blocks;

  // take a forward and then a backward unitary transform.
  // the result should be within working precision
  DoSmartUnitaryBlockDCT(blocks, 1);
  DoSmartUnitaryBlockDCT(blocks, -1);

  VectorXd error(blocks.size());
  for (int i = 0; i < blocks.size(); i++) {
    double diff_i = (blocks[i].flattenedEigen() - truth[i].flattenedEigen()).norm();
    error[i] = diff_i;
  }

  cout << "Error at each block: " << endl;
  cout << error << endl;
}

void BasicDCTTest()
{
  VECTOR3_FIELD_3D A(2, 2, 2);
  BuildRandomV(&A);
  FIELD_3D X, Y, Z;
  GetScalarFields(A, X, Y, Z);
  FIELD_3D Xcopy = X;
  cout << "X original: " << endl;
  cout << Xcopy.flattenedEigen() << endl;
  double* in = (double*) fftw_malloc(sizeof(double) * 2 * 2 * 2);
  fftw_plan plan = Create_DCT_Plan(in, 1); 
  DCT_Smart_Unitary(X, plan, in, 1);
  cout << "Xhat: " << endl;
  cout << X.flattenedEigen() << endl;
  plan = Create_DCT_Plan(in, -1);
  DCT_Smart_Unitary(X, plan, in, -1);
  cout << "Xhathat: " << endl;
  cout << X.flattenedEigen() << endl;
  VectorXd ratio(2*2*2);

  int index = 0;
  for (int z = 0; z < 2; z++) {
    for (int y = 0; y < 2; y++) {
      for (int x = 0; x < 2; x++, index++) {

        ratio[index] = X(x, y, z) / Xcopy(x, y, z);
      }
    }
  }
  cout << "ratio: " << endl;
  cout << ratio << endl;  

}
  
  


   



  


