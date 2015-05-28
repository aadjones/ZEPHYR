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
int xRes = 46;
int yRes = 62;
int zRes = 46;
int numRows = 3 * xRes * yRes * zRes;
int numCols = 151;
int numBlocks = xRes * yRes * zRes / (BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE);
VEC3I dims(xRes, yRes, zRes);
MATRIX U(numRows, numCols);
VECTOR3_FIELD_3D V(xRes, yRes, zRes);
FIELD_3D F(xRes, yRes, zRes);
fftw_plan plan;
string path("../../U.preadvect.matrix");
COMPRESSION_DATA compression_data;
int nBits = 16;
double percent = 0.85;
int maxIterations = 16;

////////////////////////////////////////////////////////
// Function Declarations
////////////////////////////////////////////////////////

// set up global variables
void InitGlobals();

// double-check the sparse construction 
// of a block-diagonal matrix
void SparseBlockDiagonalTest();

// check the difference between the original field and
// taking a dct followed by an idct
void DCTTest();

// check the difference between an original field
// and a block dct/idct field
void BlockDCTTest();

// check the difference between taking the SVD coordinate 
// transform and undoing it
void SVDTest();

// print out the 3 singular values of a particular velocity
// SVD transform
void SingularValuesTest(); 

// test the binary search algorithm for damping tuning
void GammaSearchTest();

// test the block encoding function
void EncodeBlockTest();

// test the encoding/decoding chain
void EncodeDecodeBlockTest();

////////////////////////////////////////////////////////
// Main
////////////////////////////////////////////////////////

int main(int argc, char* argv[]) 
{
  TIMER functionTimer(__FUNCTION__);
  InitGlobals();
  
  EncodeDecodeBlockTest();

  functionTimer.printTimings();
  return 0;
}

///////////////////////////////////////////////////////
// Function implementations
///////////////////////////////////////////////////////

////////////////////////////////////////////////////////
// initialize the global variables
////////////////////////////////////////////////////////
void InitGlobals()
{
  TIMER functionTimer(__FUNCTION__);
  srand(time(NULL));
  VECTOR::printVertical = false;

  U.read(path.c_str()); 
  cout << "U dims: " << "(" << U.rows() << ", " << U.cols() << ")\n";

  int col = 127;
  V = VECTOR3_FIELD_3D(U.getColumn(col), xRes, yRes, zRes);

  VectorXd s;
  MatrixXd v;
  TransformVectorFieldSVD(&s, &v, &V);
  F = V.scalarField(0);

  compression_data.set_percent(percent);
  compression_data.set_maxIterations(maxIterations);
  compression_data.set_nBits(nBits);
  compression_data.set_numBlocks(numBlocks);
  compression_data.set_dampingArray();
  
}

////////////////////////////////////////////////////////
// print the difference between an original 8 x 8 x 8
// field and taking its 3d dct and 3d idct in sequence
////////////////////////////////////////////////////////
void DCTTest()
{
  double* buffer = (double*) fftw_malloc(xRes * yRes * zRes * sizeof(double));
  int direction = 1;
  Create_DCT_Plan(buffer, direction, &plan);

  FIELD_3D F_old = F;
  DCT_Smart_Unitary(plan, direction, buffer, &F);

  direction = -1;
  Create_DCT_Plan(buffer, direction, &plan);
  DCT_Smart_Unitary(plan, direction, buffer, &F);

  double diff = ( F_old.flattenedEigen() - F.flattenedEigen() ).norm();
  cout << "Error from dct and idct: " << diff << endl;
  
  fftw_free(buffer);
  fftw_destroy_plan(plan);
  fftw_cleanup();

}

void BlockDCTTest()

{
  VEC3I paddedDims(0, 0, 0);
  GetPaddings(dims, &paddedDims);
  dims += paddedDims;
  FIELD_3D assimilatedF(dims[0], dims[1], dims[2]);
  FIELD_3D assimilatedTransformedF(dims[0], dims[1], dims[2]);

  vector<FIELD_3D> blocks;
  GetBlocks(F, &blocks);
  AssimilateBlocks(dims, blocks, &assimilatedF);

  UnitaryBlockDCT(1, &blocks);
  UnitaryBlockDCT(-1, &blocks);

  AssimilateBlocks(dims, blocks, &assimilatedTransformedF);

  double diff = (assimilatedF.flattenedEigen() - assimilatedTransformedF.flattenedEigen()).norm();
  cout << "Error from block dct and idct: " << diff << endl;
  
}

void SVDTest()
{
  VectorXd s;
  MatrixXd v;
  VECTOR3_FIELD_3D V_old = V;
  TransformVectorFieldSVD(&s, &v, &V);
  UntransformVectorFieldSVD(v, &V);
  double diff = (V.flattenedEigen() - V_old.flattenedEigen()).norm();
  cout << "Error from SVD and inverse SVD: " << diff << endl;
}

void SparseBlockDiagonalTest()
{
  MatrixXd A(3, 3);
  A.setRandom(3, 3);
  cout << "Random 3 x 3 matrix: " << endl;
  cout << A << endl;

  SparseMatrix<double> B;
  int count = 4;
  SparseBlockDiagonal(A, count, &B);
  
  cout << "Block diagonal B with count equal to " << count << ":" << endl;
  cout << B << endl;

}

void SingularValuesTest() 
{
  VectorXd s;
  MatrixXd v;
  VECTOR3_FIELD_3D V_old = V;
  TransformVectorFieldSVD(&s, &v, &V);
  cout << "Singular values: " << endl;
  cout << s << endl;
}

void GammaSearchTest()
{
  vector<FIELD_3D> blocks;
  GetBlocks(F, &blocks);
  UnitaryBlockDCT(1, &blocks);
  
  int blockNumber = 2;
  FIELD_3D block = blocks[blockNumber];
  PreprocessBlock(&block, blockNumber, &compression_data);
  
  const FIELD_3D& dampingArray = compression_data.get_dampingArray();
  FIELD_3D damp = dampingArray;

  TuneGamma(block, blockNumber, &compression_data, &damp);
}

void EncodeBlockTest()
{
  vector<FIELD_3D> blocks;
  GetBlocks(F, &blocks);
  UnitaryBlockDCT(1, &blocks);
  
  for (int blockNumber = 0; blockNumber < numBlocks; blockNumber++) {

    PreprocessBlock(&(blocks[blockNumber]), blockNumber, &compression_data);
  
    INTEGER_FIELD_3D quantized;
    EncodeBlock(blocks[blockNumber], blockNumber, &compression_data, &quantized);

  }

  VECTOR* sList = compression_data.get_sList();
  VECTOR* gammaList = compression_data.get_gammaList();
  cout << "sList: " << (*sList) << endl;
  cout << "gammaList: " << (*gammaList) << endl; 
}

void EncodeDecodeBlockTest()
{
  vector<FIELD_3D> blocks;
  GetBlocks(F, &blocks);
  UnitaryBlockDCT(1, &blocks);

  int blockNumber = 24;
  FIELD_3D oldBlock = blocks[blockNumber];
  cout << "oldBlock: " << endl;
  cout << oldBlock.flattened() << endl;

  double oldEnergy = oldBlock.sumSq();
  PreprocessBlock(&(blocks[blockNumber]), blockNumber, &compression_data);

  INTEGER_FIELD_3D quantized;
  EncodeBlock(blocks[blockNumber], blockNumber, &compression_data, &quantized);
  cout << "encoded block: " << endl;
  cout << quantized.flattened() << endl;

  FIELD_3D decoded;
  DecodeBlockWithCompressionData(quantized, blockNumber, compression_data, &decoded);

  cout << "newBlock: " << endl;
  cout << decoded.flattened() << endl;

  double newEnergy = decoded.sumSq();
  double diff = abs(oldEnergy - newEnergy) / oldEnergy; 
  cout << "Percent error from encoding and decoding: " << diff << endl;
  cout << "Accuracy was within: " << (1 - diff) << endl;
}
