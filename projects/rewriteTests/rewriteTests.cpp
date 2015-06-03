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
MatrixXd U(numRows, numCols);
VECTOR3_FIELD_3D V(xRes, yRes, zRes);
FIELD_3D F(xRes, yRes, zRes);
fftw_plan plan;
string path("../../U.preadvect.matrix");
COMPRESSION_DATA compression_data0, compression_data1, compression_data2;
int nBits = 16;
double percent = 0.95;
int maxIterations = 16;

////////////////////////////////////////////////////////
// Function Declarations
////////////////////////////////////////////////////////

// set up global variables
void InitGlobals();

// set up the compression data
void InitCompressionData(double percent, int maxIterations, int nBits, 
    int numBlocks, int numCols);

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

// test the zigzag scanning and reassembly
void ZigzagTest();

// test the modified cum sum routine
void CumSumTest();

// test the full matrix compression pipeline
void MatrixCompressionTest();

////////////////////////////////////////////////////////
// Main
////////////////////////////////////////////////////////

int main(int argc, char* argv[]) 
{
  TIMER functionTimer(__FUNCTION__);
  InitGlobals();
 
  MatrixCompressionTest();

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

  InitCompressionData(percent, maxIterations, nBits, numBlocks, numCols);

  EIGEN::read(path.c_str(), U); 

}

////////////////////////////////////////////////////////
// initialize the compression data
////////////////////////////////////////////////////////
void InitCompressionData(double percent, int maxIterations, int nBits, 
    int numBlocks, int numCols)
{
  compression_data0.set_percent(percent);
  compression_data0.set_maxIterations(maxIterations);
  compression_data0.set_nBits(nBits);
  compression_data0.set_numBlocks(numBlocks);
  compression_data0.set_numCols(numCols);
  compression_data0.set_dims(dims);

  // build the damping and zigzag arrays
  compression_data0.set_dampingArray();
  compression_data0.set_zigzagArray();

  compression_data1.set_percent(percent);
  compression_data1.set_maxIterations(maxIterations);
  compression_data1.set_nBits(nBits);
  compression_data1.set_numBlocks(numBlocks);
  compression_data1.set_numCols(numCols);
  compression_data1.set_dims(dims);

  compression_data1.set_dampingArray();
  compression_data1.set_zigzagArray();

  compression_data2.set_percent(percent);
  compression_data2.set_maxIterations(maxIterations);
  compression_data2.set_nBits(nBits);
  compression_data2.set_numBlocks(numBlocks);
  compression_data2.set_numCols(numCols);
  compression_data2.set_dims(dims);

  compression_data2.set_dampingArray();
  compression_data2.set_zigzagArray();
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
  cout << "V: " << endl;
  cout << v << endl;
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
  int col = 0;
  vector<FIELD_3D> blocks;
  GetBlocks(F, &blocks);
  UnitaryBlockDCT(1, &blocks);
  
  int blockNumber = 2;
  FIELD_3D block = blocks[blockNumber];
  PreprocessBlock(&block, blockNumber, col, &compression_data0);
  
  const FIELD_3D& dampingArray = compression_data0.get_dampingArray();
  FIELD_3D damp = dampingArray;

  TuneGamma(block, blockNumber, col, &compression_data0, &damp);
}

void EncodeBlockTest()
{ 
  int col = 0;
  vector<FIELD_3D> blocks;
  GetBlocks(F, &blocks);
  UnitaryBlockDCT(1, &blocks);
  
  for (int blockNumber = 0; blockNumber < numBlocks; blockNumber++) {

    PreprocessBlock(&(blocks[blockNumber]), blockNumber, col, &compression_data0);
  
    INTEGER_FIELD_3D quantized;
    EncodeBlock(blocks[blockNumber], blockNumber, col, &compression_data0, &quantized);

  }
  
  MatrixXd* sList = compression_data0.get_sListMatrix();
  MatrixXd* gammaList = compression_data0.get_gammaListMatrix();
  cout << "sList: " << (*sList) << endl;
  cout << "gammaList: " << (*gammaList) << endl; 
}

void EncodeDecodeBlockTest()
{
  int col = 0;
  vector<FIELD_3D> blocks;
  GetBlocks(F, &blocks);
  UnitaryBlockDCT(1, &blocks);

  int blockNumber = 24;
  FIELD_3D oldBlock = blocks[blockNumber];
  cout << "oldBlock: " << endl;
  cout << oldBlock.flattened() << endl;

  double oldEnergy = oldBlock.sumSq();
  PreprocessBlock(&(blocks[blockNumber]), blockNumber, col, &compression_data0);

  INTEGER_FIELD_3D quantized;
  EncodeBlock(blocks[blockNumber], blockNumber, col, &compression_data0, &quantized);
  cout << "encoded block: " << endl;
  cout << quantized.flattened() << endl;

  FIELD_3D decoded;
  DecodeBlockWithCompressionData(quantized, blockNumber, compression_data0, &decoded);

  cout << "newBlock: " << endl;
  cout << decoded.flattened() << endl;

  double newEnergy = decoded.sumSq();
  double diff = abs(oldEnergy - newEnergy) / oldEnergy; 
  cout << "Percent error from encoding and decoding: " << diff << endl;
  cout << "Accuracy was within: " << (1 - diff) << endl;
}

void ZigzagTest()
{
  int col = 0;
  vector<FIELD_3D> blocks;
  GetBlocks(F, &blocks);
  UnitaryBlockDCT(1, &blocks);

  int blockNumber = 24;
  FIELD_3D oldBlock = blocks[blockNumber];
  cout << "oldBlock: " << endl;
  cout << oldBlock.flattened() << endl;
  PreprocessBlock(&(blocks[blockNumber]), blockNumber, col, &compression_data0);

  INTEGER_FIELD_3D quantized;
  EncodeBlock(blocks[blockNumber], blockNumber, col, &compression_data0, &quantized);
  cout << "encoded block: " << endl;
  cout << quantized.flattened() << endl;

  VectorXi zigzagged;
  const INTEGER_FIELD_3D& zigzagArray = compression_data0.get_zigzagArray();
  ZigzagFlatten(quantized, zigzagArray, &zigzagged);
  cout << "zigzag scanned: " << endl;
  cout << EIGEN::convert(zigzagged) << endl;

  ZigzagUnflatten(zigzagged, zigzagArray, &quantized);
  cout << "unzigzagged: " << endl;
  cout << quantized.flattened() << endl;

}

void CumSumTest()
{
  int size = 10;
  VectorXi x(size);
  for (int i = 0; i < size; i++) {
    x[i] = i + 1;
  }

  VectorXi sum;
  ModifiedCumSum(x, &sum);

  cout << "x: " << endl;
  cout << x << endl;
  cout << "cum sum of x: " << endl;
  cout << sum << endl;
  
}

void MatrixCompressionTest()
{
  const char* filename = "U.preadvect.compressed.matrix";
  CompressAndWriteMatrixComponents(filename, U, &compression_data0, 
      &compression_data1, &compression_data2);
}
