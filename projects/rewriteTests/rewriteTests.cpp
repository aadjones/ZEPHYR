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
int numBlocks = (48 * 64 * 48) / (BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE);
VEC3I dims(xRes, yRes, zRes);
MatrixXd U(numRows, numCols);
VECTOR3_FIELD_3D V(xRes, yRes, zRes);
FIELD_3D F(xRes, yRes, zRes);
fftw_plan plan;
string path("../../U.preadvect.matrix");
COMPRESSION_DATA compression_data0, compression_data1, compression_data2;
int* allData0;
int* allData1;
int* allData2;
MATRIX_COMPRESSION_DATA matrix_data;
int nBits = 16;
double percent = 0.99;
int maxIterations = 16;

////////////////////////////////////////////////////////
// Function Declarations
////////////////////////////////////////////////////////

// set up global variables
void InitGlobals();

// set up the compression data
void InitCompressionData(double percent, int maxIterations, int nBits, 
    int numBlocks, int numCols);

// set up the matrix compression data from metadata and block data
void InitMatrixCompressionData();

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

// test encoding just one block
void EncodeOneBlockTest(int blockNumber, int col, COMPRESSION_DATA* data);

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

// test the reading of a binary file to memory
void ReadToMemoryTest();

// test the run-length codec
void RunLengthTest(int blockNumber, int col, COMPRESSION_DATA* data);

// test the block indices matrix construction
void BuildBlockIndicesTest();

// test the run-length decoder and dequantization methods
void DequantizationTest(int blockNumber, int col, COMPRESSION_DATA* data);

// test the decoding of an entire scalar field of a particular matrix column
void DecodeScalarFieldTest(int col, COMPRESSION_DATA* data);

// test the decoding of an entire vector field of a particular matrix column
void DecodeVectorFieldTest(int col);

// test the frequency domain projection
void PeeledCompressedProjectTransformTest();

////////////////////////////////////////////////////////
// Main
////////////////////////////////////////////////////////

int main(int argc, char* argv[]) 
{
  TIMER functionTimer(__FUNCTION__);
  InitGlobals();

  InitMatrixCompressionData();
 
  // int blockNumber = 287;
  int col = 0;

  // EncodeOneBlockTest(blockNumber, col, &compression_data0);

  // DecodeScalarFieldTest(col, &compression_data0);
  
  PeeledCompressedProjectTransformTest();
  

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

  // V = VECTOR3_FIELD_3D(U.col(150), xRes, yRes, zRes);
  // TransformVectorFieldSVDCompression(&V, &compression_data1);
  // F = V.scalarField(0);
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

////////////////////////////////////////////////////////
// print the error from doing a block dct and idct
////////////////////////////////////////////////////////
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

////////////////////////////////////////////////////////
// check the result of doing transform and untransform 
// svd on a vector field
////////////////////////////////////////////////////////
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

////////////////////////////////////////////////////////
// ensure that the sparse block diagonal construction
// builds properly
////////////////////////////////////////////////////////
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

////////////////////////////////////////////////////////
// display the three singular values of a particular
// SVD coordinate transform
////////////////////////////////////////////////////////
void SingularValuesTest() 
{
  VectorXd s;
  MatrixXd v;
  VECTOR3_FIELD_3D V_old = V;
  TransformVectorFieldSVD(&s, &v, &V);
  cout << "Singular values: " << endl;
  cout << s << endl;
}

////////////////////////////////////////////////////////
// test the binary search function TuneGamma
////////////////////////////////////////////////////////
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

////////////////////////////////////////////////////////
// encode all the blocks in one column and check whether
// the compression data is updated
////////////////////////////////////////////////////////
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
  cout << "sList, column 0: " << endl;
  cout << EIGEN::convert(sList->col(0)) << endl;
  cout << "gammaList, column 0: " << endl; 
  cout << EIGEN::convert(gammaList->col(0)) << endl;
}


////////////////////////////////////////////////////////
// encode just one block at blockNumber, col, data. useful
// for debugging codec
////////////////////////////////////////////////////////
void EncodeOneBlockTest(int blockNumber, int col, COMPRESSION_DATA* data)
{
  vector<FIELD_3D> blocks;
  GetBlocks(F, &blocks);
  UnitaryBlockDCT(1, &blocks);
  cout << "post dct: " << endl;
  cout << blocks[blockNumber].flattened() << endl;

  PreprocessBlock(&(blocks[blockNumber]), blockNumber, col, data);
 
  INTEGER_FIELD_3D quantized;
  EncodeBlock(blocks[blockNumber], blockNumber, col, data, &quantized);

  cout << "quantized block at blockNumber " << blockNumber << ", col " << col << ":\n";
  cout << quantized.flattened() << endl;

  const INTEGER_FIELD_3D& zigzagArray = data->get_zigzagArray();
  VectorXi zigzagged;
  ZigzagFlatten(quantized, zigzagArray, &zigzagged);
  cout << "quantized and zigzagged: " << endl;
  cout << EIGEN::convertInt(zigzagged) << endl;


  
}


////////////////////////////////////////////////////////
// check the error between encoding and decoding a block
////////////////////////////////////////////////////////
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
  DecodeBlockWithCompressionData(quantized, blockNumber, col, &compression_data0, &decoded);

  cout << "newBlock: " << endl;
  cout << decoded.flattened() << endl;

  double newEnergy = decoded.sumSq();
  double diff = abs(oldEnergy - newEnergy) / oldEnergy; 
  cout << "Percent error from encoding and decoding: " << diff << endl;
  cout << "Accuracy was within: " << (1 - diff) << endl;
}

////////////////////////////////////////////////////////
// ensure the zigzag scan works properly
////////////////////////////////////////////////////////
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
  cout << EIGEN::convertInt(zigzagged) << endl;

  ZigzagUnflatten(zigzagged, zigzagArray, &quantized);
  cout << "unzigzagged: " << endl;
  cout << quantized.flattened() << endl;

}

////////////////////////////////////////////////////////
// ensure the cumulative sum works properly
////////////////////////////////////////////////////////
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

////////////////////////////////////////////////////////
// test the compression and writing to a binary file
// of a full matrix
////////////////////////////////////////////////////////
void MatrixCompressionTest()
{
  const char* filename = "U.preadvect.compressed.matrix";
  CompressAndWriteMatrixComponents(filename, U, &compression_data0, 
      &compression_data1, &compression_data2);
}


////////////////////////////////////////////////////////
// test the reading into memory of a binary file that
// is the result of a previous matrix compression
////////////////////////////////////////////////////////
void ReadToMemoryTest()
{
  const char* filename = "U.preadvect.compressed.matrix0";
  int* allData = ReadBinaryFileToMemory(filename, &compression_data0);
  filename = "U.preadvect.SVD.data";
  ReadSVDData(filename, &compression_data0);
  vector<Matrix3d>* vList = compression_data0.get_vList();
  cout << "v at column 150: " << endl;
  cout << (*vList)[150] << endl;
  cout << "det v:" << endl;
  cout << (*vList)[150].determinant() << endl;
}

////////////////////////////////////////////////////////
// test the run-length decoding from a binary file
////////////////////////////////////////////////////////
void RunLengthTest(int blockNumber, int col, COMPRESSION_DATA* data) 
{
  const char* filename = "U.preadvect.compressed.matrix0";

  int* allData = ReadBinaryFileToMemory(filename, data);
  VectorXi parsedData;
  RunLengthDecodeBinary(allData, blockNumber, col, data, &parsedData);
  cout << "run-length decoded, not yet unzigzagged: " << endl;
  cout << EIGEN::convertInt(parsedData) << endl;


  INTEGER_FIELD_3D unflattened;
  const INTEGER_FIELD_3D& zigzagArray = data->get_zigzagArray();
 
  ZigzagUnflatten(parsedData, zigzagArray, &unflattened);

  cout << "parsedData for block " << blockNumber << ", col " << col << ':' << endl;
  cout << unflattened.flattened() << endl;
}

void BuildBlockIndicesTest() 
{

  MatrixXi testLengths;
  testLengths.setRandom(3, 3);
  cout << "testLengths: " << endl;
  cout << testLengths << endl;

  MatrixXi indices;
  BuildBlockIndicesMatrixDebug(testLengths, &indices);
  cout << "computed indices: " << endl;
  cout << indices << endl;

}


////////////////////////////////////////////////////////
// test the run-length decoding and dequantization
// from a binary file
////////////////////////////////////////////////////////
void DequantizationTest(int blockNumber, int col, COMPRESSION_DATA* data) 
{
  const char* filename = "U.preadvect.compressed.matrix0";

  int* allData = ReadBinaryFileToMemory(filename, data);
  VectorXi parsedData;
  RunLengthDecodeBinary(allData, blockNumber, col, data, &parsedData);
  cout << "run-length decoded, not yet unzigzagged: " << endl;
  cout << EIGEN::convertInt(parsedData) << endl;

  INTEGER_FIELD_3D unflattened;
  const INTEGER_FIELD_3D& zigzagArray = data->get_zigzagArray();
 
  ZigzagUnflatten(parsedData, zigzagArray, &unflattened);

  cout << "parsedData for block " << blockNumber << ", col " << col << ':' << endl;
  cout << unflattened.flattened() << endl;

  FIELD_3D decoded;
  DecodeBlockWithCompressionData(unflattened, blockNumber, col, data, &decoded);
  cout << "decoded: " << endl;
  cout << decoded.flattened() << endl;

}

////////////////////////////////////////////////////////
// test the decoding of an entire scalar field from a
// particular column of the original matrix
////////////////////////////////////////////////////////
void DecodeScalarFieldTest(int col, COMPRESSION_DATA* data)
{
 
  const char* filename = "U.preadvect.compressed.matrix0";
  
  int* allData = ReadBinaryFileToMemory(filename, data);
  
  FIELD_3D decoded;
  DecodeScalarField(data, allData, col, &decoded);

  FILE* pFile;
  pFile = fopen("U.preadvect.decoded.col150.x", "wb");

  if (pFile == NULL) {
    perror("Error opening file");
    exit(EXIT_FAILURE);
  }

  // write out the field for debugging
  int totalCells = decoded.totalCells();
  fwrite(&totalCells, sizeof(int), 1, pFile);
  fwrite(decoded.data(), sizeof(double), totalCells, pFile);

  
  // write out the 'ground truth' field
  pFile = fopen("U.preadvect.col150.x", "wb");

  if (pFile == NULL) {
    perror("Error opening file");
    exit(EXIT_FAILURE);
  }

  totalCells = F.totalCells();
  fwrite(&totalCells, sizeof(int), 1, pFile);
  fwrite(F.data(), sizeof(double), totalCells, pFile);

  fclose(pFile);

}


////////////////////////////////////////////////////////
// set up the matrix compression data
////////////////////////////////////////////////////////
void InitMatrixCompressionData() 
{
  // put the svd data inside the 0 component
  const char* filename = "U.preadvect.SVD.data";
  ReadSVDData(filename, &compression_data0);
  filename = "U.preadvect.compressed.matrix0";
  allData0 = ReadBinaryFileToMemory(filename, &compression_data0);
  filename = "U.preadvect.compressed.matrix1";
  allData1 = ReadBinaryFileToMemory(filename, &compression_data1);
  filename = "U.preadvect.compressed.matrix2";
  allData2 = ReadBinaryFileToMemory(filename, &compression_data2);

  matrix_data = MATRIX_COMPRESSION_DATA(allData0, allData1, allData2, 
      &compression_data0, &compression_data1, &compression_data2);
}


////////////////////////////////////////////////////////
// test the decoding of a full vector field
////////////////////////////////////////////////////////
void DecodeVectorFieldTest(int col)
{
 
  InitMatrixCompressionData();

  VECTOR3_FIELD_3D decoded;
  DecodeVectorField(&matrix_data, col, &decoded);
  VectorXd ground = U.col(col);
  double energyError = abs(decoded.flattenedEigen().squaredNorm() - ground.squaredNorm());
  energyError /= ground.squaredNorm();
  cout << "energy error: " << energyError << endl;
} 


void PeeledCompressedProjectTransformTest()
{
  InitMatrixCompressionData();

  VECTOR3_FIELD_3D randomV(xRes, yRes, zRes);
  randomV.setToRandom();
  VectorXd q;
  PeeledCompressedProjectTransform(V, &matrix_data, &q);

  VectorXd ground = randomV.peeledProject(U);

  double diff = (q - ground).norm();
  cout << "projection diff: " << diff << endl;

}
