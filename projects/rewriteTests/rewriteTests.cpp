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
 * Summer 2015
 */

#include <iostream>
#include <time.h>
#include "COMPRESSION_REWRITE.h"
#include "BIG_MATRIX.h"

///////////////////////////////////////////////////////
// Globals
////////////////////////////////////////////////////////
// int xRes = 46;
// int yRes = 62;
// int zRes = 46;
int xRes = 198;
int yRes = 264;
int zRes = 198;
// int xPadded = 48;
// int yPadded = 64;
// int zPadded = 48;
int xPadded = 200;
int yPadded = 264;
int zPadded = 200;
int numRows = 3 * xRes * yRes * zRes;
int numCols = 151;
int numBlocks = (xPadded * yPadded * zPadded) / (BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE);
VEC3I dims(xRes, yRes, zRes);
VEC3I g_paddedDims(xPadded, yPadded, zPadded);
MatrixXd U(numRows, numCols);
BIG_MATRIX Q(numRows, numCols);
VECTOR3_FIELD_3D V(xRes, yRes, zRes);
FIELD_3D F(xRes, yRes, zRes);
FIELD_3D F0(xRes, yRes, zRes);
FIELD_3D F1(xRes, yRes, zRes);
FIELD_3D F2(xRes, yRes, zRes);
vector<FIELD_3D> g_blocks;
vector<VectorXd> g_blocksEigen;
fftw_plan plan;
// string path("projects/rewriteTests/U.final.matrix");
// string path("projects/rewriteTests/U.preadvect.matrix");
string bigMatrixPath("scratch/Q.final.bigmatrix");
COMPRESSION_DATA compression_data0, compression_data1, compression_data2;
int* allData0;
int* allData1;
int* allData2;
MATRIX_COMPRESSION_DATA matrix_data;
int nBits = 32;
double percent = 0.999;
int maxIterations = 32;

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

// compute the block dct of a passed in field
// and write out the contents 
void DoBlockDCTFromFile(string fieldFile, string outputFile);

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

// visualize the preprocessing of a block using fieldViewer
void BlockVisualizationTest(int blockNumber, int col, COMPRESSION_DATA* data);

// visualize a scalar component of the entire SVD column at the indicated col
void FullColumnVisualizationTest(int col, int component);

// visualize a scalar component of an entire column of the passed in written out matrix file
void FullColumnVisualizationFromFile(int col, const char* fileName, int component);

// test the block encoding function
void EncodeBlockTest();

// test the encoding/decoding chain
void EncodeDecodeBlockTest();

// test the encoding/decoding chain with
// quantized cached gammas
void EncodeDecodeBlockTestNoFastPow();  
 
// test the zigzag scanning and reassembly
void ZigzagTest();

// test the modified cum sum routine
void CumSumTest();

// test the full matrix compression pipeline
void MatrixCompressionTest();

// test the matrix compression pipeline using gamma zero everywhere
void MatrixCompressionDebugTest();

// test the reading of a binary file to memory
void ReadToMemoryTest();

// test the run-length codec
void RunLengthTest(int blockNumber, int col, COMPRESSION_DATA* data);

// test the run-length codec's sparsity
void RunLengthSparseTest(COMPRESSION_DATA* compression_data, const char* filename, string outStr);

// test the block indices matrix construction
void BuildBlockIndicesTest();

// test the run-length decoder and dequantization methods
void DequantizationTest(int blockNumber, int col, COMPRESSION_DATA* data);

// test the decoding of an entire scalar field of a particular matrix column
void DecodeScalarFieldTest(int col, COMPRESSION_DATA* data);

// test the decoding of an entire vector field of a particular matrix column
void DecodeVectorFieldTest(int col);

// test the spatial domain projection
void PeeledCompressedProjectTest();

// test the frequency domain projection
void PeeledCompressedProjectTransformTest();

// test the entire matrix decoder
void DecodeMatrixTest();

// test the 'eigen' version of decode scalar field
void DecodeScalarFieldEigenTest(int col, COMPRESSION_DATA* data);

// test the eigen version of block dct
void UnitaryBlockDCTEigenTest();

// test the frequency domain projection without using compression
void PeeledProjectTransformTest();

// test the peeled unprojection
void PeeledCompressedUnprojectTest();

// test the peeled unprojection in the frequency domain
void PeeledCompressedUnprojectTransformTest();

// test DecodeFromRowCol
void DecodeFromRowColTest(int row, int col);

// test getting a submatrix for the reduced compressed advection
void GetSubmatrixTest(int startRow);

// test the distribution of gamma values in a data set
void GammaAnalyticsTest(COMPRESSION_DATA* data);

// test the timings of the projection on blocks of different sparsity
void ProjectSparsityTimingTest(int col, int blockNumber, int numIterations);

// utility function to read a single column from a (large) binary matrix file
void ReadColumnFromMatrix(int col, const char* fileName, VECTOR* result);

// function to test ReadColumnFromMatrix
void ReadColumnFromMatrixTest(int col);

// utility function to read a single column from a big matrix file
void ReadColumnFromBigMatrix(int col, const char* fileName, VECTOR* result);

// just read the necessary data for one column from a matrix binary file
void FullColumnVisualizationFromFile(int col, const char* fileName, int component);

// same as above but from a big matrix file. works better since big matrix uses
// column-major storage
void FullColumnVisualizationFromBigMatrixFile(int col, const char* fileName, int component);

// the naive approach, reading in the entire big matrix. mainly just checking that
// BIGMATRIX::read is working
void FullColumnVisualizationFromBigMatrixFileNaive(int col, const char* fileName, int component);

////////////////////////////////////////////////////////
// Main
////////////////////////////////////////////////////////

int main(int argc, char* argv[]) 
{
  TIMER functionTimer(__FUNCTION__);
  InitGlobals();

  // int blockNumber = 36;
  // int blockNumber = 9436;
  // int blockNumber = 36 + (25 * 33);
  // int blockNumber = 4974;

  // EncodeOneBlockTest(blockNumber, col, &compression_data0);

  // DecodeScalarFieldTest(col, &compression_data0);
  
  // PeeledCompressedProjectTransformTest();

  // MatrixCompressionTest();

  // PeeledCompressedUnprojectTest();

  // PeeledCompressedUnprojectTransformTest();

  // DecodeVectorFieldTest(col);

  // DecodeFromRowColTest(row, col);

  // MatrixCompressionDebugTest();
  
  // DecodeMatrixTest();
 
  // PeeledCompressedProjectTest();

  // GammaSearchTest();
  
  // GammaAnalyticsTest(&compression_data0);
  
  // EncodeDecodeBlockTestNoFastPow();  
  
  // BlockVisualizationTest(blockNumber, col, &compression_data0); 
  // blockNumber = 10264;
  // BlockVisualizationTest(blockNumber, col, &compression_data0); 

  // int numIterations = 1000000;
  // ProjectSparsityTimingTest(col, blockNumber, numIterations);
 
  int col = 150;
  int component = 0; 
  // FullColumnVisualizationTest(col, component);
  // const char* UfinalPath = "projects/rewriteTests/U.final.matrix";
  const char* QPath = "scratch/Q.final.bigmatrix";
  FullColumnVisualizationFromBigMatrixFile(col, QPath, component); 
  // FullColumnVisualizationFromBigMatrixFileNaive(col, QPath, component); 
  // string fieldFile("Q.velocity.col0.0.field"); 
  // FIELD_3D viewedField;
  // viewedField.read(fieldFile);
  // FIELDVIEW3D(viewedField);

  // string fieldFile("Q.velocity.col0.0.field");
  // string outputFile("Q.velocity.col0.0.dct.field");
  // DoBlockDCTFromFile(fieldFile, outputFile);
  // FIELD_3D dct;
  // dct.read(outputFile);
  // FIELDVIEW3D(dct);

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

  // InitCompressionData(percent, maxIterations, nBits, numBlocks, numCols);

  // InitMatrixCompressionData();

  // EIGEN::read(path.c_str(), U); 
  // Q.read(bigMatrixPath);
  // Q is of type BIG_MATRIX, and the column accessor is []
  // V = VECTOR3_FIELD_3D(Q[0], xRes, yRes, zRes);
  // TransformVectorFieldSVDCompression(&V, &compression_data1);
  // F = V.scalarField(0);
  // F0 = V.scalarField(0);
  // F1 = V.scalarField(1);
  // F2 = V.scalarField(2);
  // GetBlocks(F, &g_blocks);
  // UnitaryBlockDCT(1, &g_blocks);
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
  compression_data0.set_paddedDims(g_paddedDims);

  // build the damping and zigzag arrays
  compression_data0.set_dampingArray();
  compression_data0.set_dampingArrayList();
  compression_data0.set_zigzagArray();

  compression_data1.set_percent(percent);
  compression_data1.set_maxIterations(maxIterations);
  compression_data1.set_nBits(nBits);
  compression_data1.set_numBlocks(numBlocks);
  compression_data1.set_numCols(numCols);
  compression_data1.set_dims(dims);
  compression_data1.set_paddedDims(g_paddedDims);

  compression_data1.set_dampingArray();
  compression_data1.set_zigzagArray();
  compression_data1.set_dampingArrayList();

  compression_data2.set_percent(percent);
  compression_data2.set_maxIterations(maxIterations);
  compression_data2.set_nBits(nBits);
  compression_data2.set_numBlocks(numBlocks);
  compression_data2.set_numCols(numCols);
  compression_data2.set_dims(dims);
  compression_data2.set_paddedDims(g_paddedDims);

  compression_data2.set_dampingArray();
  compression_data2.set_zigzagArray();
  compression_data2.set_dampingArrayList();
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

  // TuneGamma(block, blockNumber, col, &compression_data0, &damp);
  TuneGammaVerbose(block, blockNumber, col, &compression_data0, &damp);

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
// visualize a particular block after preprocess; i.e.,
// after DCT and normalization/quantization
////////////////////////////////////////////////////////
void BlockVisualizationTest(int blockNumber, int col, COMPRESSION_DATA* data)
{
  vector<FIELD_3D> blocks;
  GetBlocks(F0, &blocks);
  // UnitaryBlockDCT(1, &blocks);

  // PreprocessBlock(&(blocks[blockNumber]), blockNumber, col, data);
  // dump to a viewer using the FIELD_3D macro
  // FIELDVIEW3D(blocks[blockNumber]);

  string name0("Field3DX_col_");
  name0 += to_string(col);
  name0 += "_block_";
  name0 += to_string(blockNumber);
  FIELD_3D blockX = blocks[blockNumber];
  // blockX.write(name0.c_str()); 
  // FIELDVIEW3D(blockX);
  blocks.clear();
  GetBlocks(F1, &blocks);
  // UnitaryBlockDCT(1, &blocks);
  string name1("Field3DY_col_");
  name1 += to_string(col);
  name1 += "_block_";
  name1 += to_string(blockNumber);
  FIELD_3D blockY = blocks[blockNumber];
  // blockY.write(name1.c_str()); 
  blocks.clear();
  GetBlocks(F2, &blocks);
  // UnitaryBlockDCT(1, &blocks);
  string name2("Field3DZ_col_");
  name2 += to_string(col);
  name2 += "_block_";
  name2 += to_string(blockNumber);
  FIELD_3D blockZ = blocks[blockNumber];
  // blockZ.write(name2.c_str()); 
  VECTOR3_FIELD_3D blockVectorField(blockX.data(), blockY.data(), blockZ.data(), blockX.xRes(), blockX.yRes(), blockX.zRes()); 
  string name("VectorField_col_");
  name += to_string(col);
  name += "_block_";
  name += to_string(blockNumber);
  blockVectorField.write(name.c_str());
}

void FullColumnVisualizationTest(int col, int component)
{
  VECTOR3_FIELD_3D fullColumn = VECTOR3_FIELD_3D(Q[0], xRes, yRes, zRes);
  FIELD_3D fullColumn0, fullColumn1, fullColumn2;
  GetScalarFields(fullColumn, &fullColumn0, &fullColumn1, &fullColumn2);

  if (component == 0) {
    FIELDVIEW3D(fullColumn0);
  }

  else if (component == 1) {
   FIELDVIEW3D(fullColumn1);
  }

  else if (component == 2) {
   FIELDVIEW3D(fullColumn2);
  } 
}

// Read from a binary matrix file and put one of its columns into a VECTOR
void ReadColumnFromMatrix(int col, const char* fileName, VECTOR* result)
{
  FILE* file = fopen(fileName, "rb");
  int numRows = 0;
  fread(&numRows, sizeof(int), 1, file);
  int numCols = 0;
  fread(&numCols, sizeof(int), 1, file);
  result->resizeAndWipe(numRows);
  // move to the first entry of the column once before the loop begins
  fseek(file, col * sizeof(double), SEEK_CUR);
  for (int row = 0; row < numRows; row++) {
    fread(result->data() + row, sizeof(double), 1, file);
    // move backward 1 since fread moved it ahead one
    fseek(file, -1 * sizeof(double), SEEK_CUR);
    fseek(file, numCols * sizeof(double), SEEK_CUR);
  }
  fclose(file);
}

void ReadColumnFromMatrixTest(int col)
{
  MATRIX m(3, 3);
  m(0,0) = 1.0;
  m(1,0) = 2.0;
  m(2,0) = 3.0;
  m(0,1) = 4.0;
  m(1,1) = 5.0;
  m(2,1) = 6.0;
  m(0,2) = 7.0;
  m(1,2) = 8.0;
  m(2,2) = 9.0;
  std::cout << "Here is the matrix m:\n" << m << std::endl;
  
  const char* filename = "m.matrix";
  m.write(filename);
  VECTOR result;
  ReadColumnFromMatrix(col, filename, &result);  
  cout << "Column " << col << ": " << endl;
  cout << result << endl;

}

// Read from a binary big matrix file and put one of its columns into a VECTOR
void ReadColumnFromBigMatrix(int col, const char* fileName, VECTOR* result)
{
  FILE* file = fopen(fileName, "rb");
  int numRows = 0;
  fread(&numRows, sizeof(int), 1, file);
  cout << "numRows: " << numRows << endl;
  int numCols = 0;
  fread(&numCols, sizeof(int), 1, file);
  cout << "numCols: " << numCols << endl;
  result->resizeAndWipe(numRows);
  // each column is headed with the number of entries as an int
  unsigned long colSizeInBytes = sizeof(int) + numRows * sizeof(double);
  cout << "colSizeInBytes: " << colSizeInBytes << endl;
  // move to the first entry of the column once before the loop begins
  fseek(file, col * colSizeInBytes, SEEK_CUR);
  // skip the header int for number of entries
  fseek(file, 1 * sizeof(int), SEEK_CUR);
  // now we can read in the entire vector in one block
  fread(result->data(), sizeof(double), numRows, file);
  fclose(file);
}

void FullColumnVisualizationFromFile(int col, const char* fileName, int component)
{
  VECTOR result;
  ReadColumnFromMatrix(col, fileName, &result);
  VECTOR3_FIELD_3D fullColumn = VECTOR3_FIELD_3D(result, xRes, yRes, zRes);
  FIELD_3D fullColumn0, fullColumn1, fullColumn2;
  GetScalarFields(fullColumn, &fullColumn0, &fullColumn1, &fullColumn2);

  if (component == 0) {
    FIELDVIEW3D(fullColumn0);
  }

  else if (component == 1) {
   FIELDVIEW3D(fullColumn1);
  }

  else if (component == 2) {
   FIELDVIEW3D(fullColumn2);
  } 
}

void FullColumnVisualizationFromBigMatrixFileNaive(int col, const char* fileName, int component)
{
  BIG_MATRIX Q;
  Q.read(fileName);
  VECTOR result = Q[col];
  VECTOR3_FIELD_3D fullColumn = VECTOR3_FIELD_3D(result, xRes, yRes, zRes);
  FIELD_3D fullColumn0, fullColumn1, fullColumn2;
  GetScalarFields(fullColumn, &fullColumn0, &fullColumn1, &fullColumn2);

  if (component == 0) {
    FIELDVIEW3D(fullColumn0);
  }

  else if (component == 1) {
   FIELDVIEW3D(fullColumn1);
  }

  else if (component == 2) {
   FIELDVIEW3D(fullColumn2);
  } 
}

void FullColumnVisualizationFromBigMatrixFile(int col, const char* fileName, int component)
{
  VECTOR result;
  ReadColumnFromBigMatrix(col, fileName, &result);
  VECTOR3_FIELD_3D fullColumn = VECTOR3_FIELD_3D(result, xRes, yRes, zRes);
  FIELD_3D fullColumn0, fullColumn1, fullColumn2;
  GetScalarFields(fullColumn, &fullColumn0, &fullColumn1, &fullColumn2);

  if (component == 0) {
    FIELDVIEW3D(fullColumn0);
  }

  else if (component == 1) {
   FIELDVIEW3D(fullColumn1);
  }

  else if (component == 2) {
   FIELDVIEW3D(fullColumn2);
  } 
}
////////////////////////////////////////////////////////
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
  DecodeBlockWithCompressionData(quantized, blockNumber, col, &compression_data0, decoded.data());

  cout << "newBlock: " << endl;
  cout << decoded.flattened() << endl;

  double newEnergy = decoded.sumSq();
  double diff = abs(oldEnergy - newEnergy) / oldEnergy; 
  cout << "Percent error from encoding and decoding: " << diff << endl;
  cout << "Accuracy was within: " << (1 - diff) << endl;
}

////////////////////////////////////////////////////////
// check the error between encoding and decoding a block
// using quantized cached gamma values
////////////////////////////////////////////////////////
void EncodeDecodeBlockTestNoFastPow()
{
  int col = 0;
  int blockNumber = 1;
 
  VECTOR3_FIELD_3D U0 = VECTOR3_FIELD_3D(U.col(col), xRes, yRes, zRes);
  FIELD_3D U0Field = U0.scalarField(0);
  vector<FIELD_3D> blocks;
  GetBlocks(U0Field, &blocks);
  UnitaryBlockDCT(1, &blocks);
  INTEGER_FIELD_3D quantized(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
  double oldEnergy = blocks[blockNumber].sumSq();
  cout << "original block: " << endl;
  cout << blocks[blockNumber].flattened() << endl;
  EncodeBlock(blocks[blockNumber], blockNumber, col, &compression_data0, &quantized); 
  cout << "DCT and damped:" << endl;
  cout << quantized.flattened() << endl;
  const INTEGER_FIELD_3D& zigzagArray = compression_data0.get_zigzagArray();
  VectorXi zigzagged;
  ZigzagFlatten(quantized, zigzagArray, &zigzagged);
  cout << "Zigzagged: " << endl;
  cout << EIGEN::convertInt(zigzagged) << endl;

  NONZERO_ENTRIES nonZeros;
  FIELD_3D decoded(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
  nonZeros.clear();
  decoded.clear();
  INTEGER_FIELD_3D parsedDataField(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
  parsedDataField.clear();
  RunLengthDecodeBinaryInPlaceSparse(allData0, blockNumber, col, zigzagArray, &compression_data0, parsedDataField, nonZeros);
  DecodeBlockWithCompressionDataSparseNoFastPow(parsedDataField, blockNumber, col, &compression_data0, decoded.data(), nonZeros);
  
  cout << "Unzigzagged and run length decoded: " << endl;
  cout << decoded.flattened() << endl;


  double* buffer = (double*) fftw_malloc(BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE * sizeof(double));
  int direction = -1;
  Create_DCT_Plan(buffer, direction, &plan);

  DCT_Smart_Unitary(plan, direction, buffer, &decoded);
  cout << "newBlock after IDCT" << endl;
  cout << decoded.flattened() << endl;

  fftw_free(buffer);
  fftw_destroy_plan(plan);
  fftw_cleanup();
 

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
// test the compression and writing to a binary file
// of a full matrix using gamma as zero everywhere
////////////////////////////////////////////////////////
void MatrixCompressionDebugTest()
{
  const char* filename = "U.preadvect.compressed.matrix";
  CompressAndWriteMatrixComponentsDebug(filename, U, &compression_data0,
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

////////////////////////////////////////////////////////
// test the run-length decoding's sparsity using the
// passed in inputs
////////////////////////////////////////////////////////
void RunLengthSparseTest(COMPRESSION_DATA* compression_data, const char* filename, string outStr)
{
  int* allData = ReadBinaryFileToMemory(filename, compression_data);
  const INTEGER_FIELD_3D& reverseZigzag = compression_data->get_zigzagArray();
  // int numBlocks = compression_data->get_numBlocks();
  
  cout << " Num blocks: " << numBlocks << endl;
  cout << " Block size: " << BLOCK_SIZE << endl;
  int numCols = compression_data->get_numCols();
  MatrixXd percents(numBlocks, numCols);
  INTEGER_FIELD_3D parsedDataField(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
  
  for (int col = 0; col < numCols; col++) {
    for (int blockNumber = 0; blockNumber < numBlocks; blockNumber++) {
      NONZERO_ENTRIES nonZeros;
        double s = RunLengthDecodeBinaryInPlaceSparseGetSparsity(allData, blockNumber, col, reverseZigzag, compression_data, parsedDataField, nonZeros);
          percents(blockNumber, col) = s;
    }
  }
  EIGEN::write(outStr, percents);
}

////////////////////////////////////////////////////////
// test whether the indices are built correctly
////////////////////////////////////////////////////////
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
  DecodeBlockWithCompressionData(unflattened, blockNumber, col, data, decoded.data());
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
  // const char* filename = "U.preadvect.SVD.data";
  // ReadSVDData(filename, &compression_data0);

  // read all three 'allDatas' to memory
  // const char* filename = "U.preadvect.compressed.matrix0";
  const char* filename = "U.final.compressed.matrix0";
  allData0 = ReadBinaryFileToMemory(filename, &compression_data0);
  // filename = "U.preadvect.compressed.matrix1";
  filename = "U.final.compressed.matrix1";
  allData1 = ReadBinaryFileToMemory(filename, &compression_data1);
  // filename = "U.preadvect.compressed.matrix2";
  filename = "U.final.compressed.matrix2";
  allData2 = ReadBinaryFileToMemory(filename, &compression_data2);

  // call the constructor
  matrix_data = MATRIX_COMPRESSION_DATA(allData0, allData1, allData2, 
      &compression_data0, &compression_data1, &compression_data2);

  // initialize dct information 
  int direction = -1;
  matrix_data.dct_setup(direction);

  // initialize the cache for GetSubmatrix
  matrix_data.init_cache();
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


////////////////////////////////////////////////////////
// compare the frequency domain projection to the ground truth
////////////////////////////////////////////////////////
void PeeledCompressedProjectTransformTest()
{
  InitMatrixCompressionData();


  VECTOR3_FIELD_3D randomV(xPadded, yPadded, zPadded);
  randomV.setToRandom();
  VectorXd q;
  PeeledCompressedProjectTransform(randomV, &matrix_data, &q);
  cout << "q from transform projection: " << endl;
  cout << EIGEN::convert(q) << endl;
 
  MatrixXd decodedMatrix;
  DecodeMatrix(&matrix_data, &decodedMatrix);
  VectorXd q_spatial = randomV.peeledProject(decodedMatrix);
  cout << "q from regular projection: " << endl;
  cout << EIGEN::convert(q_spatial) << endl;

  double diff = (q - q_spatial).norm();
  cout << "projection diff: " << diff << endl;

  VectorXd ground = randomV.peeledProject(U);
  cout << "ground truth for q: " << endl;
  cout << EIGEN::convert(ground) << endl;

  diff = (q - ground).norm();
  cout << "projection diff: " << diff << endl;

  diff = (q_spatial - ground).norm();
  cout << "what projection diff should be: " << diff << endl;



}

////////////////////////////////////////////////////////
// test the decoding of the full matrix
////////////////////////////////////////////////////////
void DecodeMatrixTest() 
{
  InitMatrixCompressionData();

  MatrixXd decodedMatrix;
  DecodeMatrix(&matrix_data, &decodedMatrix);

  EIGEN::write("U.preadvect.decoded.debug.matrix", decodedMatrix);

}

////////////////////////////////////////////////////////
// test the eigen version of decode scalar field
////////////////////////////////////////////////////////
void DecodeScalarFieldEigenTest(int col, COMPRESSION_DATA* data)
{
  const char* filename = "U.preadvect.compressed.matrix0";
  
  int* allData = ReadBinaryFileToMemory(filename, data);
  
  vector<VectorXd> decoded;
  DecodeScalarFieldEigen(data, allData, col, &decoded);

  FILE* pFile;
  pFile = fopen("U.preadvect.decoded.col150.x", "wb");

  if (pFile == NULL) {
    perror("Error opening file");
    exit(EXIT_FAILURE);
  }

  // write out the blocks for debugging
  int totalCells = decoded.size() * decoded[0].size();
  fwrite(&totalCells, sizeof(int), 1, pFile);

  int numBlocks = data->get_numBlocks();
  for (int i = 0; i < numBlocks; i++) {
    fwrite(decoded[i].data(), sizeof(double), decoded[i].size(), pFile);
  }

  // write out the 'ground truth' blocks 
  pFile = fopen("U.preadvect.col150.x", "wb");

  if (pFile == NULL) {
    perror("Error opening file");
    exit(EXIT_FAILURE);
  }

  totalCells = g_blocks.size() * g_blocks[0].totalCells();
  fwrite(&totalCells, sizeof(int), 1, pFile);
  
  for (int i = 0; i < numBlocks; i++) {
    fwrite(g_blocks[i].data(), sizeof(double), g_blocks[i].totalCells(), pFile);
  }

  fclose(pFile);

}


////////////////////////////////////////////////////////
// test the eigen version of block dct
////////////////////////////////////////////////////////
void UnitaryBlockDCTEigenTest() 
{
  // do the non-eigen version first
  GetBlocks(F, &g_blocks);
  UnitaryBlockDCT(1, &g_blocks);

  // now do the eigen one
  GetBlocksEigen(F, &g_blocksEigen);
  UnitaryBlockDCTEigen(1, &g_blocksEigen);

  // compare
  double error = 0.0;
  for (int i = 0; i < numBlocks; i++) {
    error += (g_blocks[i].flattenedEigen() - g_blocksEigen[i]).norm();
  }

  cout << "accumulated error across each block: " << error << endl;

}

////////////////////////////////////////////////////////
// test the projection naively in the spatial domain
////////////////////////////////////////////////////////
void PeeledCompressedProjectTest()
{
  VECTOR3_FIELD_3D randomV(xPadded, yPadded, zPadded);
  randomV.setToRandom();

  VectorXd ground = randomV.peeledProject(U);
  cout << "ground: " << endl;
  cout << EIGEN::convert(ground) << endl;

  VectorXd spatial;
  PeeledCompressedProject(randomV, &matrix_data, &spatial);
  cout << "spatial: " << endl;
  cout << EIGEN::convert(spatial) << endl;

  VectorXd freq;
  PeeledCompressedProjectTransformNoSVD(randomV, &matrix_data, &freq);
  cout << "freq: " << endl;
  cout << EIGEN::convert(freq) << endl;

}
////////////////////////////////////////////////////////
// test the project transform without using compression
////////////////////////////////////////////////////////
void PeeledProjectTransformTest()
{

  VECTOR3_FIELD_3D randomV(xPadded, yPadded, zPadded);
  randomV.setToRandom();

  // ground truth projection
  VectorXd ground = randomV.peeledProject(U);

  // fill svdV
  VectorXd s;
  MatrixXd svdV;
  VECTOR3_FIELD_3D randomVcopy = randomV;
  TransformVectorFieldSVD(&s, &svdV, &randomV);

  FIELD_3D V_X, V_Y, V_Z;
  GetScalarFields(randomV, &V_X, &V_Y, &V_Z);

  vector<VectorXd> blocksX, blocksY, blocksZ;
  GetBlocksEigen(V_X, &blocksX);
  GetBlocksEigen(V_Y, &blocksY);
  GetBlocksEigen(V_Z, &blocksZ);
  UnitaryBlockDCTEigen(1, &blocksX);
  UnitaryBlockDCTEigen(1, &blocksY);
  UnitaryBlockDCTEigen(1, &blocksZ);



}

////////////////////////////////////////////////////////
// test the projection timings on different blocks of
// varying sparsity
////////////////////////////////////////////////////////
void ProjectSparsityTimingTest(int col, int blockNumber, int numIterations)
{
  cout << "Doing " << numIterations << " iterations..." << endl;
  VECTOR3_FIELD_3D random_V(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
  random_V.setToRandom();
  double q = 0.0;
  {
    TIMER functionTimer("Project loop timing USE THIS");
    for (int i = 0; i < numIterations; i++) {
      PeeledCompressedProjectTransformNoSVDOneBlock(random_V, &matrix_data, col, blockNumber, &q);
    }
  }
  cout << "...done!" << endl;
  cout << "q: " << q << endl;
}
////////////////////////////////////////////////////////
// test the unprojection
////////////////////////////////////////////////////////
void PeeledCompressedUnprojectTest()
{

  InitMatrixCompressionData();

  // initialize a random vector in the subspace 
  VectorXd q;
  q.setRandom(numCols);

  // run the compressed unprojector
  VECTOR3_FIELD_3D unprojected(xPadded, yPadded, zPadded);
  PeeledCompressedUnproject(&matrix_data, q, &unprojected);
  
  FILE* pFile;
  pFile = fopen("compressed.unprojectedV", "wb");
  if (pFile == NULL) {
    perror("Error opening file");
  }
  int length = 3 * xRes * yRes * zRes;
  fwrite(&length, sizeof(int), 1, pFile);
  fwrite(unprojected.peelBoundary().flattenedEigen().data(), sizeof(double), length, pFile);

  // compute the ground truth (no compression)
  VECTOR3_FIELD_3D ground(xPadded, yPadded, zPadded);
  ground.peeledUnproject(U, q);

  pFile = fopen("uncompressed.unprojectedV", "wb");
  if (pFile == NULL) {
    perror("Error opening file");
  }
  fwrite(&length, sizeof(int), 1, pFile);
  fwrite(ground.peelBoundary().flattenedEigen().data(), sizeof(double), length, pFile);

  // compare
  double groundLength = ground.peelBoundary().flattenedEigen().norm();
  double error = ( unprojected.peelBoundary().flattenedEigen() - ground.peelBoundary().flattenedEigen() ).norm();
  error /= groundLength;
  cout << "error between compressed unproject and no compression: " << error << endl;

  fclose(pFile);

}


////////////////////////////////////////////////////////
// a mini version of GetSubmatrix for just one cell
////////////////////////////////////////////////////////
void DecodeFromRowColTest(int row, int col) 
{

  InitMatrixCompressionData();

  Vector3d cell;
  DecodeFromRowCol(row, col, &matrix_data, &cell);
  cout << "Row, col: (" << row << ", " << col << ")" << endl;
  cout << "DecodeFromRowCol: " << endl;
  cout << cell << endl;

  Vector3d ground;
  ground[0] = U(row, col);
  ground[1] = U(row + 1, col);
  ground[2] = U(row + 2, col);
  cout << "Ground: " << endl;
  cout << ground << endl;

}


////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////
void GetSubmatrixTest(int startRow)
{
  
  MatrixXd submatrix;
  GetSubmatrix(startRow, &matrix_data, &submatrix);

  // cout << "Submatrix from GetSubmatrixTest: " << endl;
  // cout << submatrix << endl;

  MatrixXd ground = U.block(startRow, 0, 3, numCols);
  // cout << "Ground: " << endl;
  // cout << ground << endl;

  double diff = (ground - submatrix).norm();
  cout << "Diff: " << diff << endl; 

}


////////////////////////////////////////////////////////
// 
////////////////////////////////////////////////////////
void PeeledCompressedUnprojectTransformTest()
{

  InitMatrixCompressionData();

  // initialize a random vector in the subspace 
  VectorXd q;
  q.setRandom(numCols);

  // run the compressed unprojector in frequency space
  VECTOR3_FIELD_3D unprojected(xPadded, yPadded, zPadded);
  PeeledCompressedUnprojectTransform(&matrix_data, q, &unprojected);
  
  FILE* pFile;
  pFile = fopen("compressed.unprojected.freq.V", "wb");
  if (pFile == NULL) {
    perror("Error opening file");
  }
  int length = 3 * xRes * yRes * zRes;
  fwrite(&length, sizeof(int), 1, pFile);
  fwrite(unprojected.peelBoundary().flattenedEigen().data(), sizeof(double), length, pFile);

  // compute the ground truth (no compression)
  VECTOR3_FIELD_3D ground(xPadded, yPadded, zPadded);
  ground.peeledUnproject(U, q);

  pFile = fopen("uncompressed.unprojectedV", "wb");
  if (pFile == NULL) {
    perror("Error opening file");
  }
  fwrite(&length, sizeof(int), 1, pFile);
  fwrite(ground.peelBoundary().flattenedEigen().data(), sizeof(double), length, pFile);

  // compare
  double groundLength = ground.peelBoundary().flattenedEigen().norm();
  double error = ( unprojected.peelBoundary().flattenedEigen() - ground.peelBoundary().flattenedEigen() ).norm();
  error /= groundLength;
  cout << "error between compressed unproject and no compression: " << error << endl;

  fclose(pFile);

}

////////////////////////////////////////////////////////
// Write the gammaListMatrix matrix to a binary file
// for analysis in MATLAB
////////////////////////////////////////////////////////
void GammaAnalyticsTest(COMPRESSION_DATA* data)
{
  int* allData = ReadBinaryFileToMemoryGammaTesting("U.final.component0", data);
  system("mv gammaListMatrix.matrix gammaListMatrixFinal0.matrix");

  allData = ReadBinaryFileToMemoryGammaTesting("U.final.component1", data);
  system("mv gammaListMatrix.matrix gammaListMatrixFinal1.matrix");

  allData = ReadBinaryFileToMemoryGammaTesting("U.final.component2", data);
  system("mv gammaListMatrix.matrix gammaListMatrixFinal2.matrix");

}

////////////////////////////////////////////////////////
// Read in a FIELD_3D from a binary file, take its
// block DCT, and write it out to a file
////////////////////////////////////////////////////////
void DoBlockDCTFromFile(string fieldFile, string outputFile)
{
  FIELD_3D X;
  X.read(fieldFile);

  VEC3I paddedDims(0, 0, 0);
  GetPaddings(dims, &paddedDims);
  dims += paddedDims;
  FIELD_3D assimilatedTransformedF(dims[0], dims[1], dims[2]);

  vector<FIELD_3D> blocks;
  GetBlocks(X, &blocks);

  UnitaryBlockDCT(1, &blocks);

  AssimilateBlocks(dims, blocks, &assimilatedTransformedF);
  assimilatedTransformedF.write(outputFile);
  

}
