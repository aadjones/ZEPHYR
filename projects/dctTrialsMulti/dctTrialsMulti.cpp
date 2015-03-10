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
#include <fftw3.h>
#include "jo_jpeg.h"
#include "EIGEN.h"
#include "SUBSPACE_FLUID_3D_EIGEN.h"
#include "FLUID_3D_MIC.h"
#include "CUBATURE_GENERATOR_EIGEN.h"
#include "MATRIX.h"
#include "BIG_MATRIX.h"
#include "SIMPLE_PARSER.h"
#include "COMPRESSION.h"
#include <string>
#include <cmath>
#include <cfenv>
#include <climits>

using std::vector;
using std::string;

///////////////////////////////////////////////////////
// Globals
////////////////////////////////////////////////////////

// string path_to_U("/Users/aaron/Desktop/U.final.uncompressed.matrix");
// string path_to_U("/Volumes/DataDrive/data/reduced.stam.200.vorticity.1.5/U.preadvect.matrix");
string path_to_U("U.final.donotmodify.matrix.48");

// make sure the read-in matrix agrees with the dimensions specified!


const int g_xRes =    46;
const int g_yRes =    62;
const int g_zRes =    46;
const VEC3I g_dims(g_xRes, g_yRes, g_zRes);
const int g_numRows = 3 * g_xRes * g_yRes * g_zRes;
const int g_numCols = 48;

/*
const int g_xRes = 198;
const int g_yRes = 264;
const int g_zRes = 198;
const int g_numRows = 3 * g_xRes * g_yRes * g_zRes;
const int g_numCols = 151;
*/

MatrixXd g_U(g_numRows, g_numCols);

////////////////////////////////////////////////////////
// End Globals
////////////////////////////////////////////////////////

////////////////////////////////////////////////////////
// Function Declarations
////////////////////////////////////////////////////////

// set the damping matrix and compute the number of blocks
void PreprocessEncoder(COMPRESSION_DATA& data);

////////////////////////////////////////////////////////
// Main
////////////////////////////////////////////////////////

int main(int argc, char* argv[]) {
  
  TIMER functionTimer(__FUNCTION__);
  
  EIGEN::read(path_to_U, g_U);

  // Compression parameters    
  const int nBits = 16;
  const double q = 1.0;
  const double power = 4.0;

  // set the parameters in compression data
  COMPRESSION_DATA compression_data(g_dims, g_numCols, q, power, nBits);

  // compute some additional parameters for compression data
  PreprocessEncoder(compression_data);

  // Old version to get the distorted matrix. Currenty has memory leak problems
  // for huge matrices!!!
   
  /*
  vector<VectorXd> columnList;
  for (int col = 0; col < g_numCols; col++) {
    cout << "Column: " << col << endl;
    VectorXd v = g_U.col(col);
    VECTOR v_vector = EIGEN::convert(v);
    VECTOR3_FIELD_3D V(v_vector, g_xRes, g_yRes, g_zRes);
    VECTOR3_FIELD_3D compressedV = SmartBlockCompressVectorField(V, compression_data);
    VECTOR flattenedV = compressedV.flattened();
    VectorXd flattenedV_eigen = EIGEN::convert(flattenedV);
    columnList.push_back(flattenedV_eigen);
  }
  MatrixXd compressedResult = EIGEN::buildFromColumns(columnList);

  EIGEN::write("newcompressedUhuge.matrix", compressedResult);
  */




  // write a binary file for each scalar field component

  /* 
  const char* filename = "runLength.bin";
  for (int component = 0; component < 3; component++) {
    cout << "Writing component: " << component << endl;
    CompressAndWriteMatrixComponent(filename, g_U, component, compression_data);
  }
  */
  //////////////////////////////////////////////////////////////////
  // currently in debug mode so no compression damping is happening!
  //////////////////////////////////////////////////////////////////

  
  // preprocessing for the decoder
  short* allDataX;
  short* allDataY;
  short* allDataZ;
  DECOMPRESSION_DATA decompression_dataX;
  DECOMPRESSION_DATA decompression_dataY;
  DECOMPRESSION_DATA decompression_dataZ;

  // fill allData and decompression_data
  ReadBinaryFileToMemory("runLength.binX", allDataX, compression_data, decompression_dataX); 
  ReadBinaryFileToMemory("runLength.binY", allDataY, compression_data, decompression_dataY);
  ReadBinaryFileToMemory("runLength.binZ", allDataZ, compression_data, decompression_dataZ);

  // set the entirety of the data for the decoder into one package
  MATRIX_COMPRESSION_DATA matrixData(compression_data, allDataX, allDataY, allDataZ, 
      decompression_dataX, decompression_dataY, decompression_dataZ);



 
  // test the decompressor on a (row, col)   
 
  int row = 8;
  int col = 32;

  double testValue = DecodeFromRowCol(row, col, matrixData);

  cout << "Test value: " << testValue << endl;
  double trueValue = g_U(row, col);
  cout << "True value: " << trueValue << endl;
  
  // use the decompressor to get a 3 x numCols submatrix of U
  
  int startRow = 23;
  int numRows = 3;
  MATRIX subMatrix = GetSubmatrix(startRow, numRows, matrixData); 
  
  // EIGEN is giving a bizarre malloc error, calling free on something that has already been freed
  // (or never been allocated)
  // EIGEN::write("sub23.matrix", subMatrix);

  subMatrix.write("sub23.matrix");
  
  
  TIMER::printTimings();

  return 0;
}



void PreprocessEncoder(COMPRESSION_DATA& data) {

  // set integer rounding 'to nearest' 
  fesetround(FE_TONEAREST);
  
  VEC3I dims = data.get_dims();
  int xRes = dims[0];
  int yRes = dims[1];
  int zRes = dims[2];
  
  // precompute and set the damping array.
  // this can only be executed after q and power are initialized!
  data.set_dampingArray();

  // this can actually be executed at any time
  data.set_zigzagArray();
   
  int xPadding;
  int yPadding;
  int zPadding;

  // fill in the appropriate paddings
  GetPaddings(dims, xPadding, yPadding, zPadding);

  // update to the padded resolutions
  xRes += xPadding;
  yRes += yPadding;
  zRes += zPadding;

  // calculates number of blocks, assuming an 8 x 8 x 8 block size.
  int numBlocks = xRes * yRes * zRes / (8 * 8 * 8);
  data.set_numBlocks(numBlocks);
  
}
  

