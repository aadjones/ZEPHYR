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
#include "ARRAY_4D.h"
#include <string>

using std::vector;
using std::string;

///////////////////////////////////////////////////////
// Globals
////////////////////////////////////////////////////////

// string path_to_U("/Users/aaron/Desktop/U.final.uncompressed.matrix");
string path_to_U("U.final.donotmodify.matrix.48");

// make sure the read-in matrix agrees with the dimensions specified!


const int g_xRes =    46;
const int g_yRes =    62;
const int g_zRes =    46;
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

///////////////////////////////////////////////////////
// End Globals
////////////////////////////////////////////////////////



////////////////////////////////////////////////////////
// Main
////////////////////////////////////////////////////////

  int main(int argc, char* argv[]) {
    
    
    const int nBits = 16;        // don't exceed 16 if using 'short' in the compressor!
    const double q = 1.0;        // change this to modify compression rate
    const double power = 4.0;    // and especially this!
    VEC3I dims(g_xRes, g_yRes, g_zRes);
    COMPRESSION_DATA compression_data(dims, q, power, nBits);
    compression_data.set_numCols(g_numCols);

    // precompute the damping array
    compression_data.set_dampingArray();
    
    EIGEN::read(path_to_U, g_U);
   
    int xPadding;
    int yPadding;
    int zPadding;
    // fill in the appropriate paddings
    GetPaddings(dims, xPadding, yPadding, zPadding);
    // update to the padded resolutions
    int xRes = g_xRes + xPadding;
    int yRes = g_yRes + yPadding;
    int zRes = g_zRes + zPadding;
    // calculates number of blocks, assuming an 8 x 8 x 8 block size.
    int numBlocks = xRes * yRes * zRes / (8 * 8 * 8);
    compression_data.set_numBlocks(numBlocks);
    
    const char* filename = "runLength.bin";
     
    vector<VectorXd> columnList;
    for (int col = 0; col < g_numCols; col++) {
      VectorXd v = g_U.col(col);
      VECTOR v_vector = EIGEN::convert(v);
      VECTOR3_FIELD_3D V(v_vector, g_xRes, g_yRes, g_zRes);
      VECTOR3_FIELD_3D compressedV = SmartBlockCompressVectorField(V, compression_data);
      VECTOR flattenedV = compressedV.flattened();
      VectorXd flattenedV_eigen = EIGEN::convert(flattenedV);
      columnList.push_back(flattenedV_eigen);
    }
    MatrixXd compressedResult = EIGEN::buildFromColumns(columnList);
    EIGEN::write("newcompressedU.matrix", compressedResult);
     

    // write a binary file for each scalar field component
    /* 
    for (int component = 0; component < 3; component++) {
      CompressAndWriteMatrixComponent(filename, g_U, component, compression_data);
    }
    */
    
    /*
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
    
    // test the decompressor on a (row, col)    
    int row = g_numRows - 1;
    int col = g_numCols - 1;
    // DecodeFromRowCol will print the decompressed value at (row, col) and return the entire decompressed block
    FIELD_3D test_block = DecodeFromRowCol(row, col, allDataX, allDataY, allDataZ, compression_data, decompression_dataX, decompression_dataY, decompression_dataZ);
    
    double trueValue = g_U(row, col);
    cout << "True value: " << trueValue << endl;
    */

    return 0;
  }

