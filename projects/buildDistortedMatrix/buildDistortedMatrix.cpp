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
 * 2014-2015
 */

#include <iostream>
#include <fftw3.h>

#include "EIGEN.h"
#include "SUBSPACE_FLUID_3D_EIGEN.h"
#include "FLUID_3D_MIC.h"
#include "CUBATURE_GENERATOR_EIGEN.h"
#include "MATRIX.h"
#include "SIMPLE_PARSER.h"
#include "COMPRESSION.h"
#include <string>

using std::vector;
using std::string;

///////////////////////////////////////////////////////
// Globals
////////////////////////////////////////////////////////

string path_to_U("/Users/aaron/Desktop/U.final.uncompressed.matrix");
// string path_to_U("U.final.donotmodify.matrix.48");

// make sure the read-in matrix agrees with the dimensions specified!
/*
const int g_xRes =    46;
const int g_yRes =    62;
const int g_zRes =    46;
const int g_numRows = 3 * g_xRes * g_yRes * g_zRes;
const int g_numCols = 48;
*/

const int g_xRes = 198;
const int g_yRes = 264;
const int g_zRes = 198;
const int g_numRows = 3 * g_xRes * g_yRes * g_zRes;
const int g_numCols = 151;


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
  
   
  vector<VectorXd> columnList(g_numCols);
  for (int col = 0; col < g_numCols; col++) {
    cout << "Col: " << col << endl;
    VectorXd v = g_U.col(col);
    VECTOR3_FIELD_3D V(v, g_xRes, g_yRes, g_zRes);
    VECTOR3_FIELD_3D compressedV = BlockCompressVectorField(V, compression_data);
    VECTOR flattenedV = compressedV.flattened();
    VectorXd flattenedV_eigen = EIGEN::convert(flattenedV);
    columnList[col] = flattenedV_eigen;
  }
  MatrixXd compressedResult = EIGEN::buildFromColumns(columnList);
  EIGEN::write("CompressedUHugepow4.matrix", compressedResult);

  TIMER::printTimings();

  return EXIT_SUCCESS;
}
      
