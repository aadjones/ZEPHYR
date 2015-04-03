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
#include "VECTOR3_FIELD_3D.h"

using std::vector;
using std::string;


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
  
  /*
  // read in the cfg file
  if (argc != 2) {
    cout << " Usage: " << argv[0] << " *.cfg" << endl;
    return 0;
  }
  */

  /*
  SIMPLE_PARSER parser(argv[1]);
  string reducedPath = parser.getString("reduced path", "./data/reduced.dummy/"); 
  int xRes = parser.getInt("xRes", 48);
  int yRes = parser.getInt("yRes", 64);
  int zRes = parser.getInt("zRes", 48);
  int numCols = parser.getInt("reduced snapshots", 47);
  */

  string reducedPath("/Volumes/DataDrive/data/reduced.stam.128.vorticity.1.5/");
  // string reducedPath("/Volumes/DataDrive/data/reduced.stam.200.vorticity.1.5/");
  // string reducedPath("./data/reduced.stam.64/");
  
  // int xRes = 200;
  // int yRes = 266;
  // int zRes = 200;
  // int numCols = 150;
  // int xRes = 48;
  // int yRes = 64;
  // int zRes = 48;
  // int numCols = 47;
  int xRes = 96;
  int yRes = 128;
  int zRes = 96;
  int numCols = 150;
  // we want the peeled resolutions for the matrices
  xRes -= 2;
  yRes -= 2;
  zRes -= 2;
  numCols += 1;

  VEC3I dims(xRes, yRes, zRes);
  
  // times 3 since it is a VELOCITY3_FIELD_3D flattened out
  int numRows = 3 * xRes * yRes * zRes;

  MatrixXd U_preadvect(numRows, numCols);
  MatrixXd U_final(numRows, numCols);

  /*
  int nBits = parser.getInt("nBits", 24); 
  double q = parser.getFloat("linear damping", 1.0);
  double power = parser.getFloat("nonlinear damping", 6.0); 
  */
  int nBits = 24;
  double q = 1.0;
  double power = 6.0;

  string preAdvectPath = reducedPath + string("U.preadvect.matrix");
  string finalPath = reducedPath + string("U.final.matrix");
 
  EIGEN::read(preAdvectPath, U_preadvect);
  EIGEN::read(finalPath, U_final);

  // set the parameters in compression data
  COMPRESSION_DATA preadvect_compression_data(dims, numCols, q, power, nBits);
  COMPRESSION_DATA final_compression_data(dims, numCols, q, power, nBits);

  // compute some additional parameters for compression data
  PreprocessEncoder(preadvect_compression_data);
  PreprocessEncoder(final_compression_data);

  // write a binary file for each scalar field component

  string preadvectFilename = reducedPath + string("U.preadvect.component");
  string finalFilename = reducedPath + string("U.final.component");

  // write out the compressed matrix files
  for (int component = 0; component < 3; component++) {
    cout << "Writing component: " << component << endl;
    CompressAndWriteMatrixComponent(preadvectFilename.c_str(), U_preadvect, component, preadvect_compression_data);
  }  
  
  
  for (int component = 0; component < 3; component++) {
    cout << "Writing component: " << component << endl;
    CompressAndWriteMatrixComponent(finalFilename.c_str(), U_final, component, final_compression_data);
  }  
  

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

   


