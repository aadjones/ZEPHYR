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
#include "COMPRESSION_REWRITE.h"
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
void PreprocessEncoder(COMPRESSION_DATA* data0, COMPRESSION_DATA* data1, COMPRESSION_DATA* data2, int maxIterations);

////////////////////////////////////////////////////////
// Main
////////////////////////////////////////////////////////

int main(int argc, char* argv[]) {
  TIMER functionTimer(__FUNCTION__);
  
  // read in the cfg file
  if (argc != 2) {
    cout << " Usage: " << argv[0] << " *.cfg" << endl;
    return 0;
  }
  
  SIMPLE_PARSER parser(argv[1]);
  string reducedPath = parser.getString("reduced path", "./data/reduced.dummy/"); 
  int xRes = parser.getInt("xRes", 48);
  int yRes = parser.getInt("yRes", 64);
  int zRes = parser.getInt("zRes", 48);
  int numCols = parser.getInt("reduced snapshots", 47);
  bool usingIOP = parser.getBool("iop", 0);
  cout << " Using IOP: " << usingIOP << endl;
  
  // we want the peeled resolutions for the matrices
  xRes -= 2;
  yRes -= 2;
  zRes -= 2;
  numCols += 1;

  VEC3I dims(xRes, yRes, zRes);
  
  // times 3 since it is a VELOCITY3_FIELD_3D flattened out
  int numRows = 3 * xRes * yRes * zRes;
  cout << "numRows: " << numRows << endl;
  cout << "numCols: "<< numCols << endl;

  MatrixXd U_preadvect(numRows, numCols);
  MatrixXd U_final(numRows, numCols);

  if (usingIOP) {
    // MatrixXd U(numRows, numCols);
    // U_preproject = U;
  }

  int nBits = parser.getInt("nBits", 24); 
  cout << " nBits: " << nBits << endl;
  double percent = parser.getFloat("percent", 0.99);
  cout << " percent: " << percent << endl;
  int maxIterations = parser.getInt("maxIterations", 32);
  cout << " maxIterations: " << maxIterations << endl;
  int debug = parser.getBool("debug", false);
  cout << "debug: " << debug << endl;

  string preAdvectPath = reducedPath + string("U.preadvect.matrix");
  string finalPath = reducedPath + string("U.final.matrix");
  // string preprojectPath = reducedPath + string("U.preproject.matrix");
 
  EIGEN::read(preAdvectPath, U_preadvect);
  EIGEN::read(finalPath, U_final);
  if (usingIOP) {
    // EIGEN::read(preprojectPath, U_preproject);
  }

  // set the parameters in compression data
  COMPRESSION_DATA preadvect_compression_data0(dims, numCols, nBits, percent);
  COMPRESSION_DATA preadvect_compression_data1(dims, numCols, nBits, percent);
  COMPRESSION_DATA preadvect_compression_data2(dims, numCols, nBits, percent);
  COMPRESSION_DATA final_compression_data0(dims, numCols, nBits, percent);
  COMPRESSION_DATA final_compression_data1(dims, numCols, nBits, percent);
  COMPRESSION_DATA final_compression_data2(dims, numCols, nBits, percent);
  // COMPRESSION_DATA preproject_compression_data;
  if (usingIOP) {
    // COMPRESSION_DATA data(dims, numCols, q, power, nBits);
    // preproject_compression_data = data;
  }

  // compute some additional parameters for compression data
  PreprocessEncoder(&preadvect_compression_data0, &preadvect_compression_data1, &preadvect_compression_data2, 
      maxIterations);
  PreprocessEncoder(&final_compression_data0, &final_compression_data1, &final_compression_data2,
      maxIterations);
  if (usingIOP) {
    // PreprocessEncoder(preproject_compression_data);
  }

  // write a binary file for each scalar field component

  string preadvectFilename = reducedPath + string("U.preadvect.component");
  string finalFilename = reducedPath + string("U.final.component");
  // string preprojectFilename = reducedPath + string("U.preproject.component");

  // write out the compressed matrix files

  if (debug) {
    CompressAndWriteMatrixComponentsDebug(preadvectFilename.c_str(), U_preadvect, &preadvect_compression_data0,
      &preadvect_compression_data1, &preadvect_compression_data2);

    CompressAndWriteMatrixComponentsDebug(finalFilename.c_str(), U_final, &final_compression_data0, 
      &final_compression_data1, &final_compression_data2);
  }

  else {
    CompressAndWriteMatrixComponents(preadvectFilename.c_str(), U_preadvect, &preadvect_compression_data0,
      &preadvect_compression_data1, &preadvect_compression_data2);
  
    CompressAndWriteMatrixComponents(finalFilename.c_str(), U_final, &final_compression_data0, 
      &final_compression_data1, &final_compression_data2);
  }
  
  if (usingIOP) {
    // for (int component = 0; component < 3; component++) {
      // cout << "Writing component: " << component << endl;
      // CompressAndWriteMatrixComponent(preprojectFilename.c_str(), U_preproject, component, preproject_compression_data);
    // }
  }
     

  TIMER::printTimings();
  
  return 0;
}

void PreprocessEncoder(COMPRESSION_DATA* data0, COMPRESSION_DATA* data1, COMPRESSION_DATA* data2, int maxIterations)
{
  // set integer rounding 'to nearest' 
  fesetround(FE_TONEAREST);
  
  // precompute and set the damping  and zigzag arrays
  data0->set_dampingArray();
  data1->set_dampingArray();
  data2->set_dampingArray();

  data0->set_zigzagArray();
  data1->set_zigzagArray();
  data2->set_zigzagArray();
   
  // fill in the appropriate paddings
  const VEC3I& dims = data0->get_dims();
  VEC3I paddings;
  GetPaddings(dims, &paddings);
  paddings += dims;

  data0->set_paddedDims(paddings);
  data1->set_paddedDims(paddings);
  data2->set_paddedDims(paddings);

  // calculates number of blocks, assuming BLOCK_SIZE 
  int numBlocks = paddings[0] * paddings[1] * paddings[2] / (BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE);

  data0->set_numBlocks(numBlocks);
  data1->set_numBlocks(numBlocks);
  data2->set_numBlocks(numBlocks);

  // set the maxIterations
  data0->set_maxIterations(maxIterations);
  data1->set_maxIterations(maxIterations);
  data2->set_maxIterations(maxIterations);
  
}

   


