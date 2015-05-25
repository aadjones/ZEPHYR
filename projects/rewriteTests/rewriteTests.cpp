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
FIELD_3D F(xRes, yRes, zRes);
fftw_plan plan;

////////////////////////////////////////////////////////
// Function Declarations
////////////////////////////////////////////////////////

// set up global variables
void InitGlobals();

// check the difference between the original field and
// taking a dct followed by an idct
void DCTTest();

////////////////////////////////////////////////////////
// Main
////////////////////////////////////////////////////////

int main(int argc, char* argv[]) 
{
  TIMER functionTimer(__FUNCTION__);
  InitGlobals();
  DCTTest();

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
  F.setToRandom();
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

