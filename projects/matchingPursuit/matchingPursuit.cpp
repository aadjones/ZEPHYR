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
 Basic orthogonal matching pursuit implementation
 June 2014 Aaron Demby Jones
 */

#include <iostream>
#include "EIGEN.h"

#include "SUBSPACE_FLUID_3D_EIGEN.h"
#include "FLUID_3D_MIC.h"
#include "CUBATURE_GENERATOR_EIGEN.h"
#include "MATRIX.h"
#include "SIMPLE_PARSER.h"

#define DBL_EPSILON 2.2204460492503131e-16     // use a library instead

//////////////////////////////////////////////////////////////////////////////
// Function signature for naiveOMP. 
// Inputs:
// - signal x
// - dictionary dict
// - number of atoms in dictionary, natom
// - tolerance, a real number between 0 and 1 
// Outputs:
// - vector containing the coefficients of each corresponding atom
//////////////////////////////////////////////////////////////////////////////

VECTOR naiveOMP(VECTOR x, MATRIX dict, int natom, double tolerance);

//////////////////////////////////////////////////////////////////////////////
// Function signature for getMaxIndex.
// Inputs:
// - vector x
// Outputs:
// - index of the maximum squared component
//////////////////////////////////////////////////////////////////////////////

int getMaxIndex(const VECTOR& x);

//////////////////////////////////////////////////////////////////////////////
// Function signature for getPseudoInverse
// Inputs:
// - an m x n matrix, A
// Outputs:
// - the n x m Moore-Penrose pseudoinverse of A
// - destroys the input, A!
//////////////////////////////////////////////////////////////////////////////
MATRIX getPseudoInverse(MATRIX A);

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
// Main
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
 
int main(int argc, char* argv[])
{
   VECTOR x("signal.vector");
   MATRIX dict("dict.matrix");
   VECTOR s = naiveOMP(x, dict, 25, 0.05);
   s.write("result.vector");
   VECTOR mpReconstruct = dict * s;
   mpReconstruct.write("reconstruct.vector");
   return 0; 
}

//////////////////////////////////////////////////////////////////////////////
// Function implementations
//////////////////////////////////////////////////////////////////////////////

int getMaxIndex(const VECTOR& x) {
  double max = x[0]*x[0];
  int maxIndex = 0;
  for (int i = 0; i < x.size(); i++) {
    if (max <= x[i]*x[i]) {
      max = x[i]*x[i];
      maxIndex = i;
    }
  }
  return maxIndex;
}

MATRIX getPseudoInverse(MATRIX A) {
  int bigger = (A.rows() > A.cols()) ? A.rows() : A.cols();
  int smaller = (A.rows() > A.cols()) ? A.cols() : A.rows(); 
  MATRIX U, VT;                                                                    // initialize matrices for singular-value decomposition
  VECTOR S;
  A.SVD(U, S, VT);   
  MATRIX SM(A.cols(), A.rows());                                                   // initialize the n x m matrix SM = pinv(S)
  SM.clear();
  double threshold = 0.5 * sqrt(A.cols() + A.rows() + 1) * S(0) * DBL_EPSILON;     // threshold copied from T. Kim
  for (int x = 0; x < smaller; x++) {
    SM(x, x) = (S(x) > threshold) ? 1.0/S(x) : 0.0;                                // all nonzero diagonal entries are replaced with their reciprocal
  }
  MATRIX V = VT.transpose();
  MATRIX UT = U.transpose();
  MATRIX pinvA = V * SM * UT;                                                      // the formula for the pseudoinverse
  return pinvA;
}

VECTOR naiveOMP(VECTOR x, MATRIX dict, int natom, double tolerance) {
  double normr2 = x.norm2() * x.norm2();
  int natoms = dict.cols();
  double normtol2 = tolerance * normr2; 
  VECTOR residual(x);
  VECTOR indexes(natom);        // keeps track of the indexes of maximum inner product
  indexes.clear();
  vector<VECTOR> subdict;       // keeps track of the corresponding dictionary atoms
  VECTOR stemp(x);              // the temporary output
  VECTOR s(natoms);             // the eventual output
  s.clear();                    // set all the entries to zero
  int k = 0;
  while (normr2 > normtol2 && k < natom) {  
    VECTOR projections = dict.transpose() * residual;      // take inner products
    int index = getMaxIndex(projections);                  // find the maximum index
    indexes[k] = index;                                    // store it in indexes
    subdict.push_back(dict.getColumn(index));              // append the corresponding atom 
    MATRIX H(subdict);                                     // form the matrix with atoms as its columns
    MATRIX Htemp(H);                                       // make a copy of H since H gets destroyed when we call getPseudoInverse
    MATRIX pinvH = getPseudoInverse(H);                    // get the pseudoinverse of H; destroy H
    stemp = pinvH * x;                                     // compute stemp, the iterative solution
    residual = x - Htemp * stemp;                          // compute the residual---note that H * stemp + residual = x
    normr2 = residual.norm2() * residual.norm2();
    k++;
  } 
  for (int i = 0; i < k; i++) {                            
    s(indexes[i]) = stemp[i];                              // fill the relevant entries of the solution s
  } 
  return s;
}
