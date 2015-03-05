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
 QR-based orthogonal matching pursuit implementation
 June 2014 Aaron Demby Jones
 */

#include <iostream>
#include "EIGEN.h"

#include "SUBSPACE_FLUID_3D_EIGEN.h"
#include "FLUID_3D_MIC.h"
#include "CUBATURE_GENERATOR_EIGEN.h"
#include "MATRIX.h"
#include "SIMPLE_PARSER.h"

//////////////////////////////////////////////////////////////////////////////
// Function signature for qrOMP. 
// Inputs:
// - signal x
// - dictionary dict, normalized!
// - number of atoms in dictionary, natom
// - tolerance, a real number between 0 and 1 
// Outputs:
// - vector containing the coefficients of each corresponding atom
// - algorithm based on QR decomposition
//////////////////////////////////////////////////////////////////////////////

VECTOR qrOMP(VECTOR x, MATRIX dict, int natom, double tolerance);

//////////////////////////////////////////////////////////////////////////////
// Function signature for getMaxIndex.
// Inputs:
// - vector x
// Outputs:
// - index of the maximum squared component
//////////////////////////////////////////////////////////////////////////////
int getMaxIndex(VECTOR x);

//////////////////////////////////////////////////////////////////////////////
// Function signature for castVectorToMatrix.
// Inputs: 
// - a vector V
// Outputs:
// - a copy of V in a MATRIX structure (n x 1)
MATRIX castVectorToMatrix(VECTOR V);

//////////////////////////////////////////////////////////////////////////////
// Function signature for castMatrixtoVector.
// Inputs:
// - an n x 1 matrix A
// Outputs:
// - a copy of A in a VECTOR structure (also n x 1)
//////////////////////////////////////////////////////////////////////////////
VECTOR castMatrixToVector(MATRIX A);

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
// Main
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
 
int main(int argc, char* argv[])
{
   VECTOR x("signal.vector");
   MATRIX dict("dict.matrix");
   VECTOR s = qrOMP(x, dict, 100, 0.001);
   s.write("resultQR.vector");
   VECTOR mpReconstruct = dict * s;
   mpReconstruct.write("reconstructQR.vector");
   return 0; 
}

//////////////////////////////////////////////////////////////////////////////
// Function implementations
//////////////////////////////////////////////////////////////////////////////

int getMaxIndex(VECTOR x) {
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

MATRIX castVectorToMatrix(VECTOR V) {
  MATRIX A(V.size(), 1);
  for (int i = 0; i < V.size(); i++) {
    A(i, 0) = V[i];
  }
  return A;
}

VECTOR castMatrixToVector(MATRIX A) {
  assert(A.cols() == 1);
  VECTOR V(A.rows());
  for (int i = 0; i < A.rows(); i++) {
    V[i] = A(i, 0);
  }
  return V;
}

VECTOR qrOMP(VECTOR x, MATRIX dict, int natom, double tolerance) {
  VECTOR residual(x);
  double normx2 = x.norm2() * x.norm2();
  double normr2 = normx2;
  double normtol2 = tolerance * normr2; 
  int dictsize = dict.cols();                                        // number of dictionary atoms
  MATRIX R(natom, natom); R.clear();                                 // initializing R and Q
  MATRIX Q(x.size(), natom); Q.clear();
  VECTOR initial_projections = dict.transpose() * x;                 // the set of initial inner products
  VECTOR projections(initial_projections);                           // a copy to be updated
  MATRIX D = dict.transpose() * dict;                                // compute the matrix Gramian of dict                   
  VECTOR s(dictsize); s.clear();                                     // initialize the eventual return vector
  vector<int> gammak;                                                // collect the indices into a C++ vector
  int k = 1;                                                         // start at k = 1 to avoid sizing any vectors or matrices with 0 rows/columns

  while (normr2 > normtol2 && k <= natom) {  
    int maxindex = getMaxIndex(projections);                         // find the maximum index of the squared(!) inner products           
    gammak.push_back(maxindex);                                      // store it in gammak
    if (k == 1) {                                                    // Base Case---set up the Q, R matrices for the first pass
      R(0, 0) = 1.0;                                                 // initialize values in Q and R
      Q.setColumn(dict.getColumn(maxindex), 0);
      double stemp = projections[maxindex];                          // find the inner product corresponding to the maximum index
      projections -= ((D.transpose()).getColumn(maxindex) * stemp);  // update the projections
      normr2 = normx2 - stemp*stemp;                                 // update residual energy
    }
    else {                                                                                              // when k > 1
      VECTOR w = (Q.getSubmatrix(0, Q.rows(), 0, k - 1)).transpose() * dict.getColumn(gammak[k-1]);     // w is the transpose of a submatrix of Q times a corresponding atom
      R(k - 1, k - 1) = sqrt(1.0 - (w.norm2() * w.norm2()));                                            // set the corner entry of R. assumes dictionary normalization!
      for (int i = 0; i < w.size(); i++) {                                                              // set the first k-1 entries of the k-1 column to the entries of w
        R(i, k - 1) = w(i);
      }
      VECTOR columnToSet = dict.getColumn(maxindex) - ((Q.getSubmatrix(0, Q.rows(), 0, k - 1) * w));    // the new direction contributed by the atom
      columnToSet = columnToSet * (1.0 / R(k - 1, k - 1));                                              // scale by the reciprocal of sqrt(1 - ||w||^2)
      Q.setColumn(columnToSet, k - 1);
      double qkTx = Q.getColumn(k - 1) * x;                                                             // dot product used to help update projections
      MATRIX toSubtractMatrix = (castVectorToMatrix(qkTx * Q.getColumn(k - 1))).transpose() * dict;     // returns a 1 x N  matrix
      VECTOR toSubtractVector = castMatrixToVector(toSubtractMatrix.transpose());                       // transposes and casts the matrix to a vector
      projections -= toSubtractVector;                                                                  // update the projections
      normr2 -= (qkTx * qkTx);                                                                          // update the norm
    }
    k++;  
  }

  MATRIX tempR = R.getSubmatrix(0, k - 1, 0, k - 1); 
  VECTOR projectionsGammak(gammak.size());                           // initialize a subvector of the projections  
  for (int i = 0; i < projectionsGammak.size(); i++) {               
    projectionsGammak[i] = initial_projections[gammak[i]];           // copy the relevant inner products into it 
  }
  projectionsGammak.write("projectionsGammak.vector");
  MATRIX tempRtranspose(tempR.transpose());
  tempRtranspose.factorLU();
  tempRtranspose.solveLU(projectionsGammak);                         // solve the linear system tempR' y = projectionsGammak; returns y in to the passed projectionsGammak
  tempR.factorLU();
  tempR.solveLU(projectionsGammak);                                  // solve the linear system tempR x = y, where y is the solution from the previous system; returns x into the passed y
  for (int i = 0; i < gammak.size(); i++) {                          
    s[gammak[i]] = projectionsGammak[i];                             // copy the components of x into the solution s at the appropriate indices
  }
  return s;
}
