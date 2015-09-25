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
// SPARSE_MATRIX.h: interface for the SPARSE_MATRIX class.
//
//////////////////////////////////////////////////////////////////////

#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H

#include "EIGEN.h"

#include <SETTINGS.h>
#include <MATRIX.h>
#include <vector>
#include <map>
#include <iostream>
#include <cstdio>

using namespace std;

//////////////////////////////////////////////////////////////////////
// A sparse matrix class based on maps
//////////////////////////////////////////////////////////////////////
class SPARSE_MATRIX {

public:
  SPARSE_MATRIX();
  SPARSE_MATRIX(int rows, int cols);

  virtual ~SPARSE_MATRIX() {};

  // see if an entry exists in the matrix
  bool exists(int row, int col) const;

  // get the reference to an entry -
  // note that if an entry doesn't exist, it will be created and
  // set to zero.
  // 
  // to check if an entry already exists, use the exists() function
  Real& operator()(int row, int col);
  
  // const version of above
  Real constEntry(int row, int col) const;

  const int& rows() const { return _rows; };
  const int& cols() const { return _cols; };
  void resize(int rows, int cols) { _rows = rows; _cols = cols; };
  SPARSE_MATRIX& operator*=(const Real& alpha);
  SPARSE_MATRIX& operator+=(const SPARSE_MATRIX& A);
  SPARSE_MATRIX& operator-=(const SPARSE_MATRIX& A);
  
  // return the dimensions packed into a vector (for output, usually)
  VECTOR dims() const { VECTOR final(2); final[0] = _rows; final[1] = _cols; return final; };

  // Set the matrix to zero. Note this will *NOT* stomp the underlying map!
  // It will instead set all current entries to zero so that we are not
  // forced to reallocate the sparsity structure again
  void clear();

  // set the matrix to zero AND stomp the sparsity pattern
  void clearAndStompSparsity() { _matrix.clear(); };

  // write to a file stream
  void write(const string& filename) const;
  void writeGz(const string& filename) const;
  void write(FILE* file) const;
  void writeGz(gzFile& file) const;

  // read from a file stream
  void readGz(gzFile& file);
  void read(FILE* file);

  // direct access to the matrix
  const map<pair<int,int>, Real>& matrix() const { return _matrix; };

  virtual int size() { return _matrix.size(); };

  // set to the identity matrix
  void setToIdentity();

  // project by the given left and right matrices, where we assume the left one needs
  // to be transposed
  MatrixXd project(const MatrixXd& left, const MatrixXd& right) const;

  // call Matlab to get the eigendecomposition
  //
  // Since a sparse solver is called, "howMany" determines how many eigenvalues
  // are solved for. The results are stored in the specified filenames
  void matlabEigs(const int howMany, const string& matrixFilename, const string& vectorFilename);

  // build arrays for static multiplies
  void buildStatic();

  // do a static multiply
  VECTOR staticMultiply(const VECTOR& v);

  // Get the matrix exponential
  SPARSE_MATRIX exp();

  // get the maximum absolute entry
  Real maxAbsEntry();

  // convert to a full matrix
  MATRIX full() const;

  // what is the 2-norm?
  Real sumSq();

protected:
  int _rows;
  int _cols;

  // a dud Real to pass back if the index is out of bounds
  Real _dud;

  // pair is <row,col>
  map<pair<int,int>, Real> _matrix;

  // static arrays for fast multiply
  vector<int> _in;
  vector<int> _out;
  vector<Real> _values;
};

ostream& operator<<(ostream &out, SPARSE_MATRIX& matrix);
VECTOR operator*(const SPARSE_MATRIX& A, const VECTOR& x);
SPARSE_MATRIX operator*(SPARSE_MATRIX& A, Real& alpha);
MATRIX operator*(const SPARSE_MATRIX& A, const MATRIX& B);
SPARSE_MATRIX operator*(const SPARSE_MATRIX& A, const SPARSE_MATRIX& B);
SPARSE_MATRIX operator*(const Real& alpha, const SPARSE_MATRIX& A);
SPARSE_MATRIX operator-(const SPARSE_MATRIX& A, const SPARSE_MATRIX& B);
SPARSE_MATRIX operator+(const SPARSE_MATRIX& A, const SPARSE_MATRIX& B);
MATRIX operator^(const MATRIX& A, const SPARSE_MATRIX& B);
MATRIX operator*(const MATRIX& A, const SPARSE_MATRIX& B);
VECTOR operator^(const SPARSE_MATRIX& A, const VECTOR& x);

// Eigen support
MatrixXd operator*(const SPARSE_MATRIX& A, const MatrixXd& B);
MatrixXd operator*(const MatrixXd& A, const SPARSE_MATRIX& B);
MatrixXd operator^(const MatrixXd& A, const SPARSE_MATRIX& B);
VectorXd operator*(const SPARSE_MATRIX& A, const VectorXd& x);
#endif
