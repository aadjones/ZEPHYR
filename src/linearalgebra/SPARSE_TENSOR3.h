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
// SPARSE_TENSOR3.h: interface for the SPARSE_TENSOR3 class.
//
//////////////////////////////////////////////////////////////////////

#ifndef SPARSE_TENSOR3_H
#define SPARSE_TENSOR3_H

#include "EIGEN.h"

#include <SETTINGS.h>
#include <vector>
#include <map>
#include <iostream>
#include <cstdio>
#include "SPARSE_MATRIX.h"
#include "TENSOR3.h"

using namespace std;

//////////////////////////////////////////////////////////////////////
// A sparse tensor class based on maps
//////////////////////////////////////////////////////////////////////
class SPARSE_TENSOR3 {

public:
  SPARSE_TENSOR3();
  SPARSE_TENSOR3(int rows, int cols, int slabs);

  virtual ~SPARSE_TENSOR3() {};

  // get the reference to an entry -
  // note that if an entry doesn't exist, it will be created and
  // set to zero.
  // 
  // to check if an entry already exists, use the exists() function
  Real& operator()(int row, int col, int slab);
  SPARSE_TENSOR3& operator+=(const SPARSE_TENSOR3& A);

  const int& rows() const { return _rows; };
  const int& cols() const { return _cols; };
  const int& slabs() const { return _slabs; };

  bool exists(const int row, const int col, const int slab) {
    return _data[slab].exists(row, col);
  };

  // take the product w.r.t. mode three, i.e. scale each slab by vector entries, and sum them
  SPARSE_MATRIX modeThreeProduct(const VECTOR& x);
    
  SPARSE_MATRIX& slab(const int i) { return _data[i]; };

  void resize(const int rows, const int cols, const int slabs);
  void clear();
  void buildStatic();

  // file IO
  void write(const string& filename) const;
  bool read(const string& filename);
  void writeGz(const string& filename) const;
  bool readGz(const string& filename);

  // how many entries does it have?
  int size();

  // what is the 2-norm?
  Real sumSq();

  // return a full version of this sparse tensor
  TENSOR3 full();

protected:
  // resize based on the current dimensions
  void resize();

  int _rows;
  int _cols;
  int _slabs;

  vector<SPARSE_MATRIX> _data;
};

ostream& operator<<(ostream &out, SPARSE_TENSOR3& tensor);

#endif
