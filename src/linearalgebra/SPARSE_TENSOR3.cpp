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

#include <map>
#include "SPARSE_TENSOR3.h"

#include <fstream>

//////////////////////////////////////////////////////////////////////
// Constructor for the sparse matrix
//////////////////////////////////////////////////////////////////////
SPARSE_TENSOR3::SPARSE_TENSOR3(int rows, int cols, int slabs) :
  _rows(rows), _cols(cols), _slabs(slabs)
{
  resize();
}

SPARSE_TENSOR3::SPARSE_TENSOR3() :
  _rows(0), _cols(0), _slabs(0)
{
}

//////////////////////////////////////////////////////////////////////
// return a reference to an entry
//////////////////////////////////////////////////////////////////////
Real& SPARSE_TENSOR3::operator()(int row, int col, int slab)
{
  // bounds check
  assert(col >= 0);
  assert(row >= 0); 
  assert(slab >= 0); 
  assert(col < _cols);
  assert(row < _rows);
  assert(slab < _slabs);

  return _data[slab](row, col);
}

//////////////////////////////////////////////////////////////////////
// resize based on the current dimensions
//////////////////////////////////////////////////////////////////////
void SPARSE_TENSOR3::resize()
{
  _data.clear();

  for (int x = 0; x < _slabs; x++)
    _data.push_back(SPARSE_MATRIX(_rows, _cols));
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
void SPARSE_TENSOR3::resize(const int rows, const int cols, const int slabs)
{
  _rows = rows;
  _cols = cols;
  _slabs = slabs;

  resize();
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
void SPARSE_TENSOR3::clear()
{
  for (int x = 0; x < _slabs; x++)
    _data[x].clear();
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
void SPARSE_TENSOR3::buildStatic()
{
  for (int x = 0; x < _slabs; x++)
    _data[x].buildStatic();
}

//////////////////////////////////////////////////////////////////////
// add two sparse matrices together
//////////////////////////////////////////////////////////////////////
SPARSE_TENSOR3& SPARSE_TENSOR3::operator+=(const SPARSE_TENSOR3& A)
{
  assert(A.slabs() == _slabs);

  for (int x = 0; x < _slabs; x++)
    _data[x] += A._data[x];

  return *this;
} 
