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
  TIMER functionTimer(__FUNCTION__);
  cout << " Building static tensor arrays ... " << flush;
#pragma omp parallel
#pragma omp for  schedule(dynamic)
  for (int x = 0; x < _slabs; x++)
  {
    _data[x].buildStatic();
    cout << x << " " << flush;
  }
  cout << "done. " << endl;
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

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
void SPARSE_TENSOR3::write(const string& filename) const
{
  FILE* file = NULL;
  file = fopen(filename.c_str(), "wb");
  if (file == NULL)
  {
    cout << __FILE__ << " " << __FUNCTION__ << " " << __LINE__ << " : " << endl;
    cout << " Failed to open file " << filename.c_str() << "!!!" << endl;
    return;
  }

  cout << " Writing file " << filename.c_str() << " ..." << flush;  
  fwrite((void*)&_rows, sizeof(int), 1, file);
  fwrite((void*)&_cols, sizeof(int), 1, file);
  fwrite((void*)&_slabs, sizeof(int), 1, file);
 
  for (int x = 0; x < _slabs; x++) 
    _data[x].write(file);

  fclose(file);
  cout << "done." << endl;
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
void SPARSE_TENSOR3::writeGz(const string& filename) const
{
  gzFile file = NULL;
  file = gzopen(filename.c_str(), "wb1");
  if (file == NULL)
  {
    cout << __FILE__ << " " << __FUNCTION__ << " " << __LINE__ << " : " << endl;
    cout << " Failed to open file " << filename.c_str() << "!!!" << endl;
    return;
  }

  cout << " Writing file " << filename.c_str() << " ..." << flush;  
  gzwrite(file, (void*)&_rows, sizeof(int));
  gzwrite(file, (void*)&_cols, sizeof(int));
  gzwrite(file, (void*)&_slabs, sizeof(int));
 
  for (int x = 0; x < _slabs; x++) 
    _data[x].writeGz(file);

  gzclose(file);
  cout << "done." << endl;
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
bool SPARSE_TENSOR3::readGz(const string& filename)
{
  gzFile file;
  file = gzopen(filename.c_str(), "rb1");
  if (file == NULL)
  {
    cout << __FILE__ << " " << __LINE__ << " : File " << filename << " not found! " << endl;
    return false;
  }

  cout << " Reading file " << filename.c_str() << " ..." << flush;

  // read dimensions
  gzread(file, (void*)&_rows, sizeof(int));
  gzread(file, (void*)&_cols, sizeof(int));
  gzread(file, (void*)&_slabs, sizeof(int));

  resize();
  for (int x = 0; x < _slabs; x++) 
    _data[x].readGz(file);

  gzclose(file);
  cout << "done." << endl;

  buildStatic();
  return true;
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
bool SPARSE_TENSOR3::read(const string& filename)
{
  FILE* file;
  file = fopen(filename.c_str(), "rb");
  if (file == NULL)
  {
    cout << __FILE__ << " " << __LINE__ << " : File " << filename << " not found! " << endl;
    return false;
  }

  cout << " Reading file " << filename.c_str() << " ..." << flush;

  // read dimensions
  fread((void*)&_rows, sizeof(int), 1, file);
  fread((void*)&_cols, sizeof(int), 1, file);
  fread((void*)&_slabs, sizeof(int), 1, file);

  resize();
  for (int x = 0; x < _slabs; x++) 
    _data[x].read(file);

  fclose(file);
  cout << "done." << endl;

  buildStatic();
  return true;
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
int SPARSE_TENSOR3::size()
{
  int final = 0;
  for (int x = 0; x < _slabs; x++)
    final += _data[x].size();

  return final;
}

//////////////////////////////////////////////////////////////////////
// take the product w.r.t. mode three, i.e. scale each slab by 
// vector entries, and sum them
//////////////////////////////////////////////////////////////////////
SPARSE_MATRIX SPARSE_TENSOR3::modeThreeProduct(const VECTOR& x)
{
  TIMER functionTimer(__FUNCTION__);
  assert(_slabs == x.size());
  assert(_slabs > 0);

  SPARSE_MATRIX result(_rows, _cols);
  for (int slab = 0; slab < _slabs; slab++)
    result += x[slab] * _data[slab]; 

  return result;
}

//////////////////////////////////////////////////////////////////////
// what is the 2-norm?
//////////////////////////////////////////////////////////////////////
Real SPARSE_TENSOR3::sumSq()
{
  Real final = 0;
  for (int slab = 0; slab < _slabs; slab++)
    final += _data[slab].sumSq();

  return final;
}

//////////////////////////////////////////////////////////////////////
// return a full version of this sparse tensor
//////////////////////////////////////////////////////////////////////
TENSOR3 SPARSE_TENSOR3::full()
{
  TENSOR3 final(_rows, _cols, _slabs);

  for (int x = 0; x < _rows; x++)
    for (int y = 0; y < _cols; y++)
      for (int z = 0; z < _slabs; z++)
        if (exists(x,y,z))
          final(x,y,z) = (*this)(x,y,z);

  return final;
}

//////////////////////////////////////////////////////////////////////
// Print matrix to stream
//////////////////////////////////////////////////////////////////////
ostream& operator<<(ostream &out, SPARSE_TENSOR3& tensor)
{
  for (int x = 0; x < tensor.slabs(); x++)
  {
    // iterate through all the entries
    map<pair<int,int>, Real>::const_iterator i;
    const map<pair<int,int>, Real>& data = tensor.slab(x).matrix();
    for (i = data.begin(); i != data.end(); i++)
    {
      const pair<int,int> index = i->first;
      const Real value = i->second;
      out << "(" << index.first << "," << index.second << "," << x << ") = " <<  value << endl;
    }
  }
  return out;
}
