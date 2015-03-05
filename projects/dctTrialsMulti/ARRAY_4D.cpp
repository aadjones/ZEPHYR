#include "ARRAY_4D.h"
#include "VECTOR.h"
#include "FIELD_3D.h"
#include <iostream>
#include <cmath>

using std::cout;
using std::endl;
using std::vector;

ARRAY_4D::ARRAY_4D(const int& xRes, const int& yRes, const int& zRes, const int& tRes) :
  _xRes(xRes), _yRes(yRes), _zRes(zRes), _tRes(tRes)
{
  _totalCells = _xRes * _yRes * _zRes *_tRes;
 
  try {
    _data = new double[_totalCells];
  }
  catch(std::bad_alloc& exc)
  {
    cout << " Failed to allocate " << _xRes << " " << _yRes << " " << _zRes << _tRes << " ARRAY_4D!" << endl;
    int bytes = _totalCells * sizeof(double);
    cout << (double) bytes / pow(2.0, 20.0) << " MB needed" << endl;
    exit(0);
  }

  for (int x = 0; x < _totalCells; x++)
    _data[x] = 0;
}

ARRAY_4D::ARRAY_4D(const double* data, const int& xRes, const int& yRes, const int& zRes, const int& tRes) :
  _xRes(xRes), _yRes(yRes), _zRes(zRes), _tRes(tRes)
{
  _totalCells = _xRes * _yRes * _zRes * _tRes;
  try {
    _data = new double[_totalCells];
  }
  catch(std::bad_alloc& exc)
  {
    cout << " Failed to allocate " << _xRes << " " << _yRes << " " << _zRes << " "  << 
      _tRes << " ARRAY_4D!" << endl;
    int bytes = _totalCells * sizeof(double);
    cout << (double) bytes / pow(2.0, 20.0) << " MB needed" << endl;
    exit(0);
  }

  for (int x = 0; x < _totalCells; x++) {
    _data[x] = data[x];
  }
}

ARRAY_4D::ARRAY_4D(vector<FIELD_3D> fieldList) :
  _xRes(fieldList[0].xRes()), _yRes(fieldList[0].yRes()), _zRes(fieldList[0].zRes()), _tRes(fieldList.size())
{
  _totalCells = _xRes * _yRes * _zRes * _tRes;
   try {
    _data = new double[_totalCells];
  }
  catch(std::bad_alloc& exc)
  {
    cout << " Failed to allocate " << _xRes << " " << _yRes << " " << _zRes << " "  << 
      _tRes << " ARRAY_4D!" << endl;
    int bytes = _totalCells * sizeof(double);
    cout << (double) bytes / pow(2.0, 20.0) << " MB needed" << endl;
    exit(0);
  }
  
  int index = 0;
  for (int t = 0; t < _tRes; t++) {
    for (int z = 0; z < _zRes; z++) {
      for (int y = 0; y < _yRes; y++) {
        for (int x = 0; x < _xRes; x++, index++) {
          _data[index] = fieldList[t](x, y, z);
        }
      }
    }
  }
}



// Return a flattened VECTOR
VECTOR ARRAY_4D::flattened() const {
  VECTOR final(_totalCells);

  int index = 0;
  for (int t = 0; t < _tRes; t++) {
    for (int z = 0; z < _zRes; z++) {
      for (int y = 0; y < _yRes; y++) {
        for (int x = 0; x < _xRes; x++, index++) {
          final[index] = (*this)(x, y, z, t);
        }
      }
    }
  }
  return final;
}

// Return a flattened VECTOR in row order
VECTOR ARRAY_4D::flattenedRow() const {
  VECTOR final(_totalCells);

  int index = 0;
  for (int x = 0; x < _xRes; x++) {
    for (int y = 0; y < _yRes; y++) {
      for (int z = 0; z < _zRes; z++) {
        for (int t = 0; t < _xRes; t++, index++) {
          final[index] = (*this)(x, y, z, t);
        }
      }
    }
  }
  return final;
}

// Explode into a vector of FIELD_3Ds
vector<FIELD_3D> ARRAY_4D::flattenedField() const {
  vector<FIELD_3D> final(_tRes);
  FIELD_3D fieldSlice(_xRes, _yRes, _zRes);
  for (int t = 0; t < _tRes; t++) {
    for (int z = 0; z < _zRes; z++) {
      for (int y = 0; y < _yRes; y++) {
        for (int x = 0; x < _xRes; x++) {
          fieldSlice(x, y, z) = (*this)(x, y, z, t);
        }
      }
    }
    final[t] = (fieldSlice);
  }
  return final;
}

// Overloaded operators
ARRAY_4D& ARRAY_4D::operator*=(const double& alpha)
{
  for (int x = 0; x < _totalCells; x++)
    _data[x] *= alpha;

  return *this;
}

// Get a subarray
ARRAY_4D ARRAY_4D::subarray(const int xBegin, const int xEnd, 
                            const int yBegin, const int yEnd,
                            const int zBegin, const int zEnd,
                            const int tBegin, const int tEnd) const {
  assert(xBegin >= 0);
  assert(yBegin >= 0);
  assert(zBegin >= 0);
  assert(tBegin >=0);
  assert(xEnd <= _xRes);
  assert(yEnd <= _yRes);
  assert(zEnd <= _zRes);
  assert(tEnd <= _tRes);
  assert(xBegin < xEnd);
  assert(yBegin < yEnd);
  assert(zBegin < zEnd);
  assert(tBegin < tEnd);

  int xInterval = xEnd - xBegin;
  int yInterval = yEnd - yBegin;
  int zInterval = zEnd - zBegin;
  int tInterval = tEnd - tBegin;

  ARRAY_4D final(xInterval, yInterval, zInterval, tInterval);

  /* cout << "size: " << final.xRes() << ", " << final.yRes() << ", " << 
     final.zRes() << ", " << final.tRes() << endl; */
  for (int t = 0; t < tInterval; t++) {
    for (int z = 0; z < zInterval; z++) {
      for (int y = 0; y < yInterval; y++) {
        for (int x = 0; x < xInterval; x++) {
          final(x, y, z, t) = (*this)(xBegin + x, yBegin + y, zBegin + z, tBegin + t);
        }
      }
    }
  }
  return final;
}

// Swap contents with annother 4d-array
void ARRAY_4D::swapPointers(ARRAY_4D& array)
{
  assert(array.xRes() == _xRes);
  assert(array.yRes() == _yRes);
  assert(array.zRes() == _zRes);
  assert(array.tRes() == _tRes);
  
  double* temp = _data;
  _data = array._data;
  array._data = temp;
}

