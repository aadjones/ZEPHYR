#include "INTEGER_FIELD_3D.h"
#include "VECTOR.h"
#include "FIELD_3D.h"
#include <iostream>
#include <cmath>

using std::cout;
using std::endl;
using std::vector;

INTEGER_FIELD_3D::INTEGER_FIELD_3D() : 
  _xRes(-1), _yRes(-1), _zRes(-1), _data(NULL)
{
  cout << "called defaut int3d constructor" << endl;
}


INTEGER_FIELD_3D::INTEGER_FIELD_3D(const int& xRes, const int& yRes, const int& zRes) :
  _xRes(xRes), _yRes(yRes), _zRes(zRes)
{
  _totalCells = _xRes * _yRes * _zRes;
 
  try {
    _data = new int[_totalCells];
    cout << "called int3d constructor with new" << endl;
  }
  catch(std::bad_alloc& exc)
  {
    cout << " Failed to allocate " << _xRes << " " << _yRes << " " << _zRes << " INTEGER_FIELD_3D!" << endl;
    int bytes = _totalCells * sizeof(int);
    cout << (double) bytes / pow(2.0, 20.0) << " MB needed" << endl;
    exit(0);
  }

  for (int x = 0; x < _totalCells; x++)
    _data[x] = 0;
}

INTEGER_FIELD_3D::INTEGER_FIELD_3D(const int* data, const int& xRes, const int& yRes, const int& zRes) :
  _xRes(xRes), _yRes(yRes), _zRes(zRes)
{
  _totalCells = _xRes * _yRes * _zRes;
  try {
    _data = new int[_totalCells];
    cout << "called int3d constructor with new" << endl;
  }
  catch(std::bad_alloc& exc)
  {
    cout << " Failed to allocate " << _xRes << " " << _yRes << " " << _zRes << " INTEGER_FIELD_3D!" << endl;
    int bytes = _totalCells * sizeof(int);
    cout << (double) bytes / pow(2.0, 20.0) << " MB needed" << endl;
    exit(0);
  }

  for (int x = 0; x < _totalCells; x++) {
    _data[x] = data[x];
  }
}
 
INTEGER_FIELD_3D::~INTEGER_FIELD_3D() {
  if (_data) {
    delete[] _data;
    cout << "called int 3d destructor and called delete" << endl;
  }
}

// Return a flattened VECTOR
VECTOR INTEGER_FIELD_3D::flattened() const {
  VECTOR final(_totalCells);
  int index = 0;
  for (int z = 0; z < _zRes; z++) {
    for (int y = 0; y < _yRes; y++) {
      for (int x = 0; x < _xRes; x++, index++) {
        final[index] = (*this)(x, y, z);
      }
    }
  }
  return final;
}

// Return a flattened VECTOR in row order
VECTOR INTEGER_FIELD_3D::flattenedRow() const {
  VECTOR final(_totalCells);
  int index = 0;
  for (int x = 0; x < _xRes; x++) {
    for (int y = 0; y < _yRes; y++) {
      for (int z = 0; z < _zRes; z++, index++) {
          final[index] = (*this)(x, y, z);
      }
    }
  }
  return final;
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////


void INTEGER_FIELD_3D::clear()
{
  for (int x = 0; x < _totalCells; x++)
    _data[x] = 0;
}

void INTEGER_FIELD_3D::resizeAndWipe(int xRes, int yRes, int zRes)
{
  if (_xRes == xRes && _yRes == yRes && _zRes == zRes)
  {
    clear();
    cout << "called int3d resize and wipe without delete" << endl;
    return;
  }

  if (_data) delete[] _data;
  cout << "called int3d resize and wipe with delete" << endl;

  _xRes = xRes;
  _yRes = yRes;
  _zRes = zRes;
  _totalCells = _xRes * _yRes * _zRes;

  try {
    _data = new int[_totalCells];
    cout << "called new within int3d resize and wipe" << endl;
  }
  catch(std::bad_alloc& exc)
  {
    cout << " Failed to allocate " << _xRes << " " << _yRes << " " << _zRes << " INTEGER_FIELD_3D!" << endl;
    double bytes = _xRes * _yRes * _zRes * sizeof(int);
    cout <<  bytes / pow(2.0,20.0) << " MB needed" << endl;
    exit(0);
  }

  for (int x = 0; x < _totalCells; x++)
    _data[x] = 0;
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
// Overloaded operators
INTEGER_FIELD_3D& INTEGER_FIELD_3D::operator*=(const double& alpha)
{
  for (int x = 0; x < _totalCells; x++)
    _data[x] *= alpha;

  return *this;
}

// Get a subfield
INTEGER_FIELD_3D INTEGER_FIELD_3D::subfield(const int xBegin, const int xEnd, 
                            const int yBegin, const int yEnd,
                            const int zBegin, const int zEnd) const {
  assert(xBegin >= 0);
  assert(yBegin >= 0);
  assert(zBegin >= 0);
  assert(xEnd <= _xRes);
  assert(yEnd <= _yRes);
  assert(zEnd <= _zRes);
  assert(xBegin < xEnd);
  assert(yBegin < yEnd);
  assert(zBegin < zEnd);

  int xInterval = xEnd - xBegin;
  int yInterval = yEnd - yBegin;
  int zInterval = zEnd - zBegin;

  INTEGER_FIELD_3D final(xInterval, yInterval, zInterval);

  /* cout << "size: " << final.xRes() << ", " << final.yRes() << ", " << 
     final.zRes() << endl; */
  for (int z = 0; z < zInterval; z++) {
    for (int y = 0; y < yInterval; y++) {
      for (int x = 0; x < xInterval; x++) {
        final(x, y, z) = (*this)(xBegin + x, yBegin + y, zBegin + z);
      }
    }
  }
  return final;
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
INTEGER_FIELD_3D& INTEGER_FIELD_3D::operator=(const INTEGER_FIELD_3D& A)
{
  resizeAndWipe(A.xRes(), A.yRes(), A.zRes());

  for (int x = 0; x < _totalCells; x++)
    _data[x] = A[x];

  cout << "called int field 3d operator=" << endl;
  return *this;
}

// Swap contents with annother 3d integer field
void INTEGER_FIELD_3D::swapPointers(INTEGER_FIELD_3D& field)
{
  assert(field.xRes() == _xRes);
  assert(field.yRes() == _yRes);
  assert(field.zRes() == _zRes);
  
  int* temp = _data;
  _data = field._data;
  field._data = temp;
}

