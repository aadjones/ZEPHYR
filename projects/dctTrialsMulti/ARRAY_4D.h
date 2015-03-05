#ifndef ARRAY_4D_H
#define ARRAY_4D_H

#include "EIGEN.h"
#include "VECTOR.h"
#include "FIELD_3D.h"

using std::vector;

class ARRAY_4D {
  public:
    ARRAY_4D();
    ARRAY_4D(const int& xRes, const int& yRes, const int& zRes, const int& tRes);
    ARRAY_4D(const double* data, const int& xRes, const int& yRes, const int& zRes, const int& tRes);
    ARRAY_4D(vector<FIELD_3D> fieldList);
    
    // Accessors
    inline Real& operator()(int x, int y, int z, int t) { 
    assert(z >= 0 && z < _zRes && y >= 0 && y < _yRes && x >= 0 && x < _xRes &&
        t >= 0 && t < _tRes);
    return _data[t * _xRes*_yRes*_zRes + z * _xRes*_yRes  + y * _xRes + x]; 
    } 
    const double operator()(int x, int y, int z, int t) const { 
      assert(z >= 0 && z < _zRes && y >= 0 && y < _yRes && x >= 0 && x < _xRes &&
          t >= 0 && t < _tRes);
      return _data[t * _xRes*_yRes*_zRes + z * _xRes*_yRes  + y * _xRes + x]; 
    }
    const double* dataConst() const { return _data; }

    inline double& operator[](int x) { return _data[x]; }
    const double operator[](int x) const { return _data[x]; }
    const int xRes() const { return _xRes; }
    const int yRes() const { return _yRes; }
    const int zRes() const { return _zRes; }
    const int tRes() const { return _tRes; }
    const int totalCells() const { return _totalCells; }
    
    // Return a flattened VECTOR of the array contents
    VECTOR flattened() const;
    VECTOR flattenedRow() const;

    // Explode into vector of FIELD_3Ds
    vector<FIELD_3D> flattenedField() const;

    // Overloaded operators
    
    ARRAY_4D& operator*=(const double& alpha);

    // Get a sub-array
    ARRAY_4D subarray(const int xBegin, const int xEnd, 
                    const int yBegin, const int yEnd, 
                    const int zBegin, const int zEnd,
                    const int tBegin, const int tEnd) const;
    // Swap contents with another 4d-array
    void swapPointers(ARRAY_4D& array);

  private:
    int _xRes;
    int _yRes;
    int _zRes;
    int _tRes;
    int _totalCells;
    double* _data;


};


#endif
