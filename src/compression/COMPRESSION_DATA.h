#ifndef COMPRESSION_DATA_H
#define COMPRESSION_DATA_H

#include "EIGEN.h"
#include <iostream>
#include <fftw3.h>
#include <assert.h>
#include "VECTOR.h"
#include "VEC3.h"
#include "FIELD_3D.h"
#include "INTEGER_FIELD_3D.h"

using std::cout;
using std::endl;

const int BLOCK_SIZE = 8;

class COMPRESSION_DATA {
  public:
    COMPRESSION_DATA();
    COMPRESSION_DATA(VEC3I dims, int numCols, double q, double power, int nBits);
    ~COMPRESSION_DATA();


    // getters
    const VEC3I& get_dims() const { return _dims; }
    int get_numCols() const { return _numCols; }
    int get_numBlocks() const { return _numBlocks; }
    int get_currBlockNum() const { return _currBlockNum; }
    double get_q() const { return _q; }
    double get_power() const { return _power; }
    double get_percent() const { return _percent; }
    int get_nBits() const { return _nBits; } 
    int get_maxIterations() const { return _maxIterations; } 
    const VECTOR& get_blockLengths() const { return _blockLengths; }
    const VECTOR& get_blockIndices() const { return _blockIndices; }

    // modified get_sList to break const-ness
    VECTOR* get_sList() { return &(_sList); }
    VECTOR* get_gammaList() { return &(_gammaList); }

    const FIELD_3D& get_dampingArray() const { return _dampingArray; }
    const INTEGER_FIELD_3D& get_zigzagArray() const { return _zigzagArray; }
    double* get_dct_in() const { return _dct_in; }
    double* get_dct_out() const { return _dct_out; }
    fftw_plan get_dct_plan() const { return _dct_plan; }

    // setters
    void set_dims(const VEC3I& dims) { _dims = dims; }
    void set_numCols(int numCols) { _numCols = numCols; }
    void set_numBlocks(int numBlocks) { _numBlocks = numBlocks; }
    void set_currBlockNum(int currBlockNum) { _currBlockNum = currBlockNum; }
    void set_q(double q) { _q = q; }
    void set_power(double power) { _power = power; }
    void set_percent(double percent) { _percent = percent; }
    void set_nBits(int nBits) { _nBits = nBits; }
    void set_maxIterations(int maxIterations) { _maxIterations = maxIterations; }

    void set_blockLengths(const VECTOR& blockLengths) { 
      int length = blockLengths.size();
      assert(length == _numBlocks);
      _blockLengths = blockLengths; 
    }

    void set_blockIndices(const VECTOR& blockIndices) { 
      int length = blockIndices.size();
      assert(length == _numBlocks);
      _blockIndices = blockIndices;
    }

    void set_sList(const VECTOR& sList) { 
      int length = sList.size();
      assert(length == _numBlocks);
      _sList = sList;
    
    }

    // compute and set damping array

    void set_dampingArray() {
      int uRes = BLOCK_SIZE;
      int vRes = BLOCK_SIZE;
      int wRes = BLOCK_SIZE;
      FIELD_3D damp(uRes, vRes, wRes);

      for (int w = 0; w < wRes; w++) {
        for (int v = 0; v < vRes; v++) {
          for (int u = 0; u < uRes; u++) {
            damp(u, v, w) = 1 + u + v + w;
          }
        }
      }
      _dampingArray = damp;
    }

    
  void set_zigzagArray() {
    TIMER functionTimer(__FUNCTION__);

    int xRes = BLOCK_SIZE;
    int yRes = BLOCK_SIZE; 
    int zRes = BLOCK_SIZE; 
    INTEGER_FIELD_3D zigzagArray(xRes, yRes, zRes);
    int sum;
    int i = 0;
    for (sum = 0; sum < xRes + yRes + zRes; sum++) {
      for (int z = 0; z < zRes; z++) {
        for (int y = 0; y < yRes; y++) {
          for (int x = 0; x < xRes; x++) {
            if (x + y + z == sum) {
              zigzagArray(x, y, z) = i;
              i++;
            }
          }
        }
      }
    }
    _zigzagArray = zigzagArray;
  }


  void dct_setup(int direction) {
    const int xRes = BLOCK_SIZE;
    const int yRes = BLOCK_SIZE;
    const int zRes = BLOCK_SIZE;

    _dct_in = (double*) fftw_malloc(xRes * yRes * zRes * sizeof(double));
    _dct_out = (double*) fftw_malloc(xRes * yRes * zRes * sizeof(double));

    if (direction == 1) { // forward transform
       _dct_plan = fftw_plan_r2r_3d(zRes, yRes, xRes, _dct_in, _dct_out, 
           FFTW_REDFT10, FFTW_REDFT10, FFTW_REDFT10, FFTW_MEASURE); 
    }

    else { // direction == -1; backward transform
       _dct_plan = fftw_plan_r2r_3d(zRes, yRes, xRes, _dct_in, _dct_out, 
    FFTW_REDFT01, FFTW_REDFT01, FFTW_REDFT01, FFTW_MEASURE);
    }
  }

 void dct_cleanup() {
   fftw_destroy_plan(_dct_plan);
   fftw_free(_dct_in);
   fftw_free(_dct_out);
   fftw_cleanup();
 }

  private:
    VEC3I _dims;
    int _numCols;
    int _numBlocks;
    int _currBlockNum;
    int _maxIterations;
    double _q;
    double _power;
    double _nBits;
    double _percent;
    VECTOR _blockLengths;
    VECTOR _blockIndices;
    VECTOR _sList;
    VECTOR _gammaList;
    FIELD_3D _dampingArray;
    INTEGER_FIELD_3D _zigzagArray;

    double* _dct_in;
    double* _dct_out;
    fftw_plan _dct_plan;
};

#endif

