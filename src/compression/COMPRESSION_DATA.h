#ifndef COMPRESSION_DATA_H
#define COMPRESSION_DATA_H

#include "EIGEN.h"
#include <iostream>
#include <fftw3.h>
#include "FIELD_3D.h"
#include "INTEGER_FIELD_3D.h"

// pushed to SETTINGS.h
//#define BLOCK_SIZE 8
//#define BLOCK_SIZE 16

using std::cout;
using std::endl;


class COMPRESSION_DATA {
  public:
    COMPRESSION_DATA();
    COMPRESSION_DATA(const VEC3I& dims, int numCols, int nBits, double percent);
    ~COMPRESSION_DATA();


    // getters
    const VEC3I& get_dims() const { return _dims; }
    const VEC3I& get_paddedDims() const { return _paddedDims; }
    int get_numCols() const { return _numCols; }
    int get_numBlocks() const { return _numBlocks; }
    int get_currBlockNum() const { return _currBlockNum; }
    double get_percent() const { return _percent; }
    int get_nBits() const { return _nBits; } 
    int get_maxIterations() const { return _maxIterations; } 
    const VectorXi& get_blockLengths() const { return _blockLengths; }
    const VectorXi& get_blockIndices() const { return _blockIndices; }
    
    // adding matrix versions within compression data
    MatrixXi* get_blockLengthsMatrix() { return &(_blockLengthsMatrix); }
    MatrixXi* get_blockIndicesMatrix() { return &(_blockIndicesMatrix); }

    // modified get_sList and gammaList to break const-ness
    VectorXd* get_sList() { return &(_sList); }
    VectorXd* get_gammaList() { return &(_gammaList); }

    MatrixXd* get_sListMatrix() { return &(_sListMatrix); }
    MatrixXd* get_gammaListMatrix() { return &(_gammaListMatrix); }

    vector<Vector3d>* get_singularList() { return &(_singularList); }
    vector<Matrix3d>* get_vList() { return &(_vList); }
    
    const FIELD_3D& get_dampingArray() const { return _dampingArray; }
    const INTEGER_FIELD_3D& get_zigzagArray() const { return _zigzagArray; }
    const INTEGER_FIELD_3D& get_reverseZigzag() const { return _reverseZigzag; }
    double* get_dct_in() const { return _dct_in; }
    double* get_dct_out() const { return _dct_out; }
    fftw_plan get_dct_plan() const { return _dct_plan; }

    // setters
    void set_dims(const VEC3I& dims) { _dims = dims; }
    void set_paddedDims(const VEC3I& paddedDims) { _paddedDims = paddedDims; }
    void set_numCols(int numCols) { _numCols = numCols; }
    void set_numBlocks(int numBlocks) { _numBlocks = numBlocks; }
    void set_currBlockNum(int currBlockNum) { _currBlockNum = currBlockNum; }
    void set_percent(double percent) { _percent = percent; }
    void set_nBits(int nBits) { _nBits = nBits; }
    void set_maxIterations(int maxIterations) { _maxIterations = maxIterations; }

    void set_blockLengths(const VectorXi& blockLengths) { 
      int length = blockLengths.size();
      assert(length == _numBlocks);
      _blockLengths = blockLengths; 
    }

    void set_blockIndices(const VectorXi& blockIndices) { 
      int length = blockIndices.size();
      assert(length == _numBlocks);
      _blockIndices = blockIndices;
    }

    void set_sList(const VectorXd& sList) { 
      int length = sList.size();
      assert(length == _numBlocks);
      _sList = sList;
    
    }
    
    void set_vList(const vector<Matrix3d>& vList) {
      int length = vList.size();
      assert(length == _numCols);
      _vList = vList;
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

    // cache the reverse too
    _reverseZigzag = _zigzagArray;
    for (int x = 0; x < zigzagArray.totalCells(); x++)
      _reverseZigzag[zigzagArray[x]] = x;
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
    VEC3I _paddedDims;
    int _numCols;
    int _numBlocks;
    int _currBlockNum;
    int _maxIterations;
    double _nBits;
    double _percent;

    VectorXi _blockLengths;
    VectorXi _blockIndices;
    VectorXd _sList;
    VectorXd _gammaList;

    MatrixXi _blockLengthsMatrix;
    MatrixXi _blockIndicesMatrix;
    MatrixXd _sListMatrix;
    MatrixXd _gammaListMatrix;

    vector<Matrix3d> _vList;
    vector<Vector3d> _singularList;

    FIELD_3D _dampingArray;
    INTEGER_FIELD_3D _zigzagArray;
    INTEGER_FIELD_3D _reverseZigzag;

    double* _dct_in;
    double* _dct_out;
    fftw_plan _dct_plan;
};

#endif

