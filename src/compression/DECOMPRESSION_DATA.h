#ifndef DECOMPRESSION_DATA_H
#define DECOMPRESSION_DATA_H

#include <iostream>
#include <fftw3.h>
#include "MATRIX.h"
#include "VEC3.h"
#include "FIELD_3D.h"
#include "INTEGER_FIELD_3D.h"

// pushed to SETTINGS.h
//#define BLOCK_SIZE 8
//#define BLOCK_SIZE 16

using std::vector;

class DECOMPRESSION_DATA {
  public:
    DECOMPRESSION_DATA();
    DECOMPRESSION_DATA(double q, double power, int nBits, VEC3I dims, int numCols);
    ~DECOMPRESSION_DATA();

    // getters
    double get_q() const { return _q; }
    double get_power() const { return _power; }
    int nBits() const { return _nBits; }
    const VEC3I& get_dims() const { return _dims; }
    int get_numCols() const { return _numCols; }
    int get_numBlocks() const { return _numBlocks; }
    const MatrixXi get_blockLengthsMatrix() const { return _blockLengthsMatrix; } 
    const MatrixXi& get_blockIndicesMatrix() const { return _blockIndicesMatrix; }
    const MatrixXd& get_sListMatrix() const { return _sListMatrix; }
    const MatrixXd& get_gammaListMatrix() const { return _gammaListMatrix; }

    const INTEGER_FIELD_3D& get_zigzagArray() const { return _zigzagArray; }
    const FIELD_3D& get_dampingArray() const { return _dampingArray; }
    double* get_dct_in() const { return _dct_in; }
    double* get_dct_out() const { return _dct_out; }
    fftw_plan get_dct_plan() const { return _dct_plan; }

    // setters
    void set_q(double q) { _q = q; }
    void set_power(double power) { _power = power; }
    void set_nBits(int nBits) { _nBits = nBits; }
    void set_dims(const VEC3I& dims) { _dims = dims; }
    void set_numCols(int numCols) { _numCols = numCols; }
    void set_numBlocks(int numBlocks) { _numBlocks = numBlocks; }
    void set_blockLengthsMatrix(const MatrixXi& blockLengthsMatrix) { _blockLengthsMatrix = blockLengthsMatrix; }
    void set_blockIndicesMatrix(const MatrixXi& blockIndicesMatrix) { _blockIndicesMatrix = blockIndicesMatrix; }
    void set_sListMatrix(const MatrixXd& sListMatrix) { _sListMatrix = sListMatrix; }
    void set_gammaListMatrix(const MatrixXd& gammaListMatrix) { _gammaListMatrix = gammaListMatrix; }

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

    // compute and set zigzag array
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
    double _q;
    double _power;
    int _nBits;
    VEC3I _dims;
    int _numCols;
    int _numBlocks;
    MatrixXi _blockLengthsMatrix;
    MatrixXi _blockIndicesMatrix;
    MatrixXd _sListMatrix;
    MatrixXd _gammaListMatrix;

    FIELD_3D _dampingArray;
    INTEGER_FIELD_3D _zigzagArray;

    double* _dct_in;
    double* _dct_out;
    fftw_plan _dct_plan;

};

#endif

