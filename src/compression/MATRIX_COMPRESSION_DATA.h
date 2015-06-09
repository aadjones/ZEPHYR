#ifndef MATRIX_COMPRESSION_DATA_H
#define MATRIX_COMPRESSION_DATA_H

#include <iostream>
#include <vector>
#include <fftw3.h>
#include "COMPRESSION_DATA.h"
#include "DECOMPRESSION_DATA.h"
#include "FIELD_3D.h"

using std::vector;

class MATRIX_COMPRESSION_DATA {
  public: 
    MATRIX_COMPRESSION_DATA(); 

    MATRIX_COMPRESSION_DATA(int* dataX, int* dataY, int* dataZ,
        const DECOMPRESSION_DATA& decompression_dataX, const DECOMPRESSION_DATA& decompression_dataY, const DECOMPRESSION_DATA& decompression_dataZ);

    MATRIX_COMPRESSION_DATA(int* dataX, int* dataY, int* dataZ,
        COMPRESSION_DATA* compression_dataX, COMPRESSION_DATA* compression_dataY, COMPRESSION_DATA* decompression_dataZ);
    ~MATRIX_COMPRESSION_DATA();

   
    // getters
    
    int* get_dataX() const { return _dataX; }
    int* get_dataY() const { return _dataY; }
    int* get_dataZ() const { return _dataZ; }

    const DECOMPRESSION_DATA& get_decompression_dataX() const { return _decompression_dataX; }
    const DECOMPRESSION_DATA& get_decompression_dataY() const { return _decompression_dataY; }
    const DECOMPRESSION_DATA& get_decompression_dataZ() const { return _decompression_dataZ; }

    COMPRESSION_DATA* get_compression_dataX() { return &_compression_dataX; }
    COMPRESSION_DATA* get_compression_dataY() { return &_compression_dataY; }
    COMPRESSION_DATA* get_compression_dataZ() { return &_compression_dataZ; }

    vector<FIELD_3D>& get_cachedBlocksX() { return _cachedBlocksX; }
    vector<FIELD_3D>& get_cachedBlocksY() { return _cachedBlocksY; }
    vector<FIELD_3D>& get_cachedBlocksZ() { return _cachedBlocksZ; }

    int get_cachedBlockNumber() const { return _cachedBlockNumber; }
    // int get_decodeCounterX() const { return _decodeCounterX; }
    // int get_decodeCounterY() const { return _decodeCounterY; }
    // int get_decodeCounterZ() const { return _decodeCounterZ; }
    
    // setters
    
    void set_dataX(int* dataX) { _dataX = dataX; }
    void set_dataY(int* dataY) { _dataY = dataY; }
    void set_dataZ(int* dataZ) { _dataZ = dataZ; }

    void set_decompression_dataX(const DECOMPRESSION_DATA& decompression_dataX) { _decompression_dataX = decompression_dataX; }
    void set_decompression_dataY(const DECOMPRESSION_DATA& decompression_dataY) { _decompression_dataY = decompression_dataY; }
    void set_decompression_dataZ(const DECOMPRESSION_DATA& decompression_dataZ) { _decompression_dataZ = decompression_dataZ; }

    void set_cachedBlocksX(const vector<FIELD_3D>& cachedBlocksX) { _cachedBlocksX = cachedBlocksX; }
    void set_cachedBlocksY(const vector<FIELD_3D>& cachedBlocksY) { _cachedBlocksY = cachedBlocksY; }
    void set_cachedBlocksZ(const vector<FIELD_3D>& cachedBlocksZ) { _cachedBlocksZ = cachedBlocksZ; }

    void set_cachedBlockNumber(int cachedBlockNumber) { _cachedBlockNumber = cachedBlockNumber; }
    // void set_decodeCounterX(int decodeCounterX) { _decodeCounterX = decodeCounterX; }
    // void set_decodeCounterY(int decodeCounterY) { _decodeCounterY = decodeCounterY; }
    // void set_decodeCounterZ(int decodeCounterZ) { _decodeCounterZ = decodeCounterZ; }
   
    // initializations
    void init_cache() {
      // don't call this until _numCols has been set!

      // set block number to nonsense
      _cachedBlockNumber = -1;
      // initialize decode counter
      // _decodeCounter = 0;

      const int xRes = BLOCK_SIZE;
      const int yRes = BLOCK_SIZE;
      const int zRes = BLOCK_SIZE;

      // clunky, but it works
      int numCols = (*this)._decompression_dataX.get_numCols();

      _cachedBlocksX.resize(numCols);
      for (auto itr = _cachedBlocksX.begin(); itr != _cachedBlocksX.end(); ++itr) {
        (*itr).resizeAndWipe(xRes, yRes, zRes);
      }
     
      _cachedBlocksY.resize(numCols);
      for (auto itr = _cachedBlocksY.begin(); itr != _cachedBlocksY.end(); ++itr) {
        (*itr).resizeAndWipe(xRes, yRes, zRes);
      }
      
      _cachedBlocksZ.resize(numCols);
      for (auto itr = _cachedBlocksZ.begin(); itr != _cachedBlocksZ.end(); ++itr) {
        (*itr).resizeAndWipe(xRes, yRes, zRes);
      }

    }

    // incrementer
    /* 
    void increment_decodeCounter(const char c) {
      if (c == 'X') {
         _decodeCounterX++;
      }
      else if (c == 'Y') {
        _decodeCounterY++;
      }
      else {
        _decodeCounterZ++;
      }
    }
    */

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
    int* _dataX;
    int* _dataY;
    int* _dataZ;

    DECOMPRESSION_DATA _decompression_dataX;
    DECOMPRESSION_DATA _decompression_dataY;
    DECOMPRESSION_DATA _decompression_dataZ;

    COMPRESSION_DATA _compression_dataX;
    COMPRESSION_DATA _compression_dataY;
    COMPRESSION_DATA _compression_dataZ;

    vector<FIELD_3D> _cachedBlocksX;
    vector<FIELD_3D> _cachedBlocksY;
    vector<FIELD_3D> _cachedBlocksZ;

    int _cachedBlockNumber;
    // int _decodeCounterX;
    // int _decodeCounterY;
    // int _decodeCounterZ;

    double* _dct_in;
    double* _dct_out;
    fftw_plan _dct_plan;
};

#endif
