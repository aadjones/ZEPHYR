#ifndef COMPRESSION_DATA_H
#define COMPRESSION_DATA_H

#include "EIGEN.h"
#include <iostream>
#include <assert.h>
#include "VECTOR.h"
#include "VEC3.h"
#include "FIELD_3D.h"

using std::cout;
using std::endl;

  class COMPRESSION_DATA {
    public:
      COMPRESSION_DATA();
      COMPRESSION_DATA(VEC3I dims, double q, double power, int nBits);


      // getters
      VEC3I get_dims() const { return _dims; }
      int get_numCols() const { return _numCols; }
      int get_numBlocks() const { return _numBlocks; }
      int get_currBlockNum() const { return _currBlockNum; }
      double get_q() const { return _q; }
      double get_power() const { return _power; }
      int get_nBits() const { return _nBits; } 
      VECTOR get_blockLengths() const { return _blockLengths; }
      VECTOR get_blockIndices() const { return _blockIndices; }
      VECTOR get_sList() const { return _sList; }
      FIELD_3D get_dampingArray() const { return _dampingArray; }

      // setters
      void set_dims(const VEC3I& dims) { _dims = dims; }
      void set_numCols(int numCols) { _numCols = numCols; }
      void set_numBlocks(int numBlocks) { _numBlocks = numBlocks; }
      void set_currBlockNum(int currBlockNum) { _currBlockNum = currBlockNum; }
      void set_q(double q) { _q = q; }
      void set_power(double power) { _power = power; }
      void set_nBits(int nBits) { _nBits = nBits; }

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
        int uRes = 8;
        int vRes = 8;
        int wRes = 8;
        FIELD_3D damp(uRes, vRes, wRes);

        double q = (*this).get_q();
        double power = (*this).get_power();
        
        for (int w = 0; w < wRes; w++) {
          for (int v = 0; v < vRes; v++) {
            for (int u = 0; u < uRes; u++) {
              double r_uvw = 1 + (u + v + w) * q;
              r_uvw = pow(r_uvw, power);
              damp(u, v, w) = r_uvw;
              // for debugging!
              // damp(u, v, w) = 1.0;
            }
          }
        }
        _dampingArray = damp;
      }


    private:
      VEC3I _dims;
      int _numCols;
      int _numBlocks;
      int _currBlockNum;
      double _q;
      double _power;
      double _nBits;
      VECTOR _blockLengths;
      VECTOR _blockIndices;
      VECTOR _sList;
      FIELD_3D _dampingArray;
  };

#endif

