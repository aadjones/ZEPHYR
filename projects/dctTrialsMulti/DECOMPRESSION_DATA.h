
#ifndef DECOMPRESSION_DATA_H
#define DECOMPRESSION_DATA_H

#include <iostream>
#include "MATRIX.h"

using std::vector;

class DECOMPRESSION_DATA {
  public:
    DECOMPRESSION_DATA() {}
    DECOMPRESSION_DATA(int numBlocks, MATRIX blockLengthsMatrix, MATRIX blockIndicesMatrix, MATRIX sListMatrix);

    // getters
    int get_numBlocks() const { return _numBlocks; }
    MATRIX get_blockLengthsMatrix() const { return _blockLengthsMatrix; } 
    MATRIX get_blockIndicesMatrix() const { return _blockIndicesMatrix; }
    MATRIX get_sListMatrix() const { return _sListMatrix; }

    // setters
    void set_numBlocks(int numBlocks) { _numBlocks = numBlocks; }
    void set_blockLengthsMatrix(const MATRIX& blockLengthsMatrix) { _blockLengthsMatrix = blockLengthsMatrix; }
    void set_blockIndicesMatrix(const MATRIX& blockIndicesMatrix) { _blockIndicesMatrix = blockIndicesMatrix; }
    void set_sListMatrix(const MATRIX& sListMatrix) { _sListMatrix = sListMatrix; }

  private:
    int _numBlocks;
    MATRIX _blockLengthsMatrix;
    MATRIX _blockIndicesMatrix;
    MATRIX _sListMatrix;
};

#endif

