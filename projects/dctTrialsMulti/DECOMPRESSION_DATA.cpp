#include <iostream>
#include "DECOMPRESSION_DATA.h"
#include "MATRIX.h"

DECOMPRESSION_DATA::DECOMPRESSION_DATA(int numBlocks, MATRIX blockLengthsMatrix, MATRIX blockIndicesMatrix, MATRIX sListMatrix) :
  _numBlocks(numBlocks), _blockLengthsMatrix(blockLengthsMatrix), _blockIndicesMatrix(blockIndicesMatrix), _sListMatrix(sListMatrix) {}
