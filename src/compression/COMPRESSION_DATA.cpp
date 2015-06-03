#include "COMPRESSION_DATA.h"

COMPRESSION_DATA::COMPRESSION_DATA() {}

COMPRESSION_DATA::COMPRESSION_DATA(const VEC3I& dims, int numCols, int nBits, double percent) :
  _dims(dims), _numCols(numCols), _nBits(nBits), _percent(percent) {}


COMPRESSION_DATA::~COMPRESSION_DATA() {}
