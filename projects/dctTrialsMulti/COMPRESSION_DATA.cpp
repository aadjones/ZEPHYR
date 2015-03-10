#include <iostream>
#include "COMPRESSION_DATA.h"

using std::vector;

COMPRESSION_DATA::COMPRESSION_DATA() {}

COMPRESSION_DATA::COMPRESSION_DATA(VEC3I dims, int numCols, double q, double power, int nBits) :
  _dims(dims), _numCols(numCols), _q(q), _power(power), _nBits(nBits) {}

COMPRESSION_DATA::~COMPRESSION_DATA() {}
