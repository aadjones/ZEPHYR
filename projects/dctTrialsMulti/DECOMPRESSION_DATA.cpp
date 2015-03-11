#include <iostream>
#include "DECOMPRESSION_DATA.h"
#include "MATRIX.h"

DECOMPRESSION_DATA::DECOMPRESSION_DATA() {}


DECOMPRESSION_DATA::DECOMPRESSION_DATA(double q, double power, int nBits, VEC3I dims, int numCols) :

  _q(q), _power(power), _nBits(nBits), _dims(dims), _numCols(numCols)

{
}


DECOMPRESSION_DATA::~DECOMPRESSION_DATA() {}
