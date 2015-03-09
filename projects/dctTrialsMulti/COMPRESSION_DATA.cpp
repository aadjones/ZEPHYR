#include <iostream>
#include "COMPRESSION_DATA.h"

using std::vector;

COMPRESSION_DATA::COMPRESSION_DATA(VEC3I dims, double q, double power, int nBits) :
  _dims(dims), _q(q), _power(power), _nBits(nBits) {}

COMPRESSION_DATA::~COMPRESSION_DATA() {}


