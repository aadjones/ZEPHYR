#include "COMPRESSION_DATA.h"

COMPRESSION_DATA::COMPRESSION_DATA() 
{
  _arrayListBuilt = false;
}

COMPRESSION_DATA::COMPRESSION_DATA(const VEC3I& dims, int numCols, int nBits, double percent) :
  _dims(dims), _numCols(numCols), _nBits(nBits), _percent(percent) 
{
  _arrayListBuilt = false;
}


COMPRESSION_DATA::~COMPRESSION_DATA() {}
