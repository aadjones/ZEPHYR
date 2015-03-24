#include "MATRIX_COMPRESSION_DATA.h"
// #include "COMPRESSION_DATA.h"
// #include "DECOMPRESSION_DATA.h"

MATRIX_COMPRESSION_DATA::MATRIX_COMPRESSION_DATA() 
{
}

MATRIX_COMPRESSION_DATA::MATRIX_COMPRESSION_DATA(int* const& dataX, int* const& dataY, int* const& dataZ,
        const DECOMPRESSION_DATA& decompression_dataX, const DECOMPRESSION_DATA& decompression_dataY, const DECOMPRESSION_DATA& decompression_dataZ) :
  
  _dataX(dataX), _dataY(dataY), _dataZ(dataZ), 
  _decompression_dataX(decompression_dataX), _decompression_dataY(decompression_dataY), _decompression_dataZ(decompression_dataZ) {
  
  }
  

MATRIX_COMPRESSION_DATA::~MATRIX_COMPRESSION_DATA() {
}
 
