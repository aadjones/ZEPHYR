#include "MATRIX_COMPRESSION_DATA.h"
#include "COMPRESSION_DATA.h"
#include "DECOMPRESSION_DATA.h"

MATRIX_COMPRESSION_DATA::MATRIX_COMPRESSION_DATA(const COMPRESSION_DATA& data, short* const& dataX, short* const& dataY, short* const& dataZ,
        const DECOMPRESSION_DATA& decompression_dataX, const DECOMPRESSION_DATA& decompression_dataY, const DECOMPRESSION_DATA& decompression_dataZ) :
  
  _data(data), _dataX(dataX), _dataY(dataY), _dataZ(dataZ), 
  _decompression_dataX(decompression_dataX), _decompression_dataY(decompression_dataY), _decompression_dataZ(decompression_dataZ) {}
  

