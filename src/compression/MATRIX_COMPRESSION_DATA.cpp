#include "MATRIX_COMPRESSION_DATA.h"

MATRIX_COMPRESSION_DATA::MATRIX_COMPRESSION_DATA() 
{
}

MATRIX_COMPRESSION_DATA::MATRIX_COMPRESSION_DATA(int* dataX, int* dataY, int* dataZ,
        const DECOMPRESSION_DATA& decompression_dataX, const DECOMPRESSION_DATA& decompression_dataY, const DECOMPRESSION_DATA& decompression_dataZ) :
  
  _dataX(dataX), _dataY(dataY), _dataZ(dataZ), 
  _decompression_dataX(decompression_dataX), _decompression_dataY(decompression_dataY), _decompression_dataZ(decompression_dataZ)
{  
}
  
MATRIX_COMPRESSION_DATA::MATRIX_COMPRESSION_DATA(int* dataX, int* dataY, int* dataZ,
        COMPRESSION_DATA* compression_dataX, COMPRESSION_DATA* compression_dataY, COMPRESSION_DATA* compression_dataZ) :

  _dataX(dataX), _dataY(dataY), _dataZ(dataZ),
  _compression_dataX(*compression_dataX), _compression_dataY(*compression_dataY), _compression_dataZ(*compression_dataZ)
{
}

MATRIX_COMPRESSION_DATA::~MATRIX_COMPRESSION_DATA()
{
}
 
