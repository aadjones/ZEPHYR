#ifndef MATRIX_COMPRESSION_DATA_H
#define MATRIX_COMPRESSION_DATA_H

#include <iostream>
#include "COMPRESSION_DATA.h"
#include "DECOMPRESSION_DATA.h"

class MATRIX_COMPRESSION_DATA {
  public: 
    MATRIX_COMPRESSION_DATA(); 

    MATRIX_COMPRESSION_DATA(const COMPRESSION_DATA& data, short* const& dataX, short* const& dataY, short* const& dataZ,
        const DECOMPRESSION_DATA& decompression_dataX, const DECOMPRESSION_DATA& decompression_dataY, const DECOMPRESSION_DATA& decompression_dataZ);

   
    // getters
    
    COMPRESSION_DATA get_compression_data() const { return _data; }

    short* get_dataX() const { return _dataX; }
    short* get_dataY() const { return _dataY; }
    short* get_dataZ() const { return _dataZ; }

    DECOMPRESSION_DATA get_decompression_dataX() const { return _decompression_dataX; }
    DECOMPRESSION_DATA get_decompression_dataY() const { return _decompression_dataY; }
    DECOMPRESSION_DATA get_decompression_dataZ() const { return _decompression_dataZ; }
    
    // setters
    
    void set_compression_data(const COMPRESSION_DATA& data) { _data = data; }

    void set_dataX(short* const& dataX) { _dataX = dataX; }
    void set_dataY(short* const& dataY) { _dataY = dataY; }
    void set_dataZ(short* const& dataZ) { _dataZ = dataZ; }

    void set_decompression_dataX(const DECOMPRESSION_DATA& decompression_dataX) { _decompression_dataX = decompression_dataX; }
    void set_decompression_dataY(const DECOMPRESSION_DATA& decompression_dataY) { _decompression_dataY = decompression_dataY; }
    void set_decompression_dataZ(const DECOMPRESSION_DATA& decompression_dataZ) { _decompression_dataZ = decompression_dataZ; }

  private:
    COMPRESSION_DATA _data;

    short* _dataX;
    short* _dataY;
    short* _dataZ;

    DECOMPRESSION_DATA _decompression_dataX;
    DECOMPRESSION_DATA _decompression_dataY;
    DECOMPRESSION_DATA _decompression_dataZ;

};

#endif
