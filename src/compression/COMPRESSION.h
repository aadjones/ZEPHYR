#ifndef COMPRESSION_H
#define COMPRESSION_H

#include <iostream>
#include <fftw3.h>

#include "EIGEN.h"
#include "SUBSPACE_FLUID_3D_EIGEN.h"
#include "FLUID_3D_MIC.h"
#include "CUBATURE_GENERATOR_EIGEN.h"
#include "MATRIX.h"
#include "SIMPLE_PARSER.h"
#include "INTEGER_FIELD_3D.h"
#include "COMPRESSION_DATA.h"
#include "DECOMPRESSION_DATA.h"
#include "COMPRESSION_UTILITIES.h"

//////////////////////////////////////////////////////// 
// Function signatures
////////////////////////////////////////////////////////

// extract the three component scalar fields from a vector field
void GetScalarFields(const VECTOR3_FIELD_3D& V, FIELD_3D& X, FIELD_3D& Y, FIELD_3D& Z);

// flatten a FIELD_3D through a zigzag scan
VECTOR ZigzagFlatten(const INTEGER_FIELD_3D& F);

// reconstruct an INTEGER_FIELD_3D of size 8 x 8 x 8 from a zigzag scan
INTEGER_FIELD_3D ZigzagUnflatten(const VECTOR& V);

// out-of-place, symmetrically normalized, 3d DCT-II of a scalar field
FIELD_3D DCT(FIELD_3D& F);

// in-place version
void DCT_in_place(FIELD_3D& F);

// out-of-place, symmetrically normalized, 3d IDCT (aka DCT_III) of a scalar field
FIELD_3D IDCT(FIELD_3D& F_hat);

// 3D, 8x8x8 block DCT compression on the component scalar fields of a vector field.
// For use with buildDistortedMatrix 
VECTOR3_FIELD_3D BlockCompressVectorField(const VECTOR3_FIELD_3D& V, COMPRESSION_DATA& compression_data);

// The following few functions are helpers for BlockCompressVectorField:

// 3D, 8x8x8 block DCT compression on a scalar field.
FIELD_3D DoBlockCompression(FIELD_3D& F, COMPRESSION_DATA& compression_data);

// Calculates the necessary amount of padding in each dimension to ensure that the dimensions
// of v after padding would then be evenly divisible by 8.
void GetPaddings(VEC3I v, int& xPadding, int& yPadding, int& zPadding);

// Divides the passed in scalar field into 8x8x8 subfields (after padding)
// and returns a vector of them in row-major order.
vector<FIELD_3D> GetBlocks(const FIELD_3D& F);

// Converts a C++ vector of scalar field blocks in row-major order back into a scalar field.
FIELD_3D AssimilateBlocks(const VEC3I& dims, vector<FIELD_3D> V);

// Accepts as input a vector of scalar fields (in practice, 8x8x8 blocks) and performs
// the usual symmetrically-normalized 3d DCT-II on each one.
void DoBlockDCT(vector<FIELD_3D>& V);

// Encodes the passed in scalar field (in practice, an 8x8x8 block) by taking the 3D DCT,
// normalizes the data by dividing by the DC component, converts to an nBit integer (usually 16),
// and damps by dividing by (1 + (u + v + w)*q)^power. Returns an INTEGER_FIELD_3D.
// The passed in compression data is modified to keep track of block lengths, indices, and s values.
INTEGER_FIELD_3D EncodeBlock(FIELD_3D& F, int blockNumber, COMPRESSION_DATA& compression_data); 

// Works in chain-like fashion with EncodeBlock. 
// The function operates on an INTEGER_FIELD_3D passed in from EncodeBlock and reverses
// the encoding procedure by doing the appropriate scaling and IDCT.
// It returns a scalar field which is a compressed version of the original
// scalar field fed into EncodeBlock.
// Neither compression nor decompression data are modified.
FIELD_3D DecodeBlock(const INTEGER_FIELD_3D& intBlock, int blockNumber, int col, COMPRESSION_DATA& data, DECOMPRESSION_DATA& decompression_data); 

// The older implementation, for use with generating an uncompressed but DCT/quantized matrix
FIELD_3D DecodeBlockOld(const INTEGER_FIELD_3D& intBlock, int blockNumber, COMPRESSION_DATA& data);

// Writes a binary file using run-length encoding.
// Block size is assumed to be 8 * 8 * 8.
// Input is assume to already have been zigzagged.
// blockLengths will  accumulate a list of the encoded block lengths for the decoder.
// Make sure you call DeleteIfExists(filename) first since it is opened in append mode.
void RunLengthEncodeBinary(const char* filename, int blockNumber, int* zigzaggedArray, VECTOR& blockLengths); 

// Works in a chain with RunLengthEncodeBinary.
// Decodes a run-length encoded, zigzagged vector, returning a vector of integers.
// Assumes you've read the binary file into memory inside allData.
vector<short> RunLengthDecodeBinary(const short* allData, int blockNumber, const VECTOR& blockLengths, const VECTOR& blockIndices); 

// Reads in an encoded file with its block lengths and stores it in allData. 
// Doesn't modify compression data but does update decompression data.
void ReadBinaryFileToMemory(const char* filename, short*& allData, COMPRESSION_DATA& compression_data, DECOMPRESSION_DATA& decompression_data);

// Chains several previous functions together
void CompressAndWriteField(const char* filename, const FIELD_3D& F, COMPRESSION_DATA& compression_data); 

// Given a (row, col) entry of the original matrix, this returns what block to look in. 
// It also computes by reference the actual index within that block to access.
int ComputeBlockNumber(int row, int col, VEC3I dims, int& blockIndex);

// Given a (row, col) entry of U.final.matrix, decode the corresponding block.
// None of the data is modified.
FIELD_3D DecodeFromRowCol(int row, int col, short* allDataX, short* allDataY, short* allDataZ, COMPRESSION_DATA& compression_data,
    DECOMPRESSION_DATA& dataX, DECOMPRESSION_DATA& dataY, DECOMPRESSION_DATA& dataZ);

// Write the header information (s values, block lengths, and block indices).
void WriteMetaData(const char* filename, const MATRIX& sListMatrix, const MATRIX& blockLengths, const MATRIX& blockIndices);

// Perform the compression on one of the components (X, Y, or Z), writing out a binary file.
void CompressAndWriteMatrixComponent(const char* filename, const MatrixXd& U, int component, COMPRESSION_DATA& data); 

////////////////////////////////////////////////////////
// End Function Signatures
////////////////////////////////////////////////////////



#endif
