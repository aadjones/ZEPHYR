#ifndef COMPRESSION_H
#define COMPRESSION_H

#include <iostream>
#include <fftw3.h>
#include <sys/stat.h>
#include <numeric>

#include "EIGEN.h"
#include "SUBSPACE_FLUID_3D_EIGEN.h"
#include "FLUID_3D_MIC.h"
#include "CUBATURE_GENERATOR_EIGEN.h"
#include "MATRIX.h"
#include "SIMPLE_PARSER.h"
#include "INTEGER_FIELD_3D.h"
#include "COMPRESSION_DATA.h"
#include "DECOMPRESSION_DATA.h"

//////////////////////////////////////////////////////// 
// Function signatures
////////////////////////////////////////////////////////

// cast an array of doubles to VECTOR
VECTOR CastToVector(double* data, int size);

// cast a VECTOR to an array of doubles
double* CastToDouble(const VECTOR& x, double* array);

// cast a VECTOR (preferably with integer entries to avoid lossiness) to an array of ints
int* CastToInt(const VECTOR& x, int* array);

short* CastToShort(const VECTOR& x, short* array);

long* CastToLong(const VECTOR& x, long* array);

// cast a C++ vector of integers to a VECTOR of integers
VECTOR CastIntToVector(const vector<int>& V);

// same as above, but with th einput as a C++ vector of short integers (int16)
VECTOR CastIntToVector(const vector<short>& V);

// more fun casting
vector<int> CastVectorToInt(const VECTOR& V);

// even more!
vector<double> CastVectorToDouble(const VECTOR& V);

VECTOR CastDoubleToVector(const vector<double>& V);

// round a double to the nearest integer
int RoundToInt(const double& x);

INTEGER_FIELD_3D RoundFieldToInt(const FIELD_3D& F);

FIELD_3D CastIntFieldToDouble(const INTEGER_FIELD_3D& F);

// get a vector of cumulative sums, but starting with a zeroth entry of zero and skipping the final entry
vector<int> ModifiedCumSum(vector<int>& V);

VECTOR ModifiedCumSum(const VECTOR& V);

MATRIX ModifiedCumSum(const MATRIX& M);

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
// q is a linear damping parameter and nBits is the desired integer bit depth.
VECTOR3_FIELD_3D BlockCompressVectorField(const VECTOR3_FIELD_3D& V, COMPRESSION_DATA& compression_data);

// The following few functions are helpers for BlockCompressVectorField:

// 3D, 8x8x8 block DCT compression on a scalar field. Both q and nBits are as above. 
FIELD_3D DoBlockCompression(FIELD_3D& F, COMPRESSION_DATA& compression_data);

// Calculates the necessary amount of padding in each dimension to ensure that the dimensions
// of v after padding would then be evenly divisible by 8
void GetPaddings(VEC3I v, int& xPadding, int& yPadding, int& zPadding);

// Divides the passed in scalar field into 8x8x8 subfields (after padding)
// and returns a vector of them in row-major order.
vector<FIELD_3D> GetBlocks(const FIELD_3D& F);

// Converts a C++ vector of scalar field blocks in row-major order back into a scalar field.
FIELD_3D AssimilateBlocks(const FIELD_3D& F, vector<FIELD_3D> V);


// Accepts as input a vector of scalar fields (in practice, 8x8x8 blocks) and performs
// the usual symmetrically-normalized 3d DCT-II on each one.
void DoBlockDCT(vector<FIELD_3D>& V);

// Encodes the passed in scalar field (in practice, an 8x8x8 block) by taking the 3D DCT,
// normalizes the data by dividing by te DC component, converts to an nBit integer (usually 16),
// and damps by dividing by (1 + (u + v + w)*q)^power. Returns an INTEGER_FIELD_3D.
// The passed in sList should start out empty and accumulate a list of the
// appropriate scale factors for each block to then pass to the decoder.
INTEGER_FIELD_3D EncodeBlock(FIELD_3D& F, int blockNumber, COMPRESSION_DATA& compression_data); 

// Works in chain-like fashion with EncodeBlock. 
// The function operates on an INTEGER_FIELD_3D passed in from EncodeBlock and reverses
// the encoding procedure by doing the appropriate scaling and IDCT.
// It returns a scalar field which is a compressed version of the original
// scalar field fed into EncodeBlock.
FIELD_3D DecodeBlock(const INTEGER_FIELD_3D& intBlock, int blockNumber, int col, COMPRESSION_DATA& data, DECOMPRESSION_DATA& decompression_data); 

FIELD_3D DecodeBlockOld(const INTEGER_FIELD_3D& intBlock, int blockNumber, COMPRESSION_DATA& data);

// Writes a binary file using run-length encoding.
// Input is assume to already have been zigzagged.
// Length is generally going to be 8*8*8.
// blockLengths will be passed in as empty and will
// accumulate a list of the encoded block lengths for the decoder.
// Make sure you call DeleteIfExists(filename) first!
void RunLengthEncodeBinary(const char* filename, int* zigzaggedArray, int length, vector<int>& blockLengths);

// Works in a chain with RunLengthEncodeBinary.
// Decodes a run-length encoded, zigzagged vector, returning a vector of integers.
vector<short> RunLengthDecodeBinary(const short* allData, int blockNumber, VECTOR& blockLengths, VECTOR& blockIndices); 

// Deletes the file named 'filename' if it already exists
void DeleteIfExists(const char* filename);

// Reads in an encoded file with its block lengths and stores it in allData. 
// Needs blockLengths in order to know how much memory to allocate.
void ReadBinaryFileToMemory(const char* filename, short*& allData, COMPRESSION_DATA& compression_data, DECOMPRESSION_DATA& decompression_data);

// Chains several previous functions together
void CompressAndWriteField(const char* filename, const FIELD_3D& F, COMPRESSION_DATA& compression_data); 

int ComputeBlockNumber(int row, int col, VEC3I dims, int& blockIndex);

// Given a (row, col) entry of U.final.matrix, decode the corresponding block
FIELD_3D DecodeFromRowCol(int row, int col, short* allDataX, short* allDataY, short* allDataZ, COMPRESSION_DATA& compression_data,
    DECOMPRESSION_DATA& dataX, DECOMPRESSION_DATA& dataY, DECOMPRESSION_DATA& dataZ);

void WriteMetaData(const char* filename, const MATRIX& sListMatrix, const MATRIX& blockLengths, const MATRIX& blockIndices);

void PrefixBinary(string prefix, string filename, string newFile); 

void CleanUpPrefix(const char* prefix, const char* filename); 


void CompressAndWriteMatrixComponent(const char* filename, const MatrixXd& U, int component, COMPRESSION_DATA& data); 

////////////////////////////////////////////////////////
// End Function Signatures
////////////////////////////////////////////////////////



#endif
