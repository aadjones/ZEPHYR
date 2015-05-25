#ifndef COMPRESSION_H
#define COMPRESSION_H

#include <iostream>
#include <fftw3.h>

#include "EIGEN.h"
#include "VECTOR3_FIELD_3D.h"
#include "INTEGER_FIELD_3D.h"
#include "MATRIX_COMPRESSION_DATA.h"

//////////////////////////////////////////////////////// 
// Function signatures
////////////////////////////////////////////////////////


FIELD_3D DoSmartBlockCompression(FIELD_3D& F, COMPRESSION_DATA& data);

void DCT_Smart(FIELD_3D& F, fftw_plan& plan, double*& in); 
void DCT_Smart_Unitary(FIELD_3D& F, fftw_plan& plan, double*& in, int direction); 
void DoSmartBlockDCT(vector<FIELD_3D>& V, int direction);
void DoSmartUnitaryBlockDCT(vector<FIELD_3D>& V, int direction); 

void DecodeBlockSmart();

fftw_plan Create_DCT_Plan(double*& in, int direction); 


FIELD_3D DecodeBlockSmart(const INTEGER_FIELD_3D& intBlock, int blockNumber, COMPRESSION_DATA& data); 

VECTOR3_FIELD_3D SmartBlockCompressVectorField(const VECTOR3_FIELD_3D& V, COMPRESSION_DATA& compression_data);


// cast an array of doubles to VECTOR
VECTOR CastToVector(double* data, int size);

// cast a VECTOR to an array of doubles
double* CastToDouble(const VECTOR& x, double* array);


// cast a VECTOR (preferably with integer entries to avoid lossiness) to an array of ints
int* CastToInt(const VECTOR& x, int* array);

// cast a C++ vector of integers to a VECTOR of integers
VECTOR CastIntToVector(const vector<int>& V);

// same as above, but with the input as a C++ vector of int integers (int16)
VECTOR CastIntToVector(const vector<int>& V);

// more fun casting
vector<int> CastVectorToInt(const VECTOR& V);

// even more!
vector<double> CastVectorToDouble(const VECTOR& V);

VECTOR CastDoubleToVector(const vector<double>& V);

// round a double to the nearest integer
int RoundToInt(const double& x);

INTEGER_FIELD_3D RoundFieldToInt(const FIELD_3D& F);

void CastIntFieldToDouble(const INTEGER_FIELD_3D& F, FIELD_3D& castedField);

// get a vector of cumulative sums, but starting with a zeroth entry of zero and skipping the final entry
vector<int> ModifiedCumSum(vector<int>& V);

VECTOR ModifiedCumSum(const VECTOR& V);

MATRIX ModifiedCumSum(const MATRIX& M);

// extract the three component scalar fields from a vector field
void GetScalarFields(const VECTOR3_FIELD_3D& V, FIELD_3D& X, FIELD_3D& Y, FIELD_3D& Z);

VECTOR ZigzagFlattenSmart(const INTEGER_FIELD_3D& F, const INTEGER_FIELD_3D& zigzagArray);

// reconstruct an INTEGER_FIELD_3D of size 8 x 8 x 8 from a zigzag scan
void ZigzagUnflattenSmart(const VECTOR& V, const INTEGER_FIELD_3D& zigzagArray, INTEGER_FIELD_3D& unflattened);


// in-place version
void DCT_in_place(FIELD_3D& F);


// Calculates the necessary amount of padding in each dimension to ensure that the dimensions
// of v after padding would then be evenly divisible by 8
void GetPaddings(VEC3I v, int& xPadding, int& yPadding, int& zPadding);

// Divides the passed in scalar field into 8x8x8 subfields (after padding)
// and returns a vector of them in row-major order.
vector<FIELD_3D> GetBlocks(const FIELD_3D& F);

// Does the same as above, but pads with zero. Also flattens each block
// into a VectorXd for use in projection/unprojection.
vector<VectorXd> GetBlocksEigen(const FIELD_3D& F); 

// Converts a C++ vector of scalar field blocks in row-major order back into a scalar field.
void AssimilateBlocks(const VEC3I& dims, const vector<FIELD_3D>& V, FIELD_3D& assimilatedField);

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
FIELD_3D DecodeBlock(const INTEGER_FIELD_3D& intBlock, int blockNumber, int col, const DECOMPRESSION_DATA& decompression_data); 


// Writes a binary file using run-length encoding.
// Input is assume to already have been zigzagged.
// Length is generally going to be 8*8*8.
// blockLengths will be passed in as empty and will
// accumulate a list of the encoded block lengths for the decoder.
// Make sure you call DeleteIfExists(filename) first!
void RunLengthEncodeBinary(const char* filename, int* zigzaggedArray, int length, vector<int>& blockLengths);

// Works in a chain with RunLengthEncodeBinary.
// Decodes a run-length encoded, zigzagged vector, returning a vector of integers.
vector<int> RunLengthDecodeBinary(const int* allData, int blockNumber, VECTOR& blockLengths, VECTOR& blockIndices); 

// Deletes the file named 'filename' if it already exists
void DeleteIfExists(const char* filename);

// Reads in an encoded file with its block lengths and stores it in allData. 
// Needs blockLengths in order to know how much memory to allocate.
void ReadBinaryFileToMemory(const char* filename, int*& allData, DECOMPRESSION_DATA& decompression_data);

// Chains several previous functions together
void CompressAndWriteField(const char* filename, const FIELD_3D& F, COMPRESSION_DATA& compression_data); 

int ComputeBlockNumber(int row, VEC3I dims, int& blockIndex);

// Given a (row, col) entry of U.final.matrix, decode the corresponding block and entry
double DecodeFromRowCol(int row, int col, MATRIX_COMPRESSION_DATA& data); 

MatrixXd GetSubmatrix(int startRow, int numRows, MATRIX_COMPRESSION_DATA& data);

VectorXd GetRow(int row, MATRIX_COMPRESSION_DATA& data);
    
void WriteMetaData(const char* filename, const COMPRESSION_DATA& compression_data, const MATRIX& sListMatrix, const MATRIX& blockLengths, const MATRIX& blockIndices);

void PrefixBinary(string prefix, string filename, string newFile); 

void CleanUpPrefix(const char* prefix, const char* filename); 


void CompressAndWriteMatrixComponent(const char* filename, const MatrixXd& U, int component, COMPRESSION_DATA& data); 



FIELD_3D DecodeScalarField(const DECOMPRESSION_DATA& decompression_data, int* const& allData, int col); 

void DecodeScalarFieldWithoutTransformEigenFast(const DECOMPRESSION_DATA& decompression_data, int* const& allData, int col, vector<VectorXd>& toFill); 
vector<VectorXd> DecodeScalarFieldEigen(const DECOMPRESSION_DATA& decompression_data, int* const& allData, int col);

VECTOR3_FIELD_3D DecodeVectorField(MATRIX_COMPRESSION_DATA& data, int col); 


MatrixXd DecodeFullMatrix(MATRIX_COMPRESSION_DATA& data); 


void PeeledCompressedUnproject(VECTOR3_FIELD_3D& V, MATRIX_COMPRESSION_DATA& U_data, const VectorXd& q);   

double GetDotProductSum(vector<VectorXd> Vlist, vector<VectorXd> Wlist);

VectorXd PeeledCompressedProject(VECTOR3_FIELD_3D& V, MATRIX_COMPRESSION_DATA& U_data);

void GetRowFast(int row, int matrixRow, MATRIX_COMPRESSION_DATA& data, MatrixXd& matrixToFill);

void GetSubmatrixFast(int startRow, int numRows, MATRIX_COMPRESSION_DATA& data, MatrixXd& matrixToFill); 

void RunLengthDecodeBinaryFast(const int* allData, int blockNumber, int col, const MATRIX& blockLengthsMatrix, const MATRIX& blockIndicesMatrix, vector<int>& parsedData); 


double DecodeFromRowColFast(int row, int col, MATRIX_COMPRESSION_DATA& data); 

void DecodeScalarFieldFast(const DECOMPRESSION_DATA& decompression_data, int* const& allData, int col);


void DecodeScalarFieldEigenFast(const DECOMPRESSION_DATA& decompression_data, int* const& allData, int col); 

void DecodeVectorFieldFast(MATRIX_COMPRESSION_DATA& data, int col, VECTOR3_FIELD_3D& vecfieldToFill); 

MatrixXd DecodeFullMatrixFast(MATRIX_COMPRESSION_DATA& data);  

void DecodeBlockDecomp(const INTEGER_FIELD_3D& intBlock, int blockNumber, int col, const DECOMPRESSION_DATA& decompression_data, FIELD_3D& fieldToFill);

void DecodeBlockFast(const INTEGER_FIELD_3D& intBlock, int blockNumber, int col, const DECOMPRESSION_DATA& decompression_data, FIELD_3D& fieldToFill); 

void IDCT_Smart_Fast(FIELD_3D& F_hat, const DECOMPRESSION_DATA& decompression_data, FIELD_3D& fieldToFill); 

void CastIntToVectorFast(const vector<int>& V, VECTOR& vecToFill); 

void PeeledCompressedProjectTransformTest1(const VECTOR3_FIELD_3D& V, const MATRIX_COMPRESSION_DATA& U_data,
    VectorXd* q);

void DoSmartUnitaryBlockDCTEigen(vector<VectorXd>& V, int direction);
void DCT_Unitary_Normalize(FIELD_3D* F);
void IDCT_Unitary_Normalize(FIELD_3D* F);
void UndoNormalize(FIELD_3D* F);

////////////////////////////////////////////////////////
// End Function Signatures
////////////////////////////////////////////////////////



#endif

