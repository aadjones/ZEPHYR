#ifndef COMPRESSION_REWRITE_H
#define COMPRESSION_REWRITE_H

#include <iostream>
#include <fftw3.h>

#include "EIGEN.h"
#include "VECTOR3_FIELD_3D.h"
#include "INTEGER_FIELD_3D.h"
#include "MATRIX_COMPRESSION_DATA.h"

//////////////////////////////////////////////////////// 
// Function signatures
////////////////////////////////////////////////////////

// round a FIELD_3D to an INTEGER_FIELD_3D
void RoundFieldToInt(const FIELD_3D& F, INTEGER_FIELD_3D* castedField); 

// cast an INTEGER_FIELD_3D to a FIELD_3D
void CastIntFieldToDouble(const INTEGER_FIELD_3D& F, FIELD_3D* castedField); 

// form the cumulative sum starting at zero of a passed in integer vector
void ModifiedCumSum(const VectorXi& V, VectorXi* sum);

// extract the three scalar fields from a vector field
void GetScalarFields(const VECTOR3_FIELD_3D& V, FIELD_3D* X, FIELD_3D* Y, FIELD_3D* Z); 

// make an fftw 3d dct plan. direction 1 is forward, -1 is inverse
void Create_DCT_Plan(double* in, int direction, fftw_plan* plan); 

// perform a unitary normalization for the forward 3d dct
void DCT_Unitary_Normalize(double* buffer);

// perform a unitary normalization in preparation for the inverse 3d dct
void UndoNormalize(FIELD_3D* F);

// do the corresponding dct based on the plan and direction
void DCT_Smart_Unitary(const fftw_plan& plan, int direction, double* in, FIELD_3D* F); 

// given passed in dimensions, compute the amount of padding required to get to
// dimensions that are evenly divisible by BLOCK_SIZE
void GetPaddings(const VEC3I& v, VEC3I* paddings);  

// given passed in field, parse it into a vector of padded 3d blocks of size BLOCK_SIZE
// in row-major order
void GetBlocks(const FIELD_3D& F, vector<FIELD_3D>* blocks); 

// reassemble a big field from a row-major list of smaller blocks
void AssimilateBlocks(const VEC3I& dims, const vector<FIELD_3D>& V, FIELD_3D* assimilatedField); 

// perform a unitary dct on each block of a passed in list. direction 1 is dct,
// -1 is idct
void UnitaryBlockDCT(int direction, vector<FIELD_3D>* blocks); 

// build a block diagonal matrix with A's as the kernel for
// a 'count' number of times. inefficient usage of memory
// since it fails to use sparse matrix
void BlockDiagonal(const MatrixXd& A, int count, MatrixXd* B);

// build a sparse block diagonal matrix with A's as the kernel
// for a 'count' number of times
void SparseBlockDiagonal(const MatrixXd& A, int count, SparseMatrix<double>* B);

// given a passed in vec3 field, build a matrix with each column
// corresponding to the x-, y-, and z-components 
void BuildXYZMatrix(const VECTOR3_FIELD_3D& V, MatrixXd* A);

// build a transformed vector field using a coordinate transform computed from the
// svd decomposition of the original x-, y-, and z- coordinates
// uses v^T, not v!
void TransformVectorFieldSVD(VectorXd* s, MatrixXd* v, VECTOR3_FIELD_3D* transformedV);

// build a transformed vector field using a coordinate transform computed from
// svd decomposition of the original x-, y-, and z- coordinates. update the
// compression data to store the v matrix and the singular values.
// can be called repeatedly in chain-like fashion.
void TransformVectorFieldSVDCompression(VECTOR3_FIELD_3D* V, COMPRESSION_DATA* data);

// undo the effects of a previous svd coordinate transform on a vector field
void UntransformVectorFieldSVD(const MatrixXd& v, VECTOR3_FIELD_3D* transformedV);

// normalize the block to a resolution of nBits based on the DC component.
// update the sList.
void PreprocessBlock(FIELD_3D* F, int blockNumber, int col, COMPRESSION_DATA* data);

// Binary search to find the appropriate gamma given
// desired percent threshold within maxIterations. Prints
// out information as it goes.
///////////////////////////////////////////////////////
void TuneGammaVerbose(const FIELD_3D& F, int blockNumber, int col, 
    COMPRESSION_DATA* data, FIELD_3D* damp);

// do a binary search to find the appropriate gamma given the desired percent 
// energy accuracy and max iterations. the variable damp will be rewritten to the
// desired damping array. updates gamaList.
void TuneGamma(const FIELD_3D& F, int blockNumber, int col, COMPRESSION_DATA* data, 
    FIELD_3D* damp);

// takes a passed in FIELD_3D (which is intended to be
// the result of a DCT post-preprocess). calculates the best gamma value
// for a damping array. then damps by that array and quantizes the result to an integer. 
// stores the value of gamma for the damping.
void EncodeBlock(const FIELD_3D& F, int blockNumber, int col, COMPRESSION_DATA* data, 
    INTEGER_FIELD_3D* quantized); 

// takes a passed in INTEGER_FIELD_3D (which is intended to be run-length
// decoded and unzigzagged) at a particular blockNumber and column of the matrix.
// undoes the effects of damping/quantization as best as it can.
void DecodeBlock(const INTEGER_FIELD_3D& intBlock, int blockNumber, int col, 
    const DECOMPRESSION_DATA& decompression_data, FIELD_3D* decoded);

// performs the same operations as DecodeBlock, but with passed in compression data
// rather than passed in decompression data. due to const poisoning, compression
// data cannot be marked const, but is treated as such.
void DecodeBlockWithCompressionData(const INTEGER_FIELD_3D& intBlock, 
  int blockNumber, COMPRESSION_DATA& data, FIELD_3D* decoded); 

// flattens an INTEGER_FIELD_3D through a zig-zag scan
// into a VectorXi. Since the scan always follows the same order,
// we precompute the zigzag scan array, pass it
// as a parameter, and then just do an index lookup
void ZigzagFlatten(const INTEGER_FIELD_3D& F, const INTEGER_FIELD_3D& zigzagArray, 
    VectorXi* zigzagged);

// unflattens a VectorXi into an INTEGER_FIELD_3D 
void ZigzagUnflatten(const VectorXi& V, const INTEGER_FIELD_3D& zigzagArray, 
    INTEGER_FIELD_3D* unflattened);

// given a zigzagged integer vector, write it to a binary
// file via run-length encoding. assumes the blockLengths
// vector has already been allocated!
void RunLengthEncodeBinary(const char* filename, int blockNumber, const VectorXi& zigzaggedArray, 
    VectorXi* blockLengths);

// decode a run-length encoded binary file and fill 
// a VectorXi with the contents.
void RunLengthDecodeBinary(int* allData, int blockNumber, int col, 
    const MatrixXi& blockLengthsMatrix, const MatrixXi& blockIndicesMatrix, VectorXi* parsedData);

// takes an input FIELD_3D which is the result of
// an SVD coordinate transform, compresses it according
// to the general scheme, and writes it to a binary file 
void CompressAndWriteField(const char* filename, const FIELD_3D& F, int col, 
    COMPRESSION_DATA* compression_data);

// print four different percents for how far along each column we are
void PrintProgress(int col, int numCols);

// generate the header information for the encoded binary file
void WriteMetaData(const char* filename, const COMPRESSION_DATA& compression_data);

// delete a binary file if it already exists
void DeleteIfExists(const char* filename);
// compress all of the scalar field components
// of a matrix (which represents a vector field) and write them to
// a binary file. applies svd coordinate transform first 
void CompressAndWriteMatrixComponents(const char* filename, const MatrixXd& U,  
      COMPRESSION_DATA* data0, COMPRESSION_DATA* data1, COMPRESSION_DATA* data2);

#endif


