#include <iostream>
#include <fftw3.h>
#include <sys/stat.h>  // for using stat to check whether a file exists
#include <numeric>     // for std::accumulate

#include "EIGEN.h"
#include "SUBSPACE_FLUID_3D_EIGEN.h"
#include "FLUID_3D_MIC.h"
#include "CUBATURE_GENERATOR_EIGEN.h"
#include "MATRIX.h"
#include "SIMPLE_PARSER.h"
#include "COMPRESSION.h"
#include "INTEGER_FIELD_3D.h"
#include "COMPRESSION_DATA.h"
#include "DECOMPRESSION_DATA.h"
#include "MATRIX_COMPRESSION_DATA.h"
#include "FIELD_3D.h"

using std::vector;
using std::accumulate;
using std::cout;
using std::endl;

////////////////////////////////////////////////////////
// Function Implementations
////////////////////////////////////////////////////////

VECTOR CastToVector(double* data, int size) {
  TIMER functionTimer(__FUNCTION__);
  VECTOR x(size);
  for (int i = 0; i < size; i++) {
    x[i] = data[i];
  }
  return x;
}

VECTOR CastToVector(int* data, int size) {
  TIMER functionTimer(__FUNCTION__);
  VECTOR x(size);
  for (int i = 0; i < size; i++) {
    x[i] = data[i];
  }
  return x;
}

VECTOR CastToVector(short* data, int size) {
  TIMER functionTimer(__FUNCTION__);
  VECTOR x(size);
  for (int i = 0; i < size; i++) {
    x[i] = data[i];
  }
  return x;
}

double* CastToDouble(const VECTOR& x, double* array) {
  TIMER functionTimer(__FUNCTION__);
  for (int i = 0; i < x.size(); i++) {
    array[i] = x[i];
  }
  return array;
}

int* CastToInt(const VECTOR& x, int* array) {
  TIMER functionTimer(__FUNCTION__);
  for (int i = 0; i < x.size(); i++) {
    int data_i = (int) x[i];
    array[i] = data_i;
  }
  return array;
}

short* CastToShort(const VECTOR& x, short* array) {
  TIMER functionTimer(__FUNCTION__);
  for (int i = 0; i < x.size(); i++) {
    short data_i = (short) x[i];
    array[i] = data_i;
  }
  return array;
}

long* CastToLong(const VECTOR& x, long* array) {
  TIMER functionTimer(__FUNCTION__);
  for (int i = 0; i < x.size(); i++) {
    long data_i = (long) x[i];
    array[i] = data_i;
  }
  return array;
}

VECTOR CastIntToVector(const vector<int>& V) {
  TIMER functionTimer(__FUNCTION__);
  int length = V.size();
  VECTOR result(length);

  for (int i = 0; i < length; i++) {
    result[i] = V[i];
  }
  return result;
}

vector<int> CastVectorToInt(const VECTOR& V) {
  TIMER functionTimer(__FUNCTION__);
  int length = V.size();
  vector<int> result(length, 0);
  for (int i = 0; i < length; i++) {
    result[i] = V[i];
  }
  return result;
}

vector<double> CastVectorToDouble(const VECTOR& V) {
  TIMER functionTimer(__FUNCTION__);
  int length = V.size();
  vector<double> result(length, 0.0);
  for (int i = 0; i < length; i++) {
    result[i] = V[i];
  } 
  return result;
}

VECTOR CastIntToVector(const vector<short>& V) {
  TIMER functionTimer(__FUNCTION__);
  int length = V.size();
  VECTOR result(length);

  for (int i = 0; i < length; i++) {
    result[i] = V[i];
  }
  return result;
}

VECTOR CastDoubleToVector(const vector<double>& V) {
  TIMER functionTimer(__FUNCTION__);
  int length = V.size();
  VECTOR result(length);

  for (int i = 0; i < length; i++) {
    result[i] = V[i];
  }
  return result;
}

/*
int RoundToInt(const double& x) {
  TIMER functionTimer(__FUNCTION__);
  int result = 0;
  if (x > 0.0) {
    result = (int) (x + 0.5);
  }
  else {
    result = (int) (x - 0.5);
  }
  return result;
}
*/

INTEGER_FIELD_3D RoundFieldToInt(const FIELD_3D& F) {
  TIMER functionTimer(__FUNCTION__);
  int xRes = F.xRes();
  int yRes = F.yRes();
  int zRes = F.zRes();

  INTEGER_FIELD_3D result(xRes, yRes, zRes);

  for (int x = 0; x < xRes; x++) {
    for (int y = 0; y < yRes; y++) {
      for (int z = 0; z < zRes; z++) {
        int rounded = rint(F(x, y, z)); 
        result(x, y, z) = rounded;
      }
    }
  }
  return result;
}

FIELD_3D CastIntFieldToDouble(const INTEGER_FIELD_3D& F) {
  TIMER functionTimer(__FUNCTION__);
  int xRes = F.xRes();
  int yRes = F.yRes();
  int zRes = F.zRes();

  FIELD_3D result(xRes, yRes, zRes);

  for (int x = 0; x < xRes; x++) {
    for (int y = 0; y < yRes; y++) {
      for (int z = 0; z < zRes; z++) {
        double casted = (double) F(x, y, z);
        result(x, y, z) = casted;
      }
    }
  }
  return result;
}

vector<int> ModifiedCumSum(vector<int>& V) {
  // athough V is passed by reference, it will not be modified.
  TIMER functionTimer(__FUNCTION__);
  int length = V.size();
  vector<int> result(length + 1);
  result[0] = 0;
  for (int i = 0; i < length; i++) {
    result[i + 1 ] = V[i];
  }

  for (auto itr = result.begin() + 1; itr != result.end(); ++itr) {
    *(itr) += *(itr - 1);
  }
  result.pop_back();
  return result;
}

VECTOR ModifiedCumSum(const VECTOR& V) {
  TIMER functionTimer(__FUNCTION__);
  vector<int> V_int = CastVectorToInt(V);
  vector<int> result_int = ModifiedCumSum(V_int);
  VECTOR result = CastIntToVector(result_int);
  return result;
}

MATRIX ModifiedCumSum(const MATRIX& M) {
  TIMER functionTimer(__FUNCTION__);
  int numRows = M.rows();
  int numCols = M.cols();
  VECTOR flattened = M.flattenedColumn();
  vector<int> flattenedInt = CastVectorToInt(flattened);
  vector<int> modifiedSum = ModifiedCumSum(flattenedInt);
  assert(modifiedSum.size() == numRows * numCols);
  VECTOR modifiedSumVector = CastIntToVector(modifiedSum);
  MATRIX result(modifiedSumVector, numRows, numCols);
  return result;
}

void GetScalarFields(const VECTOR3_FIELD_3D& V, FIELD_3D& X, FIELD_3D& Y, FIELD_3D& Z) {
  TIMER functionTimer(__FUNCTION__);
  X = V.scalarField(0);
  Y = V.scalarField(1);
  Z = V.scalarField(2);
}

/*
VECTOR ZigzagFlatten(const INTEGER_FIELD_3D& F) {
  TIMER functionTimer(__FUNCTION__);
  int xRes = F.xRes();
  int yRes = F.yRes();
  int zRes = F.zRes();
  VECTOR result(xRes * yRes * zRes);
  int sum;
  int i = 0;
  for (sum = 0; sum < xRes + yRes + zRes; sum++) {
    for (int z = 0; z < zRes; z++) {
      for (int y = 0; y < yRes; y++) {
        for (int x = 0; x < xRes; x++) {
          if (x + y + z == sum) {
            result[i] = F(x, y, z);
            i++;
          }
        }
      }
    }
  }
  return result;

}
*/

VECTOR ZigzagFlattenSmart(const INTEGER_FIELD_3D& F, const INTEGER_FIELD_3D& zigzagArray) {
  TIMER functionTimer(__FUNCTION__);
  int xRes = F.xRes();
  int yRes = F.yRes();
  int zRes = F.zRes();
  assert(xRes == 8 && yRes == 8 && zRes == 8);

  VECTOR result(xRes * yRes * zRes);
  for (int z = 0; z < zRes; z++) {
    for (int y = 0; y < yRes; y++) {
      for (int x = 0; x < xRes; x++) {
        int index = zigzagArray(x, y, z);
        double data = F(x, y, z);
        result[index] = data;
      }
    }
  }
  return result;
}
/*
INTEGER_FIELD_3D ZigzagUnflatten(const VECTOR& V) {
  TIMER functionTimer(__FUNCTION__);
  // assumes original dimensions were 8 x 8 x 8
  const int xRes = 8;
  const int yRes = 8;
  const int zRes = 8;
  INTEGER_FIELD_3D result(xRes, yRes, zRes);
  int sum;
  int i = 0;
  for (sum = 0; sum < xRes + yRes + zRes; sum++) {
    for (int z = 0; z < zRes; z++) {
      for (int y = 0; y < yRes; y++) {
        for (int x = 0; x < xRes; x++) {
          if (x + y + z == sum) {
            result(x, y, z) = V[i];
            i++;
          }
        }
      }
    }
  }
  return result;
}
*/
INTEGER_FIELD_3D ZigzagUnflattenSmart(const VECTOR& V, const INTEGER_FIELD_3D& zigzagArray) {
  TIMER functionTimer(__FUNCTION__);
  // assumes original dimensions were 8 x 8 x 8
  const int xRes = 8;
  const int yRes = 8;
  const int zRes = 8;
  INTEGER_FIELD_3D result(xRes, yRes, zRes);
  for (int z = 0; z < zRes; z++) {
    for (int y = 0; y < yRes; y++) {
      for (int x = 0; x < xRes; x++) {
        int index = zigzagArray(x, y, z);
        result(x, y, z) = V[index];
      }
    }
  }
  return result;
}
FIELD_3D DCT(FIELD_3D& F) {
  TIMER functionTimer(__FUNCTION__);
  double* in;
  double* out;
  fftw_plan plan;
  const int xRes = F.xRes();
  const int yRes = F.yRes();
  const int zRes = F.zRes();
  const int N = xRes * yRes * zRes;
  in = (double*) fftw_malloc(sizeof(double) * N);
  out = (double*) fftw_malloc(sizeof(double) * N);

  VECTOR vector_in =  F.flattened();
  // VECTOR vector_in = F.flattenedRow();
  in = CastToDouble(vector_in, in);
  // data is in column order, so we pass the dimensions in reverse
  plan = fftw_plan_r2r_3d(zRes, yRes, xRes, in, out, FFTW_REDFT10, FFTW_REDFT10, FFTW_REDFT10, FFTW_ESTIMATE);        // real-to-real 3d un-normalized forward transform (DCT-II)
  fftw_execute(plan);
  FIELD_3D F_hat(out, xRes, yRes, zRes);  

  // Normalize---after going forward and backward, fftw scales by 2d in each dimension d.
  // Hence, in the 3d transform, the input will be scaled by 8*xRes*yRes*zRes.
  // To normalize symmetrically, we must correspondingly scale  by sqrt(1/(8*xRes*yRes*zRes)).
  F_hat *= sqrt(0.125 / (xRes * yRes * zRes));

  fftw_destroy_plan(plan);
  fftw_free(in);
  fftw_free(out);

  return F_hat;
}

fftw_plan Create_DCT_Plan(double*& in, int direction) {
  TIMER functionTimer(__FUNCTION__);
  // direction is 1 for a forward transform, -1 for a backward transform
  assert( direction == 1 || direction == -1 );

  int xRes = 8;
  int yRes = 8;
  int zRes = 8;

  fftw_plan plan;
  fftw_r2r_kind kind;
  if (direction == 1) {
    kind = FFTW_REDFT10;
  }
  else {
    kind = FFTW_REDFT01;
  }
  plan = fftw_plan_r2r_3d(zRes, yRes, xRes, in, in, kind, kind, kind, FFTW_MEASURE);
  
  return plan;
} 


void DCT_Smart(FIELD_3D& F, fftw_plan& plan, double*& in) {
  TIMER functionTimer(__FUNCTION__);

  int xRes = F.xRes();
  int yRes = F.yRes();
  int zRes = F.zRes();

  VECTOR vector_in = F.flattened();
  // fill the 'in' buffer
  in = CastToDouble(vector_in, in);

  fftw_execute(plan);
  // 'in' is now overwritten to the result of the transform
  FIELD_3D F_hat(in, xRes, yRes, zRes);
 
  // Normalize symmetrically
  F_hat *= sqrt(0.125 / (xRes * yRes * zRes));

  // rewrite F
  F = F_hat;
}

void DCT_in_place(FIELD_3D& F) {
  TIMER functionTimer(__FUNCTION__);
  double* in;
  double* out;
  fftw_plan plan;
  const int xRes = F.xRes();
  const int yRes = F.yRes();
  const int zRes = F.zRes();
  const int N = xRes * yRes * zRes;
  in = (double*) fftw_malloc(sizeof(double) * N);
  out = (double*) fftw_malloc(sizeof(double) * N);

  VECTOR vector_in =  F.flattened();
  in = CastToDouble(vector_in, in);
  // data is in column order, so we pass the dimensions in reverse
  plan = fftw_plan_r2r_3d(zRes, yRes, xRes, in, out, 
      FFTW_REDFT10, FFTW_REDFT10, FFTW_REDFT10, FFTW_ESTIMATE);        
  fftw_execute(plan);
  FIELD_3D F_hat(out, xRes, yRes, zRes);  

  // normalize symmetrically
  F_hat *= sqrt(0.125 / (xRes * yRes * zRes));

  fftw_destroy_plan(plan);
  fftw_free(in);
  fftw_free(out);

  F.swapPointers(F_hat);
}


FIELD_3D IDCT(FIELD_3D& F_hat) {
  TIMER functionTimer(__FUNCTION__);
  double* in;
  double* out;
  fftw_plan plan;
  const int xRes = F_hat.xRes();
  const int yRes = F_hat.yRes();
  const int zRes = F_hat.zRes();
  const int N = xRes * yRes * zRes;
  in = (double*) fftw_malloc(sizeof(double) * N);
  out = (double*) fftw_malloc(sizeof(double) * N);

  VECTOR vector_in =  F_hat.flattened();
  in = CastToDouble(vector_in, in);
  // real-to-real 3d un-normalized forward transform (dct-ii).
  // pass the dimensions in reverse due to column order.
  plan = fftw_plan_r2r_3d(zRes, yRes, xRes, in, out, 
      FFTW_REDFT01, FFTW_REDFT01, FFTW_REDFT01, FFTW_ESTIMATE); 
  fftw_execute(plan);
  FIELD_3D F(out, xRes, yRes, zRes); 

  // normalize symmetrically
  F *= sqrt(0.125 / (xRes * yRes * zRes)); 

  fftw_destroy_plan(plan);
  fftw_free(in);
  fftw_free(out);
  fftw_cleanup();

  return F;
}

VECTOR3_FIELD_3D SmartBlockCompressVectorField(const VECTOR3_FIELD_3D& V, COMPRESSION_DATA& compression_data) { 
  TIMER functionTimer(__FUNCTION__);

  const int xRes = V.xRes();
  const int yRes = V.yRes();
  const int zRes = V.zRes();

  double* X_array = (double*) malloc(sizeof(double) * xRes * yRes * zRes);
  double* Y_array = (double*) malloc(sizeof(double) * xRes * yRes * zRes);
  double* Z_array = (double*) malloc(sizeof(double) * xRes * yRes * zRes);

  for (int component = 0; component < 3; component++) {            
    cout << "Component " << component << endl;
    FIELD_3D scalarComponent = V.scalarField(component);
    FIELD_3D scalarComponentCompressed = DoSmartBlockCompression(scalarComponent, compression_data);
    VECTOR scalarComponentCompressedFlattened = scalarComponentCompressed.flattened();

    if (component == 0)      X_array = CastToDouble(scalarComponentCompressedFlattened, X_array);
    else if (component == 1) Y_array = CastToDouble(scalarComponentCompressedFlattened, Y_array);
    else                     Z_array = CastToDouble(scalarComponentCompressedFlattened, Z_array);
  }
  
  VECTOR3_FIELD_3D compressed_V(X_array, Y_array, Z_array, xRes, yRes, zRes);

  free(X_array);
  free(Y_array);
  free(Z_array);

  return compressed_V;
}
VECTOR3_FIELD_3D BlockCompressVectorField(const VECTOR3_FIELD_3D& V, COMPRESSION_DATA& compression_data) { 
  TIMER functionTimer(__FUNCTION__);

  const int xRes = V.xRes();
  const int yRes = V.yRes();
  const int zRes = V.zRes();

  double* X_array = (double*) malloc(sizeof(double) * xRes * yRes * zRes);
  double* Y_array = (double*) malloc(sizeof(double) * xRes * yRes * zRes);
  double* Z_array = (double*) malloc(sizeof(double) * xRes * yRes * zRes);

  for (int component = 0; component < 3; component++) {            
    FIELD_3D scalarComponent = V.scalarField(component);
    FIELD_3D scalarComponentCompressed = DoBlockCompression(scalarComponent, compression_data);
    VECTOR scalarComponentCompressedFlattened = scalarComponentCompressed.flattened();

    if (component == 0)      X_array = CastToDouble(scalarComponentCompressedFlattened, X_array);
    else if (component == 1) Y_array = CastToDouble(scalarComponentCompressedFlattened, Y_array);
    else                     Z_array = CastToDouble(scalarComponentCompressedFlattened, Z_array);
  }
  
  VECTOR3_FIELD_3D compressed_V(X_array, Y_array, Z_array, xRes, yRes, zRes);

  free(X_array);
  free(Y_array);
  free(Z_array);

  return compressed_V;
}

FIELD_3D DoSmartBlockCompression(FIELD_3D& F, COMPRESSION_DATA& compression_data) {
  TIMER functionTimer(__FUNCTION__);

  int xRes = F.xRes();
  int xResOriginal = xRes;
  int yRes = F.yRes();
  int yResOriginal = yRes;
  int zRes = F.zRes();
  int zResOriginal = zRes;
  VEC3I dims(xRes, yRes, zRes);
  
  double q = compression_data.get_q();
  double power = compression_data.get_power();
  int nBits = compression_data.get_nBits();
  int numBlocks = compression_data.get_numBlocks();

  // dummy initializations                                     
  int xPadding = 0;
  int yPadding = 0;
  int zPadding = 0;
  
  // fill in the paddings
  GetPaddings(dims, xPadding, yPadding, zPadding);
  // update to the padded resolutions
  xRes += xPadding;
  yRes += yPadding;
  zRes += zPadding;

  // use this dummy to pass the dims into AssimilatedBlocks later on
  FIELD_3D dummyF(xRes, yRes, zRes);                           

  vector<FIELD_3D> blocks = GetBlocks(F);     
  // 1-->forward transform
  cout << "Doing block forward transform..." << endl;
  DoSmartBlockDCT(blocks, 1);
  cout << "...done!" << endl;
  int blockNumber = 0;
  int percent = 0;
  cout << "Doing quantization on each block..." << endl;
  for (auto itr = blocks.begin(); itr != blocks.end(); ++itr) {

    // sList will be updated on each pass and modify compression_data
    INTEGER_FIELD_3D V = EncodeBlock(*itr, blockNumber, compression_data); 
    FIELD_3D compressedBlock = DecodeBlockSmart(V, blockNumber, compression_data); 
    *itr = compressedBlock;
    blockNumber++;


    percent = blockNumber / ( (double) numBlocks );
    int checkPoint1 = (numBlocks - 1) / 4;
    int checkPoint2 = (numBlocks - 1) / 2;
    int checkPoint3 = (3 * (numBlocks - 1)) / 4;
    int checkPoint4 = numBlocks - 1;

    if (blockNumber == checkPoint1 || blockNumber == checkPoint2 || blockNumber == checkPoint3 || blockNumber == checkPoint4) {
      cout << "      Percent complete: " << percent << flush;
      if (blockNumber == checkPoint4) {
        cout << endl;
      }

    }
  }

  cout << "...done!" << endl;
  cout << "Doing block inverse transform..." << endl;
  DoSmartBlockDCT(blocks, -1);
  cout << "...done!" << endl;
  FIELD_3D F_compressed = AssimilateBlocks(dummyF, blocks);
  // strip off the padding
  FIELD_3D F_compressed_peeled = F_compressed.subfield(0, xResOriginal, 0, yResOriginal, 0, zResOriginal); 

  return F_compressed_peeled;
}

FIELD_3D DoBlockCompression(FIELD_3D& F, COMPRESSION_DATA& compression_data) {
  TIMER functionTimer(__FUNCTION__);

  int xRes = F.xRes();
  int xResOriginal = xRes;
  int yRes = F.yRes();
  int yResOriginal = yRes;
  int zRes = F.zRes();
  int zResOriginal = zRes;
  VEC3I dims(xRes, yRes, zRes);
  
  double q = compression_data.get_q();
  double power = compression_data.get_power();
  int nBits = compression_data.get_nBits();

  // dummy initializations                                     
  int xPadding = 0;
  int yPadding = 0;
  int zPadding = 0;
  
  // fill in the paddings
  GetPaddings(dims, xPadding, yPadding, zPadding);
  // update to the padded resolutions
  xRes += xPadding;
  yRes += yPadding;
  zRes += zPadding;

  // use this dummy to pass the dims into AssimilatedBlocks later on
  FIELD_3D dummyF(xRes, yRes, zRes);                           

  vector<FIELD_3D> blocks = GetBlocks(F);     
  DoBlockDCT(blocks);

  int blockNumber = 0;
  for (auto itr = blocks.begin(); itr != blocks.end(); ++itr) {

    // sList will be updated on each pass and modify compression_data
    INTEGER_FIELD_3D V = EncodeBlock(*itr, blockNumber, compression_data); 
    FIELD_3D compressedBlock = DecodeBlockOld(V, blockNumber, compression_data); 
    *itr = compressedBlock;
    blockNumber++;
  }
  FIELD_3D F_compressed = AssimilateBlocks(dummyF, blocks);
  // strip off the padding
  FIELD_3D F_compressed_peeled = F_compressed.subfield(0, xResOriginal, 0, yResOriginal, 0, zResOriginal); 

  return F_compressed_peeled;
}


void GetPaddings(VEC3I v, int& xPadding, int& yPadding, int& zPadding) {
  TIMER functionTimer(__FUNCTION__);
  int xRes = v[0];
  int yRes = v[1];
  int zRes = v[2];
  xPadding = (8 - (xRes % 8)) % 8;     // how far are you from the next multiple of 8?
  yPadding = (8 - (yRes % 8)) % 8;
  zPadding = (8 - (zRes % 8)) % 8;
  return;
}

vector<FIELD_3D> GetBlocks(const FIELD_3D& F) {
  TIMER functionTimer(__FUNCTION__);
  int xRes = F.xRes();
  int yRes = F.yRes();
  int zRes = F.zRes();
  VEC3I v(xRes, yRes, zRes);

  int xPadding;
  int yPadding;
  int zPadding;
  // fill these in with the appropriate paddings
  GetPaddings(v, xPadding, yPadding, zPadding);

  FIELD_3D F_padded_x = F.pad_x(xPadding);
  FIELD_3D F_padded_xy = F_padded_x.pad_y(yPadding);
  FIELD_3D F_padded = F_padded_xy.pad_z(zPadding);
  
  // update the resolutions to the padded ones 
  xRes = F_padded.xRes();
  yRes = F_padded.yRes();
  zRes = F_padded.zRes();

  // sanity check that our padder had the desired effect
  assert(xRes % 8 == 0);
  assert(yRes % 8 == 0);
  assert(zRes % 8 == 0);

  // variable initialization before the loop
  FIELD_3D subfield(8, 8, 8);
  vector<FIELD_3D> blockList;
  
  for (int z = 0; z < zRes/8; z++) {
    for (int y = 0; y < yRes/8; y++) {
      for (int x = 0; x < xRes/8; x++) {
        subfield = F_padded.subfield(8*x, 8*(x+1), 8*y, 8*(y+1), 8*z, 8*(z+1));
        blockList.push_back(subfield);
      }
    }
  }
  return blockList;
}


FIELD_3D AssimilateBlocks(const FIELD_3D& F, vector<FIELD_3D> V) {
  TIMER functionTimer(__FUNCTION__);
  const int xRes = F.xRes();
  const int yRes = F.yRes();
  const int zRes = F.zRes();

  FIELD_3D assimilatedField(xRes, yRes, zRes);

  for (int z = 0; z < zRes; z++) {
    for (int y = 0; y < yRes; y++) {
      for (int x = 0; x < xRes; x++) {
        int index = (x/8) + (y/8) * (xRes/8) + (z/8) * (xRes/8) * (yRes/8);     // warning, evil integer division happening!
        assimilatedField(x, y, z) = V[index](x % 8, y % 8, z % 8);             
      }
    }
  }

  return assimilatedField;
}


void DoSmartBlockDCT(vector<FIELD_3D>& V, int direction) {
  TIMER functionTimer(__FUNCTION__);
  // direction determines whether it is DCT or IDCT

  // initialize block buffer
  // cout << "Calling fftw_malloc." << endl;
  // double* in = (double*) fftw_malloc(8 * 8 * 8 * sizeof(double));

  // double it just to be safe
  double* in = (double*) malloc(8 * 8 * 8 * sizeof(double) * 2);

  // make the appropriate plan
  fftw_plan plan = Create_DCT_Plan(in, direction);
  
  for (auto itr = V.begin(); itr != V.end(); ++itr) {
    // take the transform at *itr and overwrite its contents
    DCT_Smart(*itr, plan, in);
  }
  // cout << "Calling fftw_free." << endl;
  // fftw_free(in);
  free(in);
  fftw_destroy_plan(plan);
  fftw_cleanup();
}


void DoBlockDCT(vector<FIELD_3D>& V) {
  TIMER functionTimer(__FUNCTION__);
  for (auto itr = V.begin(); itr != V.end(); ++itr) {
    DCT_in_place(*itr);
  }
}

INTEGER_FIELD_3D EncodeBlock(FIELD_3D& F, int blockNumber, COMPRESSION_DATA& compression_data) {
  TIMER functionTimer(__FUNCTION__);

  const int uRes = F.xRes();
  const int vRes = F.yRes();
  const int wRes = F.zRes();
  FIELD_3D F_quantized(uRes, vRes, wRes);
  // what we will return
  INTEGER_FIELD_3D quantized(uRes, vRes, wRes);
  
  double q = compression_data.get_q();
  double power = compression_data.get_power();
  int nBits = compression_data.get_nBits();
  FIELD_3D dampingArray = compression_data.get_dampingArray();
  int numBlocks = compression_data.get_numBlocks();
  assert(blockNumber >=0 && blockNumber < numBlocks);
  
  VECTOR sList = compression_data.get_sList();
  if (sList.size() == 0) { // if it's the first time EncodeBlock is called in a chain
  sList.resizeAndWipe(numBlocks);
}

const double Fmax = F(0, 0, 0);                                 // use the DC component as the maximum
double s = (pow(2.0, nBits - 1) - 1) / Fmax;                    // a scale factor for an integer representation
// cout << "Encode thinks s is: " << s << endl;

// assign the next s value to sList
sList[blockNumber] = s; 
F_quantized = F * s;
F_quantized /= dampingArray;

// RoundFieldToInt is apparently a bit of a bottleneck!
quantized = RoundFieldToInt(F_quantized);

// update sList within compression data
compression_data.set_sList(sList);

return quantized;
}


FIELD_3D DecodeBlock(const INTEGER_FIELD_3D& intBlock, int blockNumber, int col, const COMPRESSION_DATA& data, const DECOMPRESSION_DATA& decompression_data) {

  TIMER functionTimer(__FUNCTION__);

  int numBlocks = data.get_numBlocks();
  // make sure we are not accessing an invalid block
  assert( (blockNumber >= 0) && (blockNumber < numBlocks) );

  // we use u, v, w rather than x, y , z to indicate the spatial frequency domain

  const int uRes = intBlock.xRes();
  const int vRes = intBlock.yRes();
  const int wRes = intBlock.zRes();

  // use the appropriate scale factor to decode
  MATRIX sListMatrix = decompression_data.get_sListMatrix();
  double s = sListMatrix(blockNumber, col);
  
  // dequantize by inverting the scaling by s and contracting by the damping array
  FIELD_3D dampingArray = data.get_dampingArray();
  FIELD_3D dequantized_F(uRes, vRes, wRes);
  dequantized_F = CastIntFieldToDouble(intBlock);
  dequantized_F *= (1.0 / s);
  dequantized_F *= dampingArray;

  // take the IDCT
  FIELD_3D dequantized_F_hat = IDCT(dequantized_F);
  return dequantized_F_hat;    
}

// no IDCT in this one!
FIELD_3D DecodeBlockSmart(const INTEGER_FIELD_3D& intBlock, int blockNumber, COMPRESSION_DATA& data) { 
  TIMER functionTimer(__FUNCTION__);

  int numBlocks = data.get_numBlocks();
  // make sure we are not accessing an invalid block
  assert( (blockNumber >= 0) && (blockNumber < numBlocks) );

  // we use u, v, w rather than x, y , z to indicate the spatial frequency domain

  const int uRes = intBlock.xRes();
  const int vRes = intBlock.yRes();
  const int wRes = intBlock.zRes();

  // use the appropriate scale factor to decode
  VECTOR sList = data.get_sList();
  double s = sList[blockNumber];
    
  // dequantize by inverting the scaling by s and contracting by the damping array
  FIELD_3D dampingArray = data.get_dampingArray();
  FIELD_3D dequantized_F(uRes, vRes, wRes);
  dequantized_F = CastIntFieldToDouble(intBlock);
  dequantized_F *= (1.0 / s);
  dequantized_F *= dampingArray;


  return dequantized_F;    
}

FIELD_3D DecodeBlockOld(const INTEGER_FIELD_3D& intBlock, int blockNumber, COMPRESSION_DATA& data) { 
  TIMER functionTimer(__FUNCTION__);

  int numBlocks = data.get_numBlocks();
  // make sure we are not accessing an invalid block
  assert( (blockNumber >= 0) && (blockNumber < numBlocks) );

  // we use u, v, w rather than x, y , z to indicate the spatial frequency domain

  const int uRes = intBlock.xRes();
  const int vRes = intBlock.yRes();
  const int wRes = intBlock.zRes();

  // use the appropriate scale factor to decode
  VECTOR sList = data.get_sList();
  double s = sList[blockNumber];
    
  // dequantize by inverting the scaling by s and contracting by the damping array
  FIELD_3D dampingArray = data.get_dampingArray();
  FIELD_3D dequantized_F(uRes, vRes, wRes);
  dequantized_F = CastIntFieldToDouble(intBlock);
  dequantized_F *= (1.0 / s);
  dequantized_F *= dampingArray;

  // take the IDCT
  FIELD_3D dequantized_F_hat = IDCT(dequantized_F);
  return dequantized_F_hat;    
}

void RunLengthEncodeBinary(const char* filename, int blockNumber, int* zigzaggedArray, VECTOR& blockLengths) { 
  TIMER functionTimer(__FUNCTION__);
  // blockLengths will be modified to keep track of how long
  // each block is for the decoder

  FILE* pFile;
  pFile = fopen(filename, "ab+");    // open a file in append mode since we will call this function repeatedly
  if (pFile == NULL) {
    perror ("Error opening file.");
  }
  else {

    // vector<short> dataList;            // a C++ vector container for our data (int16s)
    vector<short> dataList(8 * 8 * 8 * 2);
    auto itr = dataList.begin();
    short data;
    short runLength;
    int encodedLength = 0;             // variable used to keep track of how long our code is for the decoder

    // assuming 8 x 8 x 8 blocks
    int length = 8 * 8 * 8;
    for (int i = 0; i < length; i++) {
      data = zigzaggedArray[i];
      // dataList.push_back(data);
      *itr = data;
      encodedLength++;

      runLength = 1;
      while ( (i + 1 < length) && (zigzaggedArray[i] == zigzaggedArray[i + 1]) ) {
            // i + 1 < length ensures that i + 1 doesn't go out of bounds for zigzaggedArray[]
            runLength++;
            i++;
      }
      if (runLength > 1) {
        // use a single repeated value as an 'escape' to indicate a run
        // dataList.push_back(data);
        ++itr;
        *itr = data;
        
        encodedLength++;

        // push the runLength to the data vector
        // dataList.push_back(runLength);
        ++itr; 
        *itr = runLength;
        encodedLength++;
      }
      ++itr;
    }
    // cout << "Encoded length is: " << encodedLength << endl;
    blockLengths[blockNumber] = encodedLength;

    fwrite(&(dataList[0]), sizeof(short), encodedLength, pFile);
    // this write assumes that C++ vectors are stored in contiguous memory!
    
    fclose(pFile);
    return;
  }
}

void ReadBinaryFileToMemory(const char* filename, short*& allData, COMPRESSION_DATA& compression_data, DECOMPRESSION_DATA& decompression_data) {
  TIMER functionTimer(__FUNCTION__);
  // allData is passed by reference and will be modified, as will decompression_data

  FILE* pFile;

  // open in '+' mode for fseek
  pFile = fopen(filename, "rb+");
  if (pFile == NULL) {
    perror("Error opening file.");
    exit(EXIT_FAILURE);
  }

  else {
    int numBlocks = compression_data.get_numBlocks();
    int numCols = compression_data.get_numCols();
    int totalSize = numBlocks * numCols;

    double* double_dummy = (double*) malloc(totalSize * sizeof(double));
    fread(double_dummy, totalSize, sizeof(double), pFile);
    VECTOR flattened_s = CastToVector(double_dummy, totalSize);
    free(double_dummy);
    MATRIX sMatrix(flattened_s, numBlocks, numCols);
    decompression_data.set_sListMatrix(sMatrix);

    short* short_dummy = (short*) malloc(totalSize * sizeof(short));
    fread(short_dummy, totalSize, sizeof(short), pFile);
    VECTOR flattened_lengths = CastToVector(short_dummy, totalSize);
    free(short_dummy);
    MATRIX blockLengthsMatrix(flattened_lengths, numBlocks, numCols);
    int totalLength = blockLengthsMatrix.sum();
    decompression_data.set_blockLengthsMatrix(blockLengthsMatrix);

    int* int_dummy = (int*) malloc(totalSize * sizeof(int));
    fread(int_dummy, totalSize, sizeof(int), pFile);
    VECTOR flattened_indices = CastToVector(int_dummy, totalSize);
    free(int_dummy);
    MATRIX blockIndicesMatrix(flattened_indices, numBlocks, numCols);
    decompression_data.set_blockIndicesMatrix(blockIndicesMatrix);


    allData = (short*) malloc(totalLength * sizeof(short));
    if (allData == NULL) {
      perror("Malloc failed to allocate allData!");
      exit(EXIT_FAILURE);
    }
    fread(allData, totalLength, sizeof(short), pFile);
    }
  return;
}


vector<short> RunLengthDecodeBinary(const short* allData, int blockNumber, VECTOR& blockLengths, VECTOR& blockIndices) {
  TIMER functionTimer(__FUNCTION__);
  // although blockLengths and blockIndices are passed by reference,
  // they will not be modified.
    
    // what we will be returning
    vector<short> parsedData;                                
    
    int blockSize = blockLengths[blockNumber];
    int blockIndex = blockIndices[blockNumber];
    
    short* blockData = (short*) malloc(blockSize * sizeof(short));

    if (blockData == NULL) {
      perror("Malloc failed to allocate blockData!");
      exit(EXIT_FAILURE);
    }
    
    
    for (int i = 0; i < blockSize; i++) {
      blockData[i] = allData[blockIndex + i];
    }
    
    int i = 0;
    int runLength;
    while (i < blockSize) {
      parsedData.push_back(blockData[i]);          // write the value once
      if (blockData[i] == blockData[i + 1]) {      // if we read an 'escape' value, it indicates a run.
        i+=2;                                      // advance past the escape value to the run length value.
        runLength = blockData[i];
        for (int j = 0; j < runLength - 1; j++) {  // write the original value (index i - 2) repeatedly for runLength - 1 times,
          parsedData.push_back(blockData[i - 2]);  // since we already wrote it once
        }

      }
      i++;

    }

    free(blockData);
    return parsedData;
  }
  

  void DeleteIfExists(const char* filename) {
  TIMER functionTimer(__FUNCTION__);
    struct stat buf;                   // dummy to pass in to stat()
    if (stat(filename, &buf) == 0) {   // if a file named 'filename' already exists
      cout << filename << " exists; deleting it first..." << endl;
      if ( remove(filename) != 0 ) {
        perror("Error deleting file.");
        return;
      }
      else {
        cout << filename << " successfully deleted." << endl;
        return;
      }
    }
    else {                               // file does not yet exist
      cout << filename << " does not yet exist." << endl;
    }
    return;
  }
 void CompressAndWriteFieldSmart(const char* filename, const FIELD_3D& F, COMPRESSION_DATA& compression_data) {
  TIMER functionTimer(__FUNCTION__);
  
    int numBlocks = compression_data.get_numBlocks();
    INTEGER_FIELD_3D zigzagArray = compression_data.get_zigzagArray();
    vector<FIELD_3D> blocks = GetBlocks(F);

    // Initialize the relevant variables before looping through all the blocks
    VECTOR blockLengths(numBlocks);
    FIELD_3D block_i(8, 8, 8);
    INTEGER_FIELD_3D intEncoded_i(8, 8, 8);
    VECTOR zigzagged_i(8 * 8 * 8);
    int* zigzagArray_i = (int*) malloc(sizeof(int) * 8 * 8 * 8);
    if (zigzagArray_i == NULL) {
      perror("Malloc failed to allocate zigzagArray_i!");
      exit(1);
    }
   
    // do the forward transform 
    DoSmartBlockDCT(blocks, 1);
    // loop through the blocks and apply the encoding procedure
    for (int i = 0; i < numBlocks; i++) {
      block_i = blocks[i];
      // performs quantization and damping. updates sList
      intEncoded_i = EncodeBlock(block_i, i, compression_data);
      // zigzagged_i = ZigzagFlatten(intEncoded_i);
      zigzagged_i = ZigzagFlattenSmart(intEncoded_i, zigzagArray);
      zigzagArray_i = CastToInt(zigzagged_i, zigzagArray_i);
      // performs run-length encoding. updates blockLengths
      RunLengthEncodeBinary(filename, i, zigzagArray_i, blockLengths);  
    }
    
    // update the compression data
    compression_data.set_blockLengths(blockLengths);
    VECTOR blockIndices = ModifiedCumSum(blockLengths);
    compression_data.set_blockIndices(blockIndices);
    free(zigzagArray_i);
    return;
  }

  /*
  void CompressAndWriteField(const char* filename, const FIELD_3D& F, COMPRESSION_DATA& compression_data) {
  TIMER functionTimer(__FUNCTION__);
  
    int numBlocks = compression_data.get_numBlocks();
    vector<FIELD_3D> blocks = GetBlocks(F);

    // Initialize the relevant variables before looping through all the blocks
    VECTOR blockLengths(numBlocks);
    FIELD_3D block_i(8, 8, 8);
    INTEGER_FIELD_3D intEncoded_i(8, 8, 8);
    VECTOR zigzagged_i(8 * 8 * 8);
    int* zigzagArray_i = (int*) malloc(sizeof(int) * 8 * 8 * 8);
    if (zigzagArray_i == NULL) {
      perror("Malloc failed to allocate zigzagArray_i!");
      exit(1);
    }
    
    // loop through the blocks and apply the encoding procedure
    for (int i = 0; i < numBlocks; i++) {
      block_i = blocks[i];
      DCT_in_place(block_i);
      // performs quantization and damping. updates sList
      intEncoded_i = EncodeBlock(block_i, i, compression_data);
      zigzagged_i = ZigzagFlatten(intEncoded_i);
      zigzagArray_i = CastToInt(zigzagged_i, zigzagArray_i);
      // performs run-length encoding. updates blockLengths
      RunLengthEncodeBinary(filename, i, zigzagArray_i, blockLengths);  
    }
    
    // update the compression data
    compression_data.set_blockLengths(blockLengths);
    VECTOR blockIndices = ModifiedCumSum(blockLengths);
    compression_data.set_blockIndices(blockIndices);
    free(zigzagArray_i);
    return;
  }
  */

  int ComputeBlockNumber(int row, int col, VEC3I dims, int& blockIndex) {
  TIMER functionTimer(__FUNCTION__);
    int xRes = dims[0];
    int yRes = dims[1];
    int zRes = dims[2];
    
    assert( row >= 0 && row < 3 * xRes * yRes * zRes);
    assert( col >= 0); 
   

    // evil integer division!
    int index = row / 3;
    int z = index / (xRes * yRes);         // index = (xRes * yRes) * z + remainder1
    // cout << "z: " << z << endl;
    int rem1 = index - (xRes * yRes * z);
    int y = rem1 / xRes;                   // rem1  = xRes * y          + remainder2 
    // cout << "y: " << y << endl;
    int rem2 = rem1 - xRes * y;
    int x = rem2;
    // cout << "x: " << x << endl;

    int u = x % 8;
    int v = y % 8;
    int w = z % 8;
    blockIndex = u + 8 * v + 8 * 8 *w;
    // cout << "block index: " << blockIndex << endl;

    assert(index == z * xRes * yRes + y * xRes + x);

    
     
    int xPadding;
    int yPadding;
    int zPadding;
    
    GetPaddings(dims, xPadding, yPadding, zPadding);
    xRes += xPadding;
    yRes += yPadding;
    zRes += zPadding;
    
    // more evil integer division!
    int blockNumber = x/8 + (y/8 * (xRes/8)) + (z/8 * (xRes/8) * (yRes/8));

    return blockNumber;
  }

  
double DecodeFromRowCol(int row, int col, const MATRIX_COMPRESSION_DATA& data) { 
     
  TIMER functionTimer(__FUNCTION__);

  COMPRESSION_DATA compression_data = data.get_compression_data();
  VEC3I dims = compression_data.get_dims();
  INTEGER_FIELD_3D zigzagArray = compression_data.get_zigzagArray();
  // double q = compression_data.get_q();
  // double power = compression_data.get_power();
  
  short* allDataX = data.get_dataX();
  short* allDataY = data.get_dataY();
  short* allDataZ = data.get_dataZ();
  DECOMPRESSION_DATA dataX = data.get_decompression_dataX();
  DECOMPRESSION_DATA dataY = data.get_decompression_dataY();
  DECOMPRESSION_DATA dataZ = data.get_decompression_dataZ();


  vector<short> decoded_runLength;

  // dummy initialization
  int blockIndex = 0;
  int blockNumber = ComputeBlockNumber(row, col, dims, blockIndex);
  if (row % 3 == 0) { // X coordinate
    MATRIX blockLengthsMatrix = dataX.get_blockLengthsMatrix();
    MATRIX blockIndicesMatrix = dataX.get_blockIndicesMatrix();
    MATRIX sListMatrix = dataX.get_sListMatrix();
        
    
    VECTOR blockLengths = blockLengthsMatrix.getColumn(col);
    VECTOR blockIndices = blockIndicesMatrix.getColumn(col);
    VECTOR sList = sListMatrix.getColumn(col);
    decoded_runLength = RunLengthDecodeBinary(allDataX, blockNumber, blockLengths, blockIndices); 

    VECTOR decoded_runLengthVector = CastIntToVector(decoded_runLength);
    INTEGER_FIELD_3D unzigzagged = ZigzagUnflattenSmart(decoded_runLengthVector, zigzagArray);
    FIELD_3D decoded_block = DecodeBlock(unzigzagged, blockNumber, col, compression_data, dataX); 
    // cout << "desired value is: " << decoded_block[blockIndex] << endl;
    double result = decoded_block[blockIndex];
    return result;

  }

  else if (row % 3 == 1) { // Y coordinate
  
    MATRIX blockLengthsMatrix = dataY.get_blockLengthsMatrix();
    MATRIX blockIndicesMatrix = dataY.get_blockIndicesMatrix();
    MATRIX sListMatrix = dataY.get_sListMatrix();



    VECTOR blockLengths = blockLengthsMatrix.getColumn(col);
    VECTOR blockIndices = blockIndicesMatrix.getColumn(col);
    VECTOR sList = sListMatrix.getColumn(col);

    decoded_runLength = RunLengthDecodeBinary(allDataY, blockNumber, blockLengths, blockIndices); 

    VECTOR decoded_runLengthVector = CastIntToVector(decoded_runLength);
    INTEGER_FIELD_3D unzigzagged = ZigzagUnflattenSmart(decoded_runLengthVector, zigzagArray);
    FIELD_3D decoded_block = DecodeBlock(unzigzagged, blockNumber, col, compression_data, dataY); 
    // cout << "desired value is: " << decoded_block[blockIndex] << endl;
    double result = decoded_block[blockIndex];
    return result;
  
  }

  else { // Z coordinate
   
    MATRIX blockLengthsMatrix = dataZ.get_blockLengthsMatrix();
    MATRIX blockIndicesMatrix = dataZ.get_blockIndicesMatrix();
    MATRIX sListMatrix = dataZ.get_sListMatrix();


    VECTOR blockLengths = blockLengthsMatrix.getColumn(col);
    VECTOR blockIndices = blockIndicesMatrix.getColumn(col);
    VECTOR sList = sListMatrix.getColumn(col);

    decoded_runLength = RunLengthDecodeBinary(allDataZ, blockNumber, blockLengths, blockIndices); 

    VECTOR decoded_runLengthVector = CastIntToVector(decoded_runLength);
    INTEGER_FIELD_3D unzigzagged = ZigzagUnflattenSmart(decoded_runLengthVector, zigzagArray);
    FIELD_3D decoded_block = DecodeBlock(unzigzagged, blockNumber, col, compression_data, dataZ);
    // cout << "desired value is: " << decoded_block[blockIndex] << endl;
    double result = decoded_block[blockIndex];
    return result;
   
  }
}
  
  MATRIX GetSubmatrix(int startRow, int numRows, const MATRIX_COMPRESSION_DATA& data) {
     
    
    TIMER functionTimer(__FUNCTION__);

    assert(numRows == 3);
    COMPRESSION_DATA compression_data = data.get_compression_data();
    int numCols = compression_data.get_numCols();
    MATRIX result(numRows, numCols);

    for (int col = 0; col < numCols; col++) {
      for (int row = 0; row < numRows; row++) {
        double value = DecodeFromRowCol(row + startRow, col, data); 
        result(row, col) = value;
      }
    }
    return result;
  }





  void WriteMetaData(const char* filename, const MATRIX& sListMatrix, const MATRIX& blockLengthsMatrix, const MATRIX& blockIndicesMatrix) {
  TIMER functionTimer(__FUNCTION__);
    FILE* pFile;
    pFile = fopen(filename, "wb");
    if (pFile == NULL) {
      perror ("Error opening file.");
    }
    else {
      
      VECTOR flattened_s = sListMatrix.flattenedColumn();
      int blocksXcols = flattened_s.size();

      double* sData = (double*) malloc(sizeof(double) * blocksXcols);
      // fill sData
      sData = CastToDouble(flattened_s, sData);
      fwrite(sData, sizeof(double), blocksXcols, pFile);

      VECTOR flattened_lengths = blockLengthsMatrix.flattenedColumn();
      assert(flattened_lengths.size() == blocksXcols);

      short* lengthsData = (short*) malloc(sizeof(short) * blocksXcols);
      // fill lengthsData
      lengthsData = CastToShort(flattened_lengths, lengthsData);
      fwrite(lengthsData, sizeof(short), blocksXcols, pFile);
      
      VECTOR flattened_indices = blockIndicesMatrix.flattenedColumn();
      assert(flattened_indices.size() == blocksXcols);

      int* indicesData = (int*) malloc(sizeof(int) * blocksXcols);
      indicesData = CastToInt(flattened_indices, indicesData);
      fwrite(indicesData, sizeof(int), blocksXcols, pFile);

      fclose(pFile);
      free(sData);
      free(lengthsData);
      free(indicesData);
    }
  }


void PrefixBinary(string prefix, string filename, string newFile) {
  TIMER functionTimer(__FUNCTION__);
  string command = "cat " + prefix + ' ' + filename + "> " + newFile;
  const char* command_c = command.c_str();
  system(command_c);
}

void CleanUpPrefix(const char* prefix, const char* filename) {
  TIMER functionTimer(__FUNCTION__);
  string prefix_string(prefix);
  string filename_string(filename);
  string command1 = "rm " + prefix_string;
  string command2 = "rm " + filename_string;
  const char* command1_c = command1.c_str();
  const char* command2_c = command2.c_str();

  system(command1_c);
  system(command2_c);
}

void CompressAndWriteMatrixComponent(const char* filename, const MatrixXd& U, int component, COMPRESSION_DATA& data) {
  TIMER functionTimer(__FUNCTION__);

  assert( component >= 0 && component < 3 );

  string final_string(filename);
  if (component == 0) {
    final_string += 'X';
  }
  else if (component == 1) {
    final_string += 'Y';
  }
  else { // component == 2
    final_string += 'Z';
  }
  
  VEC3I dims = data.get_dims();
  int xRes = dims[0];
  int yRes = dims[1];
  int zRes = dims[2];
  int numBlocks = data.get_numBlocks();
  int numCols = U.cols();
  
  // initialize appropriately-sized matrices for the decoder data
  MATRIX blockLengthsMatrix(numBlocks, numCols);
  MATRIX sListMatrix(numBlocks, numCols);
  MATRIX blockIndicesMatrix(numBlocks, numCols);

  // wipe any pre-existing binary file of the same name, since we will be opening
  // in append mode! 
  DeleteIfExists(filename);

  double percent = 0.0;
  for (int col = 0; col < numCols; col++) {  
    VectorXd vXd = U.col(col);
    VECTOR v = EIGEN::convert(vXd);
    VECTOR3_FIELD_3D V(v, xRes, yRes, zRes);
    FIELD_3D F = V.scalarField(component);
    
    // CompressAndWriteField(filename, F, data); 
    CompressAndWriteFieldSmart(filename, F, data);

    // update blockLengths and sList and push them to the appropriate column
    // of their respective matrices
    VECTOR blockLengths = data.get_blockLengths();
    VECTOR sList = data.get_sList();
    blockLengthsMatrix.setColumn(blockLengths, col);
    sListMatrix.setColumn(sList, col);

    percent = col / (double) numCols;
    int checkPoint1 = (numCols - 1) / 4;
    int checkPoint2 = (numCols - 1) / 2;
    int checkPoint3 = (3 * (numCols - 1)) / 4;
    int checkPoint4 = numCols - 1;
    if (col == checkPoint1 || col == checkPoint2 || col == checkPoint3 || col == checkPoint4) {
      cout << "    Percent: " << percent << flush;
      if (col == checkPoint4) {
        cout << endl;
      }
    }
       
    
  }
  
  // build the block indices matrix from the block lengths matrix
  blockIndicesMatrix = ModifiedCumSum(blockLengthsMatrix);
  
  const char* metafile = "metadata.bin"; 
  WriteMetaData(metafile, sListMatrix, blockLengthsMatrix, blockIndicesMatrix);
  // appends the metadata as a header to the main binary file and pipes them into final_string
  PrefixBinary(metafile, filename, final_string);
  // removes the now-redundant metadata and main binary files
  CleanUpPrefix(metafile, filename);

}
    
