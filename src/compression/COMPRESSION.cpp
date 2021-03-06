#include <iostream>
// #include <fftw3.h>
// #include <sys/stat.h>  // for using stat to check whether a file exists
// #include <numeric>     // for std::accumulate
// #include <assert.h>
#include <iomanip>

#include "EIGEN.h"
// #include "SUBSPACE_FLUID_3D_EIGEN.h"
// #include "FLUID_3D_MIC.h"
#include "MATRIX.h"
// #include "SIMPLE_PARSER.h"
#include "COMPRESSION.h"
// #include "INTEGER_FIELD_3D.h"
// #include "COMPRESSION_DATA.h"
// #include "DECOMPRESSION_DATA.h"
#include "MATRIX_COMPRESSION_DATA.h"
// #include "FIELD_3D.h"
// #include "CUBATURE_GENERATOR_EIGEN.h"

using std::vector;
using std::accumulate;
using std::cout;
using std::endl;

////////////////////////////////////////////////////////
// Function Implementations
////////////////////////////////////////////////////////


////////////////////////////////////////////////////////
// fill a VECTOR from a buffer of doubles
////////////////////////////////////////////////////////
VECTOR CastToVector(double* data, int size) {
  TIMER functionTimer(__FUNCTION__);
  VECTOR x(size);
  for (int i = 0; i < size; i++) {
    x[i] = data[i];
  }
  return x;
}

////////////////////////////////////////////////////////
// fill a VECTOR from a buffer of ints
////////////////////////////////////////////////////////
VECTOR CastToVector(int* data, int size) {
  TIMER functionTimer(__FUNCTION__);
  VECTOR x(size);
  for (int i = 0; i < size; i++) {
    x[i] = data[i];
  }
  return x;
}

/*
////////////////////////////////////////////////////////
// fill a VECTOR from a buffer of ints
////////////////////////////////////////////////////////
VECTOR CastToVector(int* data, int size) {
  TIMER functionTimer(__FUNCTION__);
  VECTOR x(size);
  for (int i = 0; i < size; i++) {
    x[i] = data[i];
  }
  return x;
}
*/

////////////////////////////////////////////////////////
// fill a buffer of doubles from a VECTOR 
////////////////////////////////////////////////////////
double* CastToDouble(const VECTOR& x, double* array) {
  TIMER functionTimer(__FUNCTION__);
  for (int i = 0; i < x.size(); i++) {
    array[i] = x[i];
  }
  return array;
}


////////////////////////////////////////////////////////
// fill a buffer of ints from a VECTOR
////////////////////////////////////////////////////////
int* CastToInt(const VECTOR& x, int* array) {
  TIMER functionTimer(__FUNCTION__);
  for (int i = 0; i < x.size(); i++) {
    int data_i = (int) x[i];
    array[i] = data_i;
  }
  return array;
}


////////////////////////////////////////////////////////
// convert a vector<int> to a VECTOR
////////////////////////////////////////////////////////
void CastIntToVectorFast(const vector<int>& V, VECTOR& vecToFill) {
  TIMER functionTimer(__FUNCTION__);
  int length = V.size();
  assert(vecToFill.size() == length);

  for (int i = 0; i < length; i++) {
    vecToFill[i] = V[i];
  }
}
////////////////////////////////////////////////////////
// convert a vector<int> to a VECTOR
////////////////////////////////////////////////////////
VECTOR CastIntToVector(const vector<int>& V) {
  TIMER functionTimer(__FUNCTION__);
  int length = V.size();
  VECTOR result(length);

  for (int i = 0; i < length; i++) {
    result[i] = V[i];
  }
  return result;
}

////////////////////////////////////////////////////////
// convert a VECTOR to a vector<int>
////////////////////////////////////////////////////////
vector<int> CastVectorToInt(const VECTOR& V) {
  TIMER functionTimer(__FUNCTION__);
  int length = V.size();
  vector<int> result(length, 0);
  for (int i = 0; i < length; i++) {
    result[i] = V[i];
  }
  return result;
}

////////////////////////////////////////////////////////
// convert a VECTOR to a vector<double> 
////////////////////////////////////////////////////////
vector<double> CastVectorToDouble(const VECTOR& V) {
  TIMER functionTimer(__FUNCTION__);
  int length = V.size();
  vector<double> result(length, 0.0);
  for (int i = 0; i < length; i++) {
    result[i] = V[i];
  } 
  return result;
}

////////////////////////////////////////////////////////
// convert a vector<double> to a VECTOR
////////////////////////////////////////////////////////
VECTOR CastDoubleToVector(const vector<double>& V) {
  TIMER functionTimer(__FUNCTION__);
  int length = V.size();
  VECTOR result(length);

  for (int i = 0; i < length; i++) {
    result[i] = V[i];
  }
  return result;
}


////////////////////////////////////////////////////////
// cast a FIELD_3D to an INTEGER_FIELD_3D by rounding 
////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////
// cast an INTEGER_FIELD_3D to a FIELD_3D
////////////////////////////////////////////////////////
void CastIntFieldToDouble(const INTEGER_FIELD_3D& F, FIELD_3D& castedField) {
  TIMER functionTimer(__FUNCTION__);
  
  int totalCells = F.totalCells();

  for (int i = 0; i < totalCells; i++) {
    double casted = (double) F[i];
    castedField[i] = casted;
  }

}


////////////////////////////////////////////////////////
// operates on a vector<int> and returns a vector<int>
// of its cum sum starting at zero and omitting the last
// entry. e.g. if the input vector was (1, 2, 3, 4),
// the result would be (0, 1, 3, 5)
////////////////////////////////////////////////////////
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

////////////////////////////////////////////////////////
// same as the other ModifiedCumSum, but operates
// on VECTORs
////////////////////////////////////////////////////////
VECTOR ModifiedCumSum(const VECTOR& V) {
  TIMER functionTimer(__FUNCTION__);
  vector<int> V_int = CastVectorToInt(V);
  vector<int> result_int = ModifiedCumSum(V_int);
  VECTOR result = CastIntToVector(result_int);
  return result;
}


////////////////////////////////////////////////////////
// operates on a MATRIX. treats each column of a matrix
// separately. performs ModifiedCumSum on each one.
////////////////////////////////////////////////////////
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


////////////////////////////////////////////////////////
// returns the 3 component scalar fields 
// from a passed in vector field 
////////////////////////////////////////////////////////
void GetScalarFields(const VECTOR3_FIELD_3D& V, FIELD_3D& X, FIELD_3D& Y, FIELD_3D& Z) {
  TIMER functionTimer(__FUNCTION__);
  X = V.scalarField(0);
  Y = V.scalarField(1);
  Z = V.scalarField(2);
}





////////////////////////////////////////////////////////
// Since the scan always follows the same order,
// we precompute the zigzag scan array, pass it
// as a parameter, and then just do an index lookup
////////////////////////////////////////////////////////
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

////////////////////////////////////////////////////////
// inverse function of ZigzagFlatten. 
// ZigzagUnflattenSmart is a better implementation
////////////////////////////////////////////////////////
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

////////////////////////////////////////////////////////
// like the other 'smart' version, uses the
// precomputed zigzagArray and simple lookups
////////////////////////////////////////////////////////
void ZigzagUnflattenSmart(const VECTOR& V, const INTEGER_FIELD_3D& zigzagArray, INTEGER_FIELD_3D& unflattened) {
  TIMER functionTimer(__FUNCTION__);
  // assumes original dimensions were 8 x 8 x 8
  const int xRes = 8;
  const int yRes = 8;
  const int zRes = 8;
  
  assert( xRes == unflattened.xRes() && yRes == unflattened.yRes() && zRes == unflattened.zRes() );
  
  /*
  for (int z = 0; z < zRes; z++) {
    for (int y = 0; y < yRes; y++) {
      for (int x = 0; x < xRes; x++) {
        int index = zigzagArray(x, y, z);
        unflattened(x, y, z) = V[index];
      }
    }
  }
  */
  const int totalCells = zigzagArray.totalCells();
  for (int i = 0; i < totalCells; i++) {
    int index = zigzagArray[i];
    unflattened[i] = V[index];
  }

}



////////////////////////////////////////////////////////
// Given a passed in buffer and a 'direction' 
// (1 for forward, -1 for inverse),
// we return an fftw plan for doing an in-place  3d dct 
// which is linked to the in buffer
////////////////////////////////////////////////////////
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
  // 'in' appears twice since it is in-place
  plan = fftw_plan_r2r_3d(zRes, yRes, xRes, in, in, kind, kind, kind, FFTW_MEASURE);
  
  return plan;
} 


////////////////////////////////////////////////////////
// given a passed in FIELD_3D, fftw plan, and 
// corresponding 'in' buffer, performs the corresponding
// transform on the field 
////////////////////////////////////////////////////////
void DCT_Smart(FIELD_3D& F, fftw_plan& plan, double*& in) {
  TIMER functionTimer(__FUNCTION__);

  int xRes = F.xRes();
  int yRes = F.yRes();
  int zRes = F.zRes();

  // fill the 'in' buffer
  in = CastToDouble(F.flattened(), in);
  
  {
    TIMER fftTimer("fftw execute");
  fftw_execute(plan);
  }
  // 'in' is now overwritten to the result of the transform

  // read 'in' into F_hat
  FIELD_3D F_hat(in, xRes, yRes, zRes);
 
  // normalize symmetrically
  F_hat *= sqrt(0.125 / (xRes * yRes * zRes));

  // rewrite F with the contents of F_hat
  F.swapPointers(F_hat);
}

////////////////////////////////////////////////////////
// defunct version of DCT since it does too many
// malloc/frees when called repeatedly
////////////////////////////////////////////////////////
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




////////////////////////////////////////////////////////
// fetches the plan an input buffer from the passed in
// decompression data, so make sure all is initialized
// before calling this function! 
////////////////////////////////////////////////////////
void IDCT_Smart_Fast(FIELD_3D& F_hat, const DECOMPRESSION_DATA& decompression_data, FIELD_3D& fieldToFill) {
  TIMER functionTimer(__FUNCTION__);
 
  double* in = decompression_data.get_dct_in();
  double* out = decompression_data.get_dct_out();
  fftw_plan plan = decompression_data.get_dct_plan();
  const int xRes = F_hat.xRes();
  const int yRes = F_hat.yRes();
  const int zRes = F_hat.zRes();
  
  // VECTOR vector_in = F_hat.flattened();
  in = CastToDouble(F_hat.flattened(), in);
   
  {
    TIMER fftw2Timer("fftw execute idct smart fast");
  fftw_execute(plan);
  }
  FIELD_3D F(out, xRes, yRes, zRes);

  // normalize symmetrically
  F *= sqrt(0.125 / (xRes * yRes * zRes));
  F.swapPointers(fieldToFill);
}
////////////////////////////////////////////////////////
// fetches the plan an input buffer from the passed in
// decompression data, so make sure all is initialized
// before calling this function! 
////////////////////////////////////////////////////////
FIELD_3D IDCT_Smart(FIELD_3D& F_hat, const DECOMPRESSION_DATA& decompression_data) {
  TIMER functionTimer(__FUNCTION__);
 
  double* in = decompression_data.get_dct_in();
  double* out = decompression_data.get_dct_out();
  fftw_plan plan = decompression_data.get_dct_plan();
  const int xRes = F_hat.xRes();
  const int yRes = F_hat.yRes();
  const int zRes = F_hat.zRes();
  
  in = CastToDouble(F_hat.flattened(), in);

  fftw_execute(plan);
  FIELD_3D F(out, xRes, yRes, zRes);

  // normalize symmetrically
  F *= sqrt(0.125 / (xRes * yRes * zRes));

  return F;
}
  
  
////////////////////////////////////////////////////////
// performs 'DoSmartBlockCompression' on each
// individual scalar field of a passed in vector field,
// then reassembles the result into a lossy vector field 
////////////////////////////////////////////////////////
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


////////////////////////////////////////////////////////
// performs 8 x 8 x 8 3D DCT block compression on the
// passed in FIELD_3D, returning a lossy version 
////////////////////////////////////////////////////////
FIELD_3D DoSmartBlockCompression(FIELD_3D& F, COMPRESSION_DATA& compression_data) {
  TIMER functionTimer(__FUNCTION__);

  int xRes = F.xRes();
  int xResOriginal = xRes;
  int yRes = F.yRes();
  int yResOriginal = yRes;
  int zRes = F.zRes();
  int zResOriginal = zRes;
  
  VEC3I dimsOriginal(xRes, yRes, zRes);
  int numBlocks = compression_data.get_numBlocks();

  // dummy initializations                                     
  int xPadding = 0;
  int yPadding = 0;
  int zPadding = 0;
  
  // fill in the paddings
  GetPaddings(dimsOriginal, xPadding, yPadding, zPadding);
  // update to the padded resolutions
  xRes += xPadding;
  yRes += yPadding;
  zRes += zPadding;
  VEC3I dimsUpdated(xRes, yRes, zRes);

  vector<FIELD_3D> blocks = GetBlocks(F);     
  // 1-->forward transform
  cout << "Doing block forward transform..." << endl;
  DoSmartBlockDCT(blocks, 1);
  cout << "...done!" << endl;
  int blockNumber = 0;
  double percent = 0.0;
  cout << "Doing quantization on each block..." << endl;
  for (auto itr = blocks.begin(); itr != blocks.end(); ++itr) {

    INTEGER_FIELD_3D V = EncodeBlock(*itr, blockNumber, compression_data); 
    FIELD_3D compressedBlock = DecodeBlockSmart(V, blockNumber, compression_data); 
    *itr = compressedBlock;
    blockNumber++;

    // calculate the progress to display for the user
    percent = blockNumber / ( (double) numBlocks );
    int checkPoint1 = (numBlocks - 1) / 4;
    int checkPoint2 = (numBlocks - 1) / 2;
    int checkPoint3 = (3 * (numBlocks - 1)) / 4;
    int checkPoint4 = numBlocks - 1;

    if ( (blockNumber == checkPoint1) || (blockNumber == checkPoint2) ||
       ( blockNumber == checkPoint3) || (blockNumber == checkPoint4)) {
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
  // reconstruct a FIELD_3D from the vector of blocks
  FIELD_3D F_compressed(xRes, yRes, zRes);
  AssimilateBlocks(dimsUpdated, blocks, F_compressed);
  // strip off the padding
  FIELD_3D F_compressed_peeled = F_compressed.subfield(0, xResOriginal, 0, yResOriginal, 0, zResOriginal); 

  return F_compressed_peeled;
}


////////////////////////////////////////////////////////
// given passed in dimensions, computes how much we
// have to pad by in each dimension to reach the next
// multiple of 8 for even block subdivision 
////////////////////////////////////////////////////////
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

////////////////////////////////////////////////////////
// given a passed in FIELD_3D, pad it and  parse it 
// into a vector of 8 x 8 x 8 blocks (listed in row-major order)
////////////////////////////////////////////////////////
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


////////////////////////////////////////////////////////
// Given a passed FIELD_3D, parse it into zero-padded 
// blocks, but flatten the blocks into VectorXd for use in
// projection/unprojection. 
////////////////////////////////////////////////////////
vector<VectorXd> GetBlocksEigen(const FIELD_3D& F) {
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

  FIELD_3D F_padded_x = F.zeroPad_x(xPadding);
  FIELD_3D F_padded_xy = F_padded_x.zeroPad_y(yPadding);
  FIELD_3D F_padded = F_padded_xy.zeroPad_z(zPadding);
  
  // update the resolutions to the padded ones 
  xRes = F_padded.xRes();
  yRes = F_padded.yRes();
  zRes = F_padded.zRes();

  // sanity check that our padder had the desired effect
  assert(xRes % 8 == 0);
  assert(yRes % 8 == 0);
  assert(zRes % 8 == 0);

  // variable initialization before the loop
  int numBlocks = (xRes/8) * (yRes/8) * (zRes/8);
  vector<VectorXd> blockList(numBlocks);
  int index = 0;
  
  for (int z = 0; z < zRes/8; z++) {
    for (int y = 0; y < yRes/8; y++) {
      for (int x = 0; x < xRes/8; x++, index++) {
        FIELD_3D subfield = F_padded.subfield(8*x, 8*(x+1), 8*y, 8*(y+1), 8*z, 8*(z+1));
        VectorXd subfieldFlatEigen = subfield.flattenedEigen();
        blockList[index] = subfieldFlatEigen;
      }
    }
  }
  return blockList;
}

////////////////////////////////////////////////////////
// reconstruct a FIELD_3D with the passed in dims
// from a list of 8 x 8 x 8 blocks 
////////////////////////////////////////////////////////
void AssimilateBlocks(const VEC3I& dims, const vector<FIELD_3D>& V, FIELD_3D& assimilatedField) {
  TIMER functionTimer(__FUNCTION__);
  const int xRes = dims[0];
  const int yRes = dims[1];
  const int zRes = dims[2];

  assert( xRes % 8 == 0 && yRes % 8 == 0 && zRes % 8 == 0 );
  assert( xRes == assimilatedField.xRes() && yRes == assimilatedField.yRes() && zRes == assimilatedField.zRes() );


  for (int z = 0; z < zRes; z++) {
    for (int y = 0; y < yRes; y++) {
      for (int x = 0; x < xRes; x++) {
        int index = (x/8) + (y/8) * (xRes/8) + (z/8) * (xRes/8) * (yRes/8);     // warning, evil integer division happening!
        assimilatedField(x, y, z) = V[index](x % 8, y % 8, z % 8);             
      }
    }
  }

}


////////////////////////////////////////////////////////
// performs a dct/idct on each individual block of a passed in
// vector of blocks
////////////////////////////////////////////////////////
void DoSmartBlockDCT(vector<FIELD_3D>& V, int direction) {
  TIMER functionTimer(__FUNCTION__);
  // direction determines whether it is DCT or IDCT
  
  // allocate a buffer for the size of an 8 x 8 x 8 block
  double* in = (double*) fftw_malloc(8 * 8 * 8 * sizeof(double));

  // make the appropriate plan
  fftw_plan plan = Create_DCT_Plan(in, direction);
  
  for (auto itr = V.begin(); itr != V.end(); ++itr) {
    // take the transform at *itr (which is a FIELD_3D)
    // and overwrite its contents
    DCT_Smart(*itr, plan, in);
  }

  fftw_free(in);
  fftw_destroy_plan(plan);
  fftw_cleanup();
}


////////////////////////////////////////////////////////
// takes a passed in FIELD_3D (which is intended to be
// the result of a DCT), scales it to an nBit integer
// (typically 16), normalizes by the DC coefficient,
// damps by a precomputed damping array, and quantizes 
// to an integer
////////////////////////////////////////////////////////
INTEGER_FIELD_3D EncodeBlock(FIELD_3D& F, int blockNumber, COMPRESSION_DATA& compression_data) {
  TIMER functionTimer(__FUNCTION__);

  const int uRes = F.xRes();
  const int vRes = F.yRes();
  const int wRes = F.zRes();
  FIELD_3D F_quantized(uRes, vRes, wRes);

  // what we will return
  INTEGER_FIELD_3D quantized(uRes, vRes, wRes);
  
  int nBits = compression_data.get_nBits();
  const FIELD_3D& dampingArray = compression_data.get_dampingArray();
  int numBlocks = compression_data.get_numBlocks();
  assert(blockNumber >=0 && blockNumber < numBlocks);
  
  VECTOR sList = compression_data.get_sList();
  if (sList.size() == 0) { // if it's the first time EncodeBlock is called in a chain
    sList.resizeAndWipe(numBlocks);
  }

  const double Fmax = F(0, 0, 0);                                 // use the DC component as the maximum
  double s = (pow(2.0, nBits - 1) - 1) / Fmax;                    // a scale factor for an integer representation

  // assign the next s value to sList
  sList[blockNumber] = s; 
  // scale so that the DC component is 2^(n - 1) - 1
  F_quantized = F * s;
  // spatial damping to stomp high frequencies
  F_quantized /= dampingArray;

  quantized = RoundFieldToInt(F_quantized);

  // update sList within compression data
  compression_data.set_sList(sList);

  return quantized;
}


void DecodeBlockFast(const INTEGER_FIELD_3D& intBlock, int blockNumber, int col, const DECOMPRESSION_DATA& decompression_data, FIELD_3D& fieldToFill) {

  TIMER functionTimer(__FUNCTION__);

  int numBlocks = decompression_data.get_numBlocks();
  // make sure we are not accessing an invalid block
  assert( (blockNumber >= 0) && (blockNumber < numBlocks) );

  // we use u, v, w rather than x, y , z to indicate the spatial frequency domain

  const int uRes = intBlock.xRes();
  const int vRes = intBlock.yRes();
  const int wRes = intBlock.zRes();

  // use the appropriate scale factor to decode
  const MATRIX& sListMatrix = decompression_data.get_sListMatrix();
  double s = sListMatrix(blockNumber, col);
  
  // dequantize by inverting the scaling by s and contracting by the damping array
  const FIELD_3D& dampingArray = decompression_data.get_dampingArray();
  FIELD_3D dequantized_F(uRes, vRes, wRes);
  CastIntFieldToDouble(intBlock, dequantized_F);
  dequantized_F *= (1.0 / s);
  dequantized_F *= dampingArray;
  
  
  // take the IDCT
  // FIELD_3D dequantized_F_hat = IDCT(dequantized_F);
 
  IDCT_Smart_Fast(dequantized_F, decompression_data, fieldToFill);
}

////////////////////////////////////////////////////////
// decodes an INTEGER_FIELD_3D into a FIELD_3D by
// undoing the effects of quantization. in this version,
// an IDCT is also performed.
////////////////////////////////////////////////////////
FIELD_3D DecodeBlock(const INTEGER_FIELD_3D& intBlock, int blockNumber, int col, const DECOMPRESSION_DATA& decompression_data) {

  TIMER functionTimer(__FUNCTION__);

  int numBlocks = decompression_data.get_numBlocks();
  // make sure we are not accessing an invalid block
  assert( (blockNumber >= 0) && (blockNumber < numBlocks) );

  // we use u, v, w rather than x, y , z to indicate the spatial frequency domain

  const int uRes = intBlock.xRes();
  const int vRes = intBlock.yRes();
  const int wRes = intBlock.zRes();

  // use the appropriate scale factor to decode
  const MATRIX& sListMatrix = decompression_data.get_sListMatrix();
  double s = sListMatrix(blockNumber, col);
  
  // dequantize by inverting the scaling by s and contracting by the damping array
  const FIELD_3D& dampingArray = decompression_data.get_dampingArray();
  FIELD_3D dequantized_F(uRes, vRes, wRes);
  CastIntFieldToDouble(intBlock, dequantized_F);
  dequantized_F *= (1.0 / s);
  dequantized_F *= dampingArray;
  
  
  // take the IDCT
  // FIELD_3D dequantized_F_hat = IDCT(dequantized_F);
 
  FIELD_3D dequantized_F_hat = IDCT_Smart(dequantized_F, decompression_data);
  return dequantized_F_hat;    
}


////////////////////////////////////////////////////////
// in this version, there is no IDCT at the end
////////////////////////////////////////////////////////
void DecodeBlockDecomp(const INTEGER_FIELD_3D& intBlock, int blockNumber, int col, const DECOMPRESSION_DATA& decompression_data, FIELD_3D& fieldToFill) {

  TIMER functionTimer(__FUNCTION__);

  int numBlocks = decompression_data.get_numBlocks();
  // make sure we are not accessing an invalid block
  assert( (blockNumber >= 0) && (blockNumber < numBlocks) );

  // we use u, v, w rather than x, y , z to indicate the spatial frequency domain

  const int uRes = intBlock.xRes();
  const int vRes = intBlock.yRes();
  const int wRes = intBlock.zRes();

  // use the appropriate scale factor to decode
  const MATRIX& sListMatrix = decompression_data.get_sListMatrix();
  double s = sListMatrix(blockNumber, col);
  
  // dequantize by inverting the scaling by s and contracting by the damping array
  const FIELD_3D& dampingArray = decompression_data.get_dampingArray();
  fieldToFill.resizeAndWipe(uRes, vRes, wRes);
  CastIntFieldToDouble(intBlock, fieldToFill);
  fieldToFill *= (1.0 / s);
  fieldToFill *= dampingArray;

}

////////////////////////////////////////////////////////
// no IDCT, and no 'col' parameter. the decompression
// data parameter is replaced by a compression data
// parameter 
////////////////////////////////////////////////////////
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
  const VECTOR& sList = data.get_sList();
  double s = sList[blockNumber];
    
  // dequantize by inverting the scaling by s and contracting by the damping array
  const FIELD_3D& dampingArray = data.get_dampingArray();
  FIELD_3D dequantized_F(uRes, vRes, wRes);
  CastIntFieldToDouble(intBlock, dequantized_F);
  dequantized_F *= (1.0 / s);
  dequantized_F *= dampingArray;


  return dequantized_F;    
}




////////////////////////////////////////////////////////
// given a zigzagged integer buffer, write it to a binary
// file via run-length encoding 
////////////////////////////////////////////////////////
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

    vector<int> dataList;            // a C++ vector container for our data (int16s)
    int data = 0;
    int runLength = 0;
    // int encodedLength = 0;             // variable used to keep track of how long our code is for the decoder

    // assuming 8 x 8 x 8 blocks
    int length = 8 * 8 * 8;
    for (int i = 0; i < length; i++) {
      data = zigzaggedArray[i];
      dataList.push_back(data);
      // encodedLength++;

      runLength = 1;
      while ( (i + 1 < length) && (zigzaggedArray[i] == zigzaggedArray[i + 1]) ) {
            // i + 1 < length ensures that i + 1 doesn't go out of bounds for zigzaggedArray[]
            runLength++;
            i++;
      }
      if (runLength > 1) {
        // use a single repeated value as an 'escape' to indicate a run
        dataList.push_back(data);
        
        // encodedLength++;

        // push the runLength to the data vector
        dataList.push_back(runLength);
        // encodedLength++;
      }
    }
    int encodedLength = dataList.size();

    /*
    if (encodedLength > 8 * 8 * 8) {
      cout << "Encoded length is pretty big: " << encodedLength << endl;
    }
    */

    blockLengths[blockNumber] = encodedLength;

    {
    TIMER writeTimer("fwrite timer");

    fwrite(&(dataList[0]), sizeof(int), encodedLength, pFile);
    // this write assumes that C++ vectors are stored in contiguous memory!
    
    }

    fclose(pFile);
    return;
  }
}

////////////////////////////////////////////////////////
// reads from a binary file into a buffer, and sets
// important initializations inside decompression data
////////////////////////////////////////////////////////
void ReadBinaryFileToMemory(const char* filename, int*& allData, DECOMPRESSION_DATA& decompression_data) {
  TIMER functionTimer(__FUNCTION__);
  // allData is passed by reference and will be modified, as will decompression_data

  FILE* pFile;

  pFile = fopen(filename, "rb");
  if (pFile == NULL) {
    perror("Error opening file.");
    exit(EXIT_FAILURE);
  }

  else {

    // read in q, power, and nBits
    double q, power;
    fread(&q, 1, sizeof(double), pFile);
    fread(&power, 1, sizeof(double), pFile);
    int nBits;
    fread(&nBits, 1, sizeof(int), pFile);

    // set the decompression data
    decompression_data.set_q(q);
    decompression_data.set_power(power);
    decompression_data.set_nBits(nBits);
    // build the damping array using the previous values
    decompression_data.set_dampingArray();

    // this can be called at any time, so might as well do it now
    decompression_data.set_zigzagArray();

    // same for this. -1 for the inverse transform
    decompression_data.dct_setup(-1);

    int xRes, yRes, zRes;
    // read in the dims from the binary file
    fread(&xRes, 1, sizeof(int), pFile);
    fread(&yRes, 1, sizeof(int), pFile);
    fread(&zRes, 1, sizeof(int), pFile);
    VEC3I dims(xRes, yRes, zRes);
    // set the decompression data accordingly
    decompression_data.set_dims(dims);

    int numCols, numBlocks;
    // read in numCols and numBlocks
    fread(&numCols, 1, sizeof(int), pFile);
    fread(&numBlocks, 1, sizeof(int), pFile);
    // set the decompression data accordingly
    decompression_data.set_numCols(numCols);
    decompression_data.set_numBlocks(numBlocks);
    
    // read in the sListMatrix and set the data
    int totalSize = numBlocks * numCols;
    double* double_dummy = (double*) malloc(totalSize * sizeof(double));
    fread(double_dummy, totalSize, sizeof(double), pFile);
    VECTOR flattened_s = CastToVector(double_dummy, totalSize);
    free(double_dummy);
    MATRIX sMatrix(flattened_s, numBlocks, numCols);
    decompression_data.set_sListMatrix(sMatrix);

    // do the same for the blockLengthsMatrix
    int* int_dummy = (int*) malloc(totalSize * sizeof(int));
    fread(int_dummy, totalSize, sizeof(int), pFile);
    VECTOR flattened_lengths = CastToVector(int_dummy, totalSize);
    free(int_dummy);
    MATRIX blockLengthsMatrix(flattened_lengths, numBlocks, numCols);
    // store the total length to be able to read in the full compressed data later
    int totalLength = blockLengthsMatrix.sum();
    decompression_data.set_blockLengthsMatrix(blockLengthsMatrix);

    // do the same for the blockIndicesMatrix
    int* int_dummy2 = (int*) malloc(totalSize * sizeof(int));
    fread(int_dummy2, totalSize, sizeof(int), pFile);
    VECTOR flattened_indices = CastToVector(int_dummy2, totalSize);
    free(int_dummy2);
    MATRIX blockIndicesMatrix(flattened_indices, numBlocks, numCols);
    decompression_data.set_blockIndicesMatrix(blockIndicesMatrix);

    // finally, read in the full compressed data
    allData = (int*) malloc(totalLength * sizeof(int));
    if (allData == NULL) {
      perror("Malloc failed to allocate allData!");
      exit(EXIT_FAILURE);
    }
    fread(allData, totalLength, sizeof(int), pFile);
    }
  return;
}


////////////////////////////////////////////////////////
// decode a run-length encoded binary file and return
// a vector<int> type. fast version!
////////////////////////////////////////////////////////
void RunLengthDecodeBinaryFast(const int* allData, int blockNumber, int col, const MATRIX& blockLengthsMatrix, const MATRIX& blockIndicesMatrix, vector<int>& parsedData) {

  TIMER functionTimer(__FUNCTION__);
  // although blockLengths and blockIndices are passed by reference,
  // they will not be modified.
    
    // what we will be returning
    
    int blockSize = blockLengthsMatrix(blockNumber, col);
    assert(blockSize >= 0 && blockSize <= 3 * 8 * 8 * 8);

    int blockIndex = blockIndicesMatrix(blockNumber, col);
    
    int i = 0;
    int runLength = 1;
    auto itr = parsedData.begin();

    while (i < blockSize) {
      *itr = allData[blockIndex + i];
           // write the value once
      if ( (i + 1 < blockSize) && allData[blockIndex + i] == allData[blockIndex + i + 1]) {      // if we read an 'escape' value, it indicates a run.
        i += 2;                                     // advance past the escape value to the run length value.
        runLength = allData[blockIndex + i];
        
        assert(runLength > 1 && runLength <= 512);

        std::fill(itr + 1, itr + 1 + runLength - 1, allData[blockIndex + i - 2]);
        itr += (runLength - 1);

      }

      i++;
      ++itr;
    }

  } 

////////////////////////////////////////////////////////
// decode a run-length encoded binary file and return
// a vector<int> type
////////////////////////////////////////////////////////
vector<int> RunLengthDecodeBinary(const int* allData, int blockNumber, VECTOR& blockLengths, VECTOR& blockIndices) {

  TIMER functionTimer(__FUNCTION__);
  // although blockLengths and blockIndices are passed by reference,
  // they will not be modified.
    
    // what we will be returning
    vector<int> parsedData(512);                              
    
    int blockSize = blockLengths[blockNumber];
    assert(blockSize >= 0 && blockSize <= 3 * 8 * 8 * 8);

    /*
    if (blockSize > 3 * 8 * 8 * 8) {
      cout << "bogus block size read in: aborting!" << endl;
      cout << "block size was thought to be: " << blockSize << endl;
      exit(1);
    }
    */

    int blockIndex = blockIndices[blockNumber];
    
    /* 
    int* blockData = (int*) malloc(blockSize * sizeof(int));

    if (blockData == NULL) {
      perror("Malloc failed to allocate blockData!");
      exit(EXIT_FAILURE);
    }
     
    for (int i = 0; i < blockSize; i++) {
      blockData[i] = allData[blockIndex + i];
    }
    */
    
    int i = 0;
    int runLength = 1;
    auto itr = parsedData.begin();

    while (i < blockSize) {
      *itr = allData[blockIndex + i];
      // parsedData.push_back(allData[blockIndex + i]);          // write the value once
      if ( (i + 1 < blockSize) && allData[blockIndex + i] == allData[blockIndex + i + 1]) {      // if we read an 'escape' value, it indicates a run.
        i += 2;                                     // advance past the escape value to the run length value.
        runLength = allData[blockIndex + i];
        
        assert(runLength > 1 && runLength <= 512);

        /*
        if (runLength <= 1 || runLength > 512) {
          cout << "Parsing error: run length was read in as garbage. Aborting." << endl;
          cout << "Run length was thought to be: " << runLength << endl;
          cout << "Previous two values were: " << blockData[i-2] << ", " << blockData[i-1] << endl;
          cout << "Next value is: " << blockData[i+1] << endl;
          cout << "Index is currently at: " << i << " of " << blockSize << endl;
          exit(1);
        }
        */
        
        /*
        for (int j = 0; j < runLength - 1; j++) {  // write the original value (index i - 2) repeatedly for runLength - 1 times,
          parsedData.push_back(allData[blockIndex + i - 2]);  // since we already wrote it once
        }
        */

        std::fill(itr + 1, itr + 1 + runLength - 1, allData[blockIndex + i - 2]);
        itr += (runLength - 1);

      }

      i++;
      ++itr;
    }

    // free(blockData);

    // ensure that the parse got the whole block

    /*
    assert( parsedData.size() == 8 * 8 * 8 );

    if ( parsedData.size() != 8 * 8 * 8 ) {
      cout << "Failed to get 512 entries from the block...aborting!" << endl;
      cout << "Got " << parsedData.size() << " entries instead." << endl;
      exit(1);
    }
    */

    return parsedData;
  } 

////////////////////////////////////////////////////////
// deletes a file if it already exists
////////////////////////////////////////////////////////
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
      cout << filename << " does not yet exist; safely opening in append mode." << endl;
    }
    return;
  }


////////////////////////////////////////////////////////
// takes an input FIELD_3D, compresses it according
// to the general scheme, and writes it to a binary file 
////////////////////////////////////////////////////////
void CompressAndWriteFieldSmart(const char* filename, const FIELD_3D& F, COMPRESSION_DATA& compression_data) {
  TIMER functionTimer(__FUNCTION__);
  
    int numBlocks = compression_data.get_numBlocks();

    // why must this be declared as a reference?
    // something might be wrong with the copy constructor or the
    // = operator
    const INTEGER_FIELD_3D& zigzagArray = compression_data.get_zigzagArray();
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
      // performs run-length encoding. updates blockLengths. since
      // it opens 'filename' in append mode, it can be called in a chain
      RunLengthEncodeBinary(filename, i, zigzagArray_i, blockLengths);  
    }
    
    // update the compression data
    compression_data.set_blockLengths(blockLengths);
    VECTOR blockIndices = ModifiedCumSum(blockLengths);
    // the indices are computed by taking the modified cum sum
    compression_data.set_blockIndices(blockIndices);
    free(zigzagArray_i);
    return;
  }
  
////////////////////////////////////////////////////////
// given a row number and the dimensions, computes
// which block number we need for the decoder. populates
// blockIndex with the corresponding value as well.
////////////////////////////////////////////////////////
int ComputeBlockNumber(int row, VEC3I dims, int& blockIndex) {
  TIMER functionTimer(__FUNCTION__);
    int xRes = dims[0];
    int yRes = dims[1];
    int zRes = dims[2];
    
    assert( row >= 0 && row < 3 * xRes * yRes * zRes);

    // evil integer division!
    int index = row / 3;
    int z = index / (xRes * yRes);         // index = (xRes * yRes) * z + remainder1
    int rem1 = index - (xRes * yRes * z);
    int y = rem1 / xRes;                   // rem1  = xRes * y          + remainder2 
    int rem2 = rem1 - xRes * y;
    int x = rem2;

    int u = x % 8;
    int v = y % 8;
    int w = z % 8;
    blockIndex = u + 8 * v + 8 * 8 * w;
    
    // sanity check!
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

  
////////////////////////////////////////////////////////
// given a (row, col), decode it from the lossy matrix
////////////////////////////////////////////////////////
double DecodeFromRowColFast(int row, int col, MATRIX_COMPRESSION_DATA& data) { 
     
  TIMER functionTimer(__FUNCTION__);

  DECOMPRESSION_DATA dataX = data.get_decompression_dataX();
  // using X is arbitrary---it will be the same for all three
  VEC3I dims = dataX.get_dims();
  const INTEGER_FIELD_3D& zigzagArray = dataX.get_zigzagArray();
  
  int* allDataX = data.get_dataX();
  int* allDataY = data.get_dataY();
  int* allDataZ = data.get_dataZ();
  const DECOMPRESSION_DATA& dataY = data.get_decompression_dataY();
  const DECOMPRESSION_DATA& dataZ = data.get_decompression_dataZ();

  vector<int> decoded_runLength(512);
  VECTOR decoded_runLengthVector(512);
  INTEGER_FIELD_3D unzigzagged(8, 8, 8);
  FIELD_3D decoded_block(8, 8, 8);
 

  // dummy initialization
  int blockIndex = 0;
  // fill blockIndex and compute the block number
  int blockNumber = ComputeBlockNumber(row, dims, blockIndex);
  if (row % 3 == 0) { // X coordinate
    const MATRIX& blockLengthsMatrix = dataX.get_blockLengthsMatrix();
    const MATRIX& blockIndicesMatrix = dataX.get_blockIndicesMatrix();
        
    
    RunLengthDecodeBinaryFast(allDataX, blockNumber, col, blockLengthsMatrix, blockIndicesMatrix, decoded_runLength); 

    CastIntToVectorFast(decoded_runLength, decoded_runLengthVector);
    ZigzagUnflattenSmart(decoded_runLengthVector, zigzagArray, unzigzagged);
    DecodeBlockFast(unzigzagged, blockNumber, col, dataX, decoded_block); 
    double result = decoded_block[blockIndex];
    return result;

  }

  else if (row % 3 == 1) { // Y coordinate
  
    const MATRIX& blockLengthsMatrix = dataY.get_blockLengthsMatrix();
    const MATRIX& blockIndicesMatrix = dataY.get_blockIndicesMatrix();

    RunLengthDecodeBinaryFast(allDataY, blockNumber, col, blockLengthsMatrix, blockIndicesMatrix, decoded_runLength); 

    CastIntToVectorFast(decoded_runLength, decoded_runLengthVector);
    ZigzagUnflattenSmart(decoded_runLengthVector, zigzagArray, unzigzagged);
    DecodeBlockFast(unzigzagged, blockNumber, col, dataY, decoded_block); 
    double result = decoded_block[blockIndex];
    return result;
  
  }

  else { // Z coordinate
   
    const MATRIX& blockLengthsMatrix = dataZ.get_blockLengthsMatrix();
    const MATRIX& blockIndicesMatrix = dataZ.get_blockIndicesMatrix();


    RunLengthDecodeBinaryFast(allDataZ, blockNumber, col, blockLengthsMatrix, blockIndicesMatrix, decoded_runLength); 

    CastIntToVectorFast(decoded_runLength, decoded_runLengthVector);
    ZigzagUnflattenSmart(decoded_runLengthVector, zigzagArray, unzigzagged);
    DecodeBlockFast(unzigzagged, blockNumber, col, dataZ, decoded_block);
    double result = decoded_block[blockIndex];
    return result;
  }
} 

////////////////////////////////////////////////////////
// given a (row, col), decode it from the lossy matrix
////////////////////////////////////////////////////////
double DecodeFromRowCol(int row, int col, MATRIX_COMPRESSION_DATA& data) { 
     
  TIMER functionTimer(__FUNCTION__);

  DECOMPRESSION_DATA dataX = data.get_decompression_dataX();
  // using X is arbitrary---it will be the same for all three
  VEC3I dims = dataX.get_dims();
  const INTEGER_FIELD_3D& zigzagArray = dataX.get_zigzagArray();
  
  int* allDataX = data.get_dataX();
  int* allDataY = data.get_dataY();
  int* allDataZ = data.get_dataZ();
  DECOMPRESSION_DATA dataY = data.get_decompression_dataY();
  DECOMPRESSION_DATA dataZ = data.get_decompression_dataZ();

  vector<int> decoded_runLength;

  // dummy initialization
  int blockIndex = 0;
  // fill blockIndex and compute the block number
  int blockNumber = ComputeBlockNumber(row, dims, blockIndex);
  if (row % 3 == 0) { // X coordinate
    const MATRIX& blockLengthsMatrix = dataX.get_blockLengthsMatrix();
    const MATRIX& blockIndicesMatrix = dataX.get_blockIndicesMatrix();
    const MATRIX& sListMatrix = dataX.get_sListMatrix();
        
    
    VECTOR blockLengths = blockLengthsMatrix.getColumn(col);
    VECTOR blockIndices = blockIndicesMatrix.getColumn(col);
    VECTOR sList = sListMatrix.getColumn(col);
    decoded_runLength = RunLengthDecodeBinary(allDataX, blockNumber, blockLengths, blockIndices); 

    VECTOR decoded_runLengthVector = CastIntToVector(decoded_runLength);
    // INTEGER_FIELD_3D unzigzagged = ZigzagUnflatten(decoded_runLengthVector);
    INTEGER_FIELD_3D unzigzagged(8, 8, 8);
    ZigzagUnflattenSmart(decoded_runLengthVector, zigzagArray, unzigzagged);
    FIELD_3D decoded_block = DecodeBlock(unzigzagged, blockNumber, col, dataX); 
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
    // INTEGER_FIELD_3D unzigzagged = ZigzagUnflatten(decoded_runLengthVector);
    INTEGER_FIELD_3D unzigzagged(8, 8, 8);
    ZigzagUnflattenSmart(decoded_runLengthVector, zigzagArray, unzigzagged);
    FIELD_3D decoded_block = DecodeBlock(unzigzagged, blockNumber, col, dataY); 
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
    // INTEGER_FIELD_3D unzigzagged = ZigzagUnflatten(decoded_runLengthVector);
    INTEGER_FIELD_3D unzigzagged(8, 8, 8);
    ZigzagUnflattenSmart(decoded_runLengthVector, zigzagArray, unzigzagged);
    FIELD_3D decoded_block = DecodeBlock(unzigzagged, blockNumber, col, dataZ);
    double result = decoded_block[blockIndex];
    return result;
  }
} 

   
   
////////////////////////////////////////////////////////
// given a start row and numRows (typically 3), computes
// the submatrix given the compression data
////////////////////////////////////////////////////////
void GetSubmatrixFast(int startRow, int numRows, MATRIX_COMPRESSION_DATA& data, MatrixXd& matrixToFill) {
     
    
  TIMER functionTimer(__FUNCTION__);
  
  // only leave this commented out if you don't mind skipping an assertion! 
  /*
  DECOMPRESSION_DATA decompression_dataX = data.get_decompression_dataX();
  int numCols = decompression_dataX.get_numCols();
  assert( matrixToFill.rows() == numRows && matrixToFill.cols() == numCols );
  */

  for (int i = 0; i < numRows; i++) {
    GetRowFast(startRow + i, i, data, matrixToFill);
  }    
}

////////////////////////////////////////////////////////
// given a start row and numRows (typically 3), computes
// the submatrix given the compression data
////////////////////////////////////////////////////////
MatrixXd GetSubmatrix(int startRow, int numRows, MATRIX_COMPRESSION_DATA& data) {
     
    
  TIMER functionTimer(__FUNCTION__);

  const DECOMPRESSION_DATA& decompression_dataX = data.get_decompression_dataX();
  int numCols = decompression_dataX.get_numCols();
  MatrixXd result(numRows, numCols);
     
  for (int i = 0; i < numRows; i++) {
    VectorXd row_i = GetRow(startRow + i, data);
    result.row(i) = row_i;
  }    
  return result;
}


VectorXd GetRow(int row, MATRIX_COMPRESSION_DATA& data) {

  TIMER functionTimer(__FUNCTION__);
  
  const DECOMPRESSION_DATA& decompression_dataX = data.get_decompression_dataX();
  int numCols = decompression_dataX.get_numCols();   
  VectorXd result(numCols);
  const VEC3I& dims = decompression_dataX.get_dims();
  const INTEGER_FIELD_3D& zigzagArray = decompression_dataX.get_zigzagArray();
  int blockIndex = 0;
  // fill blockIndex
  int blockNumber = ComputeBlockNumber(row, dims, blockIndex);
  int cachedBlockNumber = data.get_cachedBlockNumber();
  // int decodeCounter = data.get_decodeCounter();

  if (blockNumber == cachedBlockNumber) { // if we've already decoded this block
    TIMER cacheTimer("cache block");

    // cout << "Used cache!" << endl;

    if (row % 3 == 0) { // X coordinate
      TIMER xTimer("x coordinate cached");
      
      // load the previously decoded data
      vector<FIELD_3D>& cachedBlocksX = data.get_cachedBlocksX();
      for (int col = 0; col < numCols; col++) {
        FIELD_3D block = cachedBlocksX[col];
        // note that square brackets are necessary to access a field data member by 
        // linear index!!
        result[col] = block[blockIndex];
      }   
      return result;
    }
    else if (row % 3 == 1) { // Y coordinate
      vector<FIELD_3D>& cachedBlocksY = data.get_cachedBlocksY();
      for (int col = 0; col < numCols; col++) {
        
        FIELD_3D block = cachedBlocksY[col];
        result[col] = block[blockIndex];
      }
      return result;
    }
    else { // Z coordinate
      vector<FIELD_3D>& cachedBlocksZ = data.get_cachedBlocksZ();
      for (int col = 0; col < numCols; col++) {
        FIELD_3D block = cachedBlocksZ[col];
        result[col] = block[blockIndex];
      }
      return result;
    }
  }

  else { // no cache; have to compute it from scratch
    // cout << "Didn't use cache!" << endl;  
    TIMER uncachedTimer("uncached block");

    if (row % 3 == 0) { // X coordinate
      TIMER x_uncachedTimer("x coordinate uncached");

      int* allDataX = data.get_dataX();
      const MATRIX& blockLengthsMatrix = decompression_dataX.get_blockLengthsMatrix();
      const MATRIX& blockIndicesMatrix = decompression_dataX.get_blockIndicesMatrix();
      const MATRIX& sListMatrix = decompression_dataX.get_sListMatrix();
    
      for (int col = 0; col < numCols; col++) {
        TIMER columnTimer("uncached column loop");  
           
        VECTOR blockLengths = blockLengthsMatrix.getColumn(col);
        VECTOR blockIndices = blockIndicesMatrix.getColumn(col);
        VECTOR sList = sListMatrix.getColumn(col);
        vector<int> decoded_runLength = RunLengthDecodeBinary(allDataX, blockNumber, blockLengths, blockIndices); 

        VECTOR decoded_runLengthVector = CastIntToVector(decoded_runLength);
        INTEGER_FIELD_3D unzigzagged(8, 8, 8);
        ZigzagUnflattenSmart(decoded_runLengthVector, zigzagArray, unzigzagged);

        FIELD_3D decoded_block = DecodeBlock(unzigzagged, blockNumber, col, decompression_dataX); 
        
        // set the cached block
        vector<FIELD_3D>& cachedBlocksX = data.get_cachedBlocksX();

        // update the cache
        cachedBlocksX[col] = decoded_block;
        data.set_cachedBlocksX(cachedBlocksX);
       
        result[col] = decoded_block[blockIndex];
      }

      return result;
    }

    else if (row % 3 == 1) { // Y coordinate
      
      int* allDataY = data.get_dataY();
      const DECOMPRESSION_DATA& decompression_dataY = data.get_decompression_dataY();
      const MATRIX& blockLengthsMatrix = decompression_dataY.get_blockLengthsMatrix();
      const MATRIX& blockIndicesMatrix = decompression_dataY.get_blockIndicesMatrix();
      const MATRIX& sListMatrix = decompression_dataY.get_sListMatrix();

      for (int col = 0; col < numCols; col++) {
           
        VECTOR blockLengths = blockLengthsMatrix.getColumn(col);
        VECTOR blockIndices = blockIndicesMatrix.getColumn(col);
        VECTOR sList = sListMatrix.getColumn(col);
        vector<int> decoded_runLength = RunLengthDecodeBinary(allDataY, blockNumber, blockLengths, blockIndices); 

        VECTOR decoded_runLengthVector = CastIntToVector(decoded_runLength);
        INTEGER_FIELD_3D unzigzagged(8, 8, 8);
        ZigzagUnflattenSmart(decoded_runLengthVector, zigzagArray, unzigzagged);
        FIELD_3D decoded_block = DecodeBlock(unzigzagged, blockNumber, col, decompression_dataY); 
        // set the cached block

        vector<FIELD_3D>& cachedBlocksY = data.get_cachedBlocksY();
        // update the cache
        cachedBlocksY[col] = decoded_block;
        data.set_cachedBlocksY(cachedBlocksY);

        result[col] = decoded_block[blockIndex];
      }

      return result;
     
    }

    else { // Z coordinate

      int* allDataZ = data.get_dataZ();
      DECOMPRESSION_DATA decompression_dataZ = data.get_decompression_dataZ();
      const MATRIX& blockLengthsMatrix = decompression_dataZ.get_blockLengthsMatrix();
      const MATRIX& blockIndicesMatrix = decompression_dataZ.get_blockIndicesMatrix();
      const MATRIX& sListMatrix = decompression_dataZ.get_sListMatrix();

      for (int col = 0; col < numCols; col++) {
           
        VECTOR blockLengths = blockLengthsMatrix.getColumn(col);
        VECTOR blockIndices = blockIndicesMatrix.getColumn(col);
        VECTOR sList = sListMatrix.getColumn(col);
        vector<int> decoded_runLength = RunLengthDecodeBinary(allDataZ, blockNumber, blockLengths, blockIndices); 

        VECTOR decoded_runLengthVector = CastIntToVector(decoded_runLength);
        // INTEGER_FIELD_3D unzigzagged = ZigzagUnflatten(decoded_runLengthVector);
        INTEGER_FIELD_3D unzigzagged(8, 8, 8);
        ZigzagUnflattenSmart(decoded_runLengthVector, zigzagArray, unzigzagged);
        FIELD_3D decoded_block = DecodeBlock(unzigzagged, blockNumber, col, decompression_dataZ); 
        // set the cached block

        vector<FIELD_3D>& cachedBlocksZ = data.get_cachedBlocksZ();
        
        // update the cache
        cachedBlocksZ[col] = decoded_block;
        data.set_cachedBlocksZ(cachedBlocksZ);

        result[col] = decoded_block[blockIndex];
      }

      // set the cached block number
      data.set_cachedBlockNumber(blockNumber);

      return result;
    }
  }
}

 
void GetRowFast(int row, int matrixRow, MATRIX_COMPRESSION_DATA& data, MatrixXd& matrixToFill) {

  TIMER functionTimer(__FUNCTION__);
  
  const DECOMPRESSION_DATA& decompression_dataX = data.get_decompression_dataX();
  int numCols = decompression_dataX.get_numCols();   
  assert( matrixToFill.cols() == numCols && matrixToFill.rows() == 3 );

  // VectorXd result(numCols);
  const VEC3I& dims = decompression_dataX.get_dims();
  const INTEGER_FIELD_3D& zigzagArray = decompression_dataX.get_zigzagArray();
  int blockIndex = 0;
  // fill blockIndex
  int blockNumber = ComputeBlockNumber(row, dims, blockIndex);
  int cachedBlockNumber = data.get_cachedBlockNumber();

  if (blockNumber == cachedBlockNumber) { // if we've already decoded this block
    TIMER cacheTimer("cache block");

    // cout << "Used cache!" << endl;

    if (row % 3 == 0) { // X coordinate
      TIMER xTimer("x coordinate cached");
      
      // load the previously decoded data
      vector<FIELD_3D>& cachedBlocksX = data.get_cachedBlocksX();
      for (int col = 0; col < numCols; col++) {
        // FIELD_3D block = cachedBlocksX[col];
        // note that square brackets are necessary to access a field data member by 
        // linear index!!
        // result[col] = block[blockIndex];
        matrixToFill(matrixRow, col) = cachedBlocksX[col][blockIndex];
      }   
    }
    else if (row % 3 == 1) { // Y coordinate
      vector<FIELD_3D>& cachedBlocksY = data.get_cachedBlocksY();
      for (int col = 0; col < numCols; col++) {
        
        // FIELD_3D block = cachedBlocksY[col];
        // result[col] = block[blockIndex];
        matrixToFill(matrixRow, col) = cachedBlocksY[col][blockIndex];
      }
    }
    else { // Z coordinate
      vector<FIELD_3D>& cachedBlocksZ = data.get_cachedBlocksZ();
      for (int col = 0; col < numCols; col++) {
        // FIELD_3D block = cachedBlocksZ[col];
        // result[col] = block[blockIndex];
        matrixToFill(matrixRow, col) = cachedBlocksZ[col][blockIndex];
      } 
    }
  }

  else { // no cache; have to compute it from scratch
    // cout << "Didn't use cache!" << endl;  
    TIMER uncachedTimer("uncached block");
    INTEGER_FIELD_3D unzigzagged(8, 8, 8);
    FIELD_3D decoded_block(8, 8, 8);
    vector<int> decoded_runLength(512);
    VECTOR decoded_runLengthVector(512);

    if (row % 3 == 0) { // X coordinate
      TIMER x_uncachedTimer("x coordinate uncached");

      int* allDataX = data.get_dataX();
      const MATRIX& blockLengthsMatrix = decompression_dataX.get_blockLengthsMatrix();
      const MATRIX& blockIndicesMatrix = decompression_dataX.get_blockIndicesMatrix();

      for (int col = 0; col < numCols; col++) {
        TIMER columnTimer("uncached column loop");  
           
        RunLengthDecodeBinaryFast(allDataX, blockNumber, col, blockLengthsMatrix, blockIndicesMatrix, decoded_runLength); 

        CastIntToVectorFast(decoded_runLength, decoded_runLengthVector);
        ZigzagUnflattenSmart(decoded_runLengthVector, zigzagArray, unzigzagged);

        DecodeBlockFast(unzigzagged, blockNumber, col, decompression_dataX, decoded_block); 
        
        // set the cached block
        vector<FIELD_3D>& cachedBlocksX = data.get_cachedBlocksX();

        // update the cache
        cachedBlocksX[col] = decoded_block;
       
        matrixToFill(matrixRow, col) = cachedBlocksX[col][blockIndex];
      } 
    }

    else if (row % 3 == 1) { // Y coordinate
      
      TIMER x_uncachedTimer("y coordinate uncached");

      int* allDataY = data.get_dataY();
      const DECOMPRESSION_DATA& decompression_dataY = data.get_decompression_dataY();
      const MATRIX& blockLengthsMatrix = decompression_dataY.get_blockLengthsMatrix();
      const MATRIX& blockIndicesMatrix = decompression_dataY.get_blockIndicesMatrix();

      for (int col = 0; col < numCols; col++) {
           
        RunLengthDecodeBinaryFast(allDataY, blockNumber, col, blockLengthsMatrix, blockIndicesMatrix, decoded_runLength); 

        CastIntToVectorFast(decoded_runLength, decoded_runLengthVector);
        ZigzagUnflattenSmart(decoded_runLengthVector, zigzagArray, unzigzagged);
        DecodeBlockFast(unzigzagged, blockNumber, col, decompression_dataY, decoded_block); 
        // set the cached block

        vector<FIELD_3D>& cachedBlocksY = data.get_cachedBlocksY();
        // update the cache
        cachedBlocksY[col] = decoded_block;

        matrixToFill(matrixRow, col) = cachedBlocksY[col][blockIndex];
      }

     
    }

    else { // Z coordinate

      TIMER x_uncachedTimer("z coordinate uncached");

      int* allDataZ = data.get_dataZ();
      const DECOMPRESSION_DATA& decompression_dataZ = data.get_decompression_dataZ();
      const MATRIX& blockLengthsMatrix = decompression_dataZ.get_blockLengthsMatrix();
      const MATRIX& blockIndicesMatrix = decompression_dataZ.get_blockIndicesMatrix();

      for (int col = 0; col < numCols; col++) {
           
      RunLengthDecodeBinaryFast(allDataZ, blockNumber, col, blockLengthsMatrix, blockIndicesMatrix, decoded_runLength); 

        CastIntToVectorFast(decoded_runLength, decoded_runLengthVector);
        ZigzagUnflattenSmart(decoded_runLengthVector, zigzagArray, unzigzagged);
        DecodeBlockFast(unzigzagged, blockNumber, col, decompression_dataZ, decoded_block); 
        // set the cached block

        vector<FIELD_3D>& cachedBlocksZ = data.get_cachedBlocksZ();
        
        // update the cache
        cachedBlocksZ[col] = decoded_block;

        matrixToFill(matrixRow, col) = cachedBlocksZ[col][blockIndex];
      }

      // set the cached block number
      data.set_cachedBlockNumber(blockNumber);

    }
  }
}

 

////////////////////////////////////////////////////////
// generates the header information in the binary file
////////////////////////////////////////////////////////
void WriteMetaData(const char* filename, const COMPRESSION_DATA& compression_data, 
    const MATRIX& sListMatrix, const MATRIX& blockLengthsMatrix, const MATRIX& blockIndicesMatrix) {

  TIMER functionTimer(__FUNCTION__);
    FILE* pFile;
    pFile = fopen(filename, "wb");
    if (pFile == NULL) {
      perror ("Error opening file.");
    }
    else {
      
      // write q, power, and nBits to the binary file
      double q = compression_data.get_q();
      fwrite(&q, sizeof(double), 1, pFile);
      double power = compression_data.get_power();
      fwrite(&power, sizeof(double), 1, pFile);
      int nBits = compression_data.get_nBits();
      fwrite(&nBits, sizeof(int), 1, pFile);

      // write dims, numCols, and numBlocks
      VEC3I dims = compression_data.get_dims();
      int xRes = dims[0];
      int yRes = dims[1];
      int zRes = dims[2];
      fwrite(&xRes, sizeof(int), 1, pFile);
      fwrite(&yRes, sizeof(int), 1, pFile);
      fwrite(&zRes, sizeof(int), 1, pFile);
      int numCols = compression_data.get_numCols();
      int numBlocks = compression_data.get_numBlocks();
      fwrite(&numCols, sizeof(int), 1, pFile);
      fwrite(&numBlocks, sizeof(int), 1, pFile);
            
       
      VECTOR flattened_s = sListMatrix.flattenedColumn();
      int blocksXcols = flattened_s.size();

      assert( blocksXcols == numBlocks * numCols );

      double* sData = (double*) malloc(sizeof(double) * blocksXcols);
      // fill sData and write it
      sData = CastToDouble(flattened_s, sData);
      fwrite(sData, sizeof(double), blocksXcols, pFile);

      VECTOR flattened_lengths = blockLengthsMatrix.flattenedColumn();
      assert(flattened_lengths.size() == blocksXcols);

      int* lengthsData = (int*) malloc(sizeof(int) * blocksXcols);
      // fill lengthsData and write it
      lengthsData = CastToInt(flattened_lengths, lengthsData);
      fwrite(lengthsData, sizeof(int), blocksXcols, pFile);
      
      VECTOR flattened_indices = blockIndicesMatrix.flattenedColumn();
      assert(flattened_indices.size() == blocksXcols);

      // fill indicesData and write it
      int* indicesData = (int*) malloc(sizeof(int) * blocksXcols);
      indicesData = CastToInt(flattened_indices, indicesData);
      fwrite(indicesData, sizeof(int), blocksXcols, pFile);

      fclose(pFile);
      free(sData);
      free(lengthsData);
      free(indicesData);
    }
  }


////////////////////////////////////////////////////////
// concatenate two binary files and put them into
// a new binary file
////////////////////////////////////////////////////////
void PrefixBinary(string prefix, string filename, string newFile) {
  TIMER functionTimer(__FUNCTION__);
  string command = "cat " + prefix + ' ' + filename + "> " + newFile;
  const char* command_c = command.c_str();
  system(command_c);
}

////////////////////////////////////////////////////////
// destroy the no longer needed metadata binary file
////////////////////////////////////////////////////////
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

////////////////////////////////////////////////////////
// compress one of the scalar field components of a matrix
// (which represents a vector field) and write it to
// a binary file 
////////////////////////////////////////////////////////
void CompressAndWriteMatrixComponent(const char* filename, const MatrixXd& U, int component, COMPRESSION_DATA& data) {
  TIMER functionTimer(__FUNCTION__);

  assert( component >= 0 && component < 3 );

  string final_string(filename);
  if (component == 0) {
    final_string += 'X';
    DeleteIfExists(final_string.c_str());
  }
  else if (component == 1) {
    final_string += 'Y';
    DeleteIfExists(final_string.c_str());
  }
  else { // component == 2
    final_string += 'Z';
    DeleteIfExists(final_string.c_str());
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
  
  // compute progress for the user
  double percent = 0.0;
  for (int col = 0; col < numCols; col++) {  
    // cout << "Column: " << col << endl;
    VectorXd vXd = U.col(col);
    VECTOR v = EIGEN::convert(vXd);
    VECTOR3_FIELD_3D V(v, xRes, yRes, zRes);
    FIELD_3D F = V.scalarField(component);
    
    CompressAndWriteFieldSmart(filename, F, data);

    // update blockLengths and sList and push them to the appropriate column
    // of their respective matrices
    VECTOR blockLengths = data.get_blockLengths();
    VECTOR sList = data.get_sList();
    blockLengthsMatrix.setColumn(blockLengths, col);
    sListMatrix.setColumn(sList, col);

    // compute progress for the user
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
  WriteMetaData(metafile, data, sListMatrix, blockLengthsMatrix, blockIndicesMatrix);
  // appends the metadata as a header to the main binary file and pipes them into final_string
  PrefixBinary(metafile, filename, final_string);
  // removes the now-redundant metadata and main binary files
  CleanUpPrefix(metafile, filename);
}

   
void DecodeScalarFieldFast(const DECOMPRESSION_DATA& decompression_data, int* const& allData, int col, FIELD_3D& fieldToFill) {
  TIMER functionTimer(__FUNCTION__);

  // get the dims from data and construct a field of the appropriate size
  VEC3I dims = decompression_data.get_dims();
  int xRes = dims[0];
  int xResOriginal = xRes;
  int yRes = dims[1];
  int yResOriginal = yRes;
  int zRes = dims[2];
  int zResOriginal = zRes;

  int xPadding = 0;
  int yPadding = 0;
  int zPadding = 0;
  // fill in the paddings
  GetPaddings(dims, xPadding, yPadding, zPadding);
  // update the res
  xRes += xPadding;
  yRes += yPadding;
  zRes += zPadding;
  VEC3I dimsUpdated(xRes, yRes, zRes);

  int numBlocks = decompression_data.get_numBlocks();

  const MATRIX& blockLengthsMatrix = decompression_data.get_blockLengthsMatrix();
  const MATRIX& blockIndicesMatrix = decompression_data.get_blockIndicesMatrix();


  const INTEGER_FIELD_3D& zigzagArray = decompression_data.get_zigzagArray();
  INTEGER_FIELD_3D unzigzagged(8, 8, 8);
  vector<int> runLengthDecoded(512);
  VECTOR runLengthDecodedVec(512);

  // container for the encoded blocks
  vector<FIELD_3D> blocks(numBlocks);
  for (int blockNumber = 0; blockNumber < numBlocks; blockNumber++) {
    // decode the run length scheme
    RunLengthDecodeBinaryFast(allData, blockNumber, col, blockLengthsMatrix, blockIndicesMatrix, runLengthDecoded);
    // cast to VECTOR to play nice with ZigzagUnflattenSmart
    CastIntToVectorFast(runLengthDecoded, runLengthDecodedVec);
    // undo the zigzag scan
    ZigzagUnflattenSmart(runLengthDecodedVec, zigzagArray, unzigzagged);
    // undo the scaling from the quantizer and push to the block container
    DecodeBlockDecomp(unzigzagged, blockNumber, col, decompression_data, blocks[blockNumber]);
  }
  
  // perform the IDCT on each block
  // -1 <-- inverse
  DoSmartBlockDCT(blocks, -1);
  
  // reassemble the blocks into one large scalar field
  FIELD_3D padded_result(xRes, yRes, zRes);
  AssimilateBlocks(dimsUpdated, blocks, padded_result);

  // strip the padding
  fieldToFill = padded_result.subfield(0, xResOriginal, 0, yResOriginal, 0, zResOriginal); 
 
}
////////////////////////////////////////////////////////
// reconstructs a lossy original of a scalar field
// which had been written to a binary file and now loaded 
// into a int buffer
////////////////////////////////////////////////////////
FIELD_3D DecodeScalarField(const DECOMPRESSION_DATA& decompression_data, int* const& allData, int col) {
  TIMER functionTimer(__FUNCTION__);

  // get the dims from data and construct a field of the appropriate size
  VEC3I dims = decompression_data.get_dims();
  int xRes = dims[0];
  int xResOriginal = xRes;
  int yRes = dims[1];
  int yResOriginal = yRes;
  int zRes = dims[2];
  int zResOriginal = zRes;

  int xPadding = 0;
  int yPadding = 0;
  int zPadding = 0;
  // fill in the paddings
  GetPaddings(dims, xPadding, yPadding, zPadding);
  // update the res
  xRes += xPadding;
  yRes += yPadding;
  zRes += zPadding;
  VEC3I dimsUpdated(xRes, yRes, zRes);

  int numBlocks = decompression_data.get_numBlocks();

  const MATRIX& blockLengthsMatrix = decompression_data.get_blockLengthsMatrix();
  const MATRIX& blockIndicesMatrix = decompression_data.get_blockIndicesMatrix();

  VECTOR blockLengths = blockLengthsMatrix.getColumn(col);
  VECTOR blockIndices = blockIndicesMatrix.getColumn(col);

  const INTEGER_FIELD_3D& zigzagArray = decompression_data.get_zigzagArray();

  // container for the encoded blocks
  vector<FIELD_3D> blocks(numBlocks);

  for (int blockNumber = 0; blockNumber < numBlocks; blockNumber++) {
    // decode the run length scheme
    vector<int> runLengthDecoded = RunLengthDecodeBinary(allData, blockNumber, blockLengths, blockIndices);
    // cast to VECTOR to play nice with ZigzagUnflattenSmart
    VECTOR runLengthDecodedVec = CastIntToVector(runLengthDecoded);
    // undo the zigzag scan
    // INTEGER_FIELD_3D unzigzagged = ZigzagUnflatten(runLengthDecodedVec);
    INTEGER_FIELD_3D unzigzagged(8, 8, 8);
    ZigzagUnflattenSmart(runLengthDecodedVec, zigzagArray, unzigzagged);
    // undo the scaling from the quantizer and push to the block container
    DecodeBlockDecomp(unzigzagged, blockNumber, col, decompression_data, blocks[blockNumber]);
  }
  
  // perform the IDCT on each block
  // -1 <-- inverse
  DoSmartBlockDCT(blocks, -1);
  
  // reassemble the blocks into one large scalar field
  FIELD_3D padded_result(xRes, yRes, zRes);
  AssimilateBlocks(dimsUpdated, blocks, padded_result);

  // strip the padding
  FIELD_3D result = padded_result.subfield(0, xResOriginal, 0, yResOriginal, 0, zResOriginal); 
 
  return result;
}


////////////////////////////////////////////////////////
// reconstructs a lossy original of a scalar field
// which had been written to a binary file and now loaded 
// into a int buffer. returns it in a vector of its blocks,
// each flattened out into a VectorXd. this is for use
// in projection/unprojection
////////////////////////////////////////////////////////
void DecodeScalarFieldEigenFast(const DECOMPRESSION_DATA& decompression_data, int* const& allData, int col, vector<VectorXd>& toFill) {
  TIMER functionTimer(__FUNCTION__);

  // get the dims from data and construct a field of the appropriate size
  VEC3I dims = decompression_data.get_dims();
  int xRes = dims[0];
  int xResOriginal = xRes;
  int yRes = dims[1];
  int yResOriginal = yRes;
  int zRes = dims[2];
  int zResOriginal = zRes;

  int xPadding = 0;
  int yPadding = 0;
  int zPadding = 0;
  // fill in the paddings
  GetPaddings(dims, xPadding, yPadding, zPadding);
  // update the res
  xRes += xPadding;
  yRes += yPadding;
  zRes += zPadding;
  VEC3I dimsUpdated(xRes, yRes, zRes);

  int numBlocks = decompression_data.get_numBlocks();
  toFill.resize(numBlocks);

  const MATRIX& blockLengthsMatrix = decompression_data.get_blockLengthsMatrix();
  const MATRIX& blockIndicesMatrix = decompression_data.get_blockIndicesMatrix();

  const INTEGER_FIELD_3D& zigzagArray = decompression_data.get_zigzagArray();
  INTEGER_FIELD_3D unzigzagged(8, 8, 8);
  vector<int> runLengthDecoded(512);
  VECTOR runLengthDecodedVec(512);

  // container for the encoded blocks
  vector<FIELD_3D> blocks(numBlocks);

  for (int blockNumber = 0; blockNumber < numBlocks; blockNumber++) {
    // decode the run length scheme
    RunLengthDecodeBinaryFast(allData, blockNumber, col, blockLengthsMatrix, blockIndicesMatrix, runLengthDecoded);
    // cast to VECTOR to play nice with ZigzagUnflattenSmart
    CastIntToVectorFast(runLengthDecoded, runLengthDecodedVec);
    // undo the zigzag scan
    ZigzagUnflattenSmart(runLengthDecodedVec, zigzagArray, unzigzagged);
    // undo the scaling from the quantizer and push to the block container
    DecodeBlockDecomp(unzigzagged, blockNumber, col, decompression_data, blocks[blockNumber]);
  }
  
  // perform the IDCT on each block
  // -1 <-- inverse
  DoSmartBlockDCT(blocks, -1);


  // this part is inefficient---we should implement a DCT that operates on VectorXd so that
  // we don't have to loop through all the blocks again
  for (int blockNumber = 0; blockNumber < numBlocks; blockNumber++) {
    VectorXd blockEigen= blocks[blockNumber].flattenedEigen();
    toFill[blockNumber] = blockEigen;
  } 
} 

////////////////////////////////////////////////////////
// reconstructs a lossy original of a scalar field
// which had been written to a binary file and now loaded 
// into a int buffer. returns it in a vector of its blocks,
// each flattened out into a VectorXd. this is for use
// in projection/unprojection
////////////////////////////////////////////////////////
vector<VectorXd> DecodeScalarFieldEigen(const DECOMPRESSION_DATA& decompression_data, int* const& allData, int col) {
  TIMER functionTimer(__FUNCTION__);

  // get the dims from data and construct a field of the appropriate size
  VEC3I dims = decompression_data.get_dims();
  int xRes = dims[0];
  int xResOriginal = xRes;
  int yRes = dims[1];
  int yResOriginal = yRes;
  int zRes = dims[2];
  int zResOriginal = zRes;

  int xPadding = 0;
  int yPadding = 0;
  int zPadding = 0;
  // fill in the paddings
  GetPaddings(dims, xPadding, yPadding, zPadding);
  // update the res
  xRes += xPadding;
  yRes += yPadding;
  zRes += zPadding;
  VEC3I dimsUpdated(xRes, yRes, zRes);

  int numBlocks = decompression_data.get_numBlocks();

  const MATRIX& blockLengthsMatrix = decompression_data.get_blockLengthsMatrix();
  const MATRIX& blockIndicesMatrix = decompression_data.get_blockIndicesMatrix();

  VECTOR blockLengths = blockLengthsMatrix.getColumn(col);
  VECTOR blockIndices = blockIndicesMatrix.getColumn(col);

  const INTEGER_FIELD_3D& zigzagArray = decompression_data.get_zigzagArray();

  // container for the encoded blocks
  vector<FIELD_3D> blocks(numBlocks);

  for (int blockNumber = 0; blockNumber < numBlocks; blockNumber++) {
    // decode the run length scheme
    vector<int> runLengthDecoded = RunLengthDecodeBinary(allData, blockNumber, blockLengths, blockIndices);
    // cast to VECTOR to play nice with ZigzagUnflattenSmart
    VECTOR runLengthDecodedVec = CastIntToVector(runLengthDecoded);
    // undo the zigzag scan
    // INTEGER_FIELD_3D unzigzagged = ZigzagUnflatten(runLengthDecodedVec);
    INTEGER_FIELD_3D unzigzagged(8, 8, 8);
    ZigzagUnflattenSmart(runLengthDecodedVec, zigzagArray, unzigzagged);
    // undo the scaling from the quantizer and push to the block container
    DecodeBlockDecomp(unzigzagged, blockNumber, col, decompression_data, blocks[blockNumber]);
  }
  
  // perform the IDCT on each block
  // -1 <-- inverse
  DoSmartBlockDCT(blocks, -1);

  vector<VectorXd> blocksEigen(numBlocks);

  // this part is inefficient---we should implement a DCT that operates on VectorXd so that
  // we don't have to loop through all the blocks again
  for (int blockNumber = 0; blockNumber < numBlocks; blockNumber++) {
    VECTOR blockFlat = blocks[blockNumber].flattened();
    VectorXd blockEigen = EIGEN::convert(blockFlat);
    blocksEigen[blockNumber] = blockEigen;
  } 
  return blocksEigen;
} 

////////////////////////////////////////////////////////
// uses DecodeScalarField three times to reconstruct
// a lossy vector field
////////////////////////////////////////////////////////
VECTOR3_FIELD_3D DecodeVectorField(MATRIX_COMPRESSION_DATA& data, int col) {
  TIMER functionTimer(__FUNCTION__);
  
  DECOMPRESSION_DATA decompression_dataX = data.get_decompression_dataX();
  DECOMPRESSION_DATA decompression_dataY = data.get_decompression_dataY();
  DECOMPRESSION_DATA decompression_dataZ = data.get_decompression_dataZ();

  VEC3I dims = decompression_dataX.get_dims();
  int xRes = dims[0];
  int yRes = dims[1];
  int zRes = dims[2];

  int* allDataX = data.get_dataX();
  int* allDataY = data.get_dataY();
  int* allDataZ = data.get_dataZ();

  FIELD_3D scalarX = DecodeScalarField(decompression_dataX, allDataX, col);
  FIELD_3D scalarY = DecodeScalarField(decompression_dataY, allDataY, col);
  FIELD_3D scalarZ = DecodeScalarField(decompression_dataZ, allDataZ, col);

  VECTOR scalarXflat = scalarX.flattened();
  VECTOR scalarYflat = scalarY.flattened();
  VECTOR scalarZflat = scalarZ.flattened();
   
  double* X_array = (double*) malloc(sizeof(double) * xRes * yRes * zRes);
  double* Y_array = (double*) malloc(sizeof(double) * xRes * yRes * zRes);
  double* Z_array = (double*) malloc(sizeof(double) * xRes * yRes * zRes);

  X_array = CastToDouble(scalarXflat, X_array);
  Y_array = CastToDouble(scalarYflat, Y_array);
  Z_array = CastToDouble(scalarZflat, Z_array);
  
  VECTOR3_FIELD_3D result(X_array, Y_array, Z_array, xRes, yRes, zRes);

  free(X_array);
  free(Y_array);
  free(Z_array);

  return result;
}

////////////////////////////////////////////////////////
// uses DecodeScalarFieldFast three times to reconstruct
// a lossy vector field
////////////////////////////////////////////////////////
void DecodeVectorFieldFast(MATRIX_COMPRESSION_DATA& data, int col, VECTOR3_FIELD_3D& vecfieldToFill) {
  TIMER functionTimer(__FUNCTION__);
  
  const DECOMPRESSION_DATA& decompression_dataX = data.get_decompression_dataX();
  const DECOMPRESSION_DATA& decompression_dataY = data.get_decompression_dataY();
  const DECOMPRESSION_DATA& decompression_dataZ = data.get_decompression_dataZ();

  const VEC3I& dims = decompression_dataX.get_dims();
  int xRes = dims[0];
  int yRes = dims[1];
  int zRes = dims[2];
  vecfieldToFill.resizeAndWipe(xRes, yRes, zRes);

  int* allDataX = data.get_dataX();
  int* allDataY = data.get_dataY();
  int* allDataZ = data.get_dataZ();

  FIELD_3D scalarX; 
  DecodeScalarFieldFast(decompression_dataX, allDataX, col, scalarX);
  FIELD_3D scalarY;
  DecodeScalarFieldFast(decompression_dataY, allDataY, col, scalarY);
  FIELD_3D scalarZ;
  DecodeScalarFieldFast(decompression_dataZ, allDataZ, col, scalarZ);

  double* X_array = scalarX.data();
  double* Y_array = scalarY.data();
  double* Z_array = scalarZ.data();

  VECTOR3_FIELD_3D result(X_array, Y_array, Z_array, xRes, yRes, zRes);

  vecfieldToFill.swapPointers(result);
}

////////////////////////////////////////////////////////
// uses the decode vector field on each column to reconstruct
// a lossy full matrix  
////////////////////////////////////////////////////////
MatrixXd DecodeFullMatrix(MATRIX_COMPRESSION_DATA& data) {
  TIMER functionTimer(__FUNCTION__);

  DECOMPRESSION_DATA decompression_dataX = data.get_decompression_dataX();
  int numCols = decompression_dataX.get_numCols();

  vector<VectorXd> columnList(numCols);
  for (int col = 0; col < numCols; col++) {
    cout << "Column: " << col << endl;
    VECTOR3_FIELD_3D decodedV = DecodeVectorField(data, col);
    VECTOR flattenedV = decodedV.flattened();
    VectorXd flattenedV_eigen = EIGEN::convert(flattenedV);
    columnList[col] = (flattenedV_eigen);
  }
  MatrixXd decodedResult = EIGEN::buildFromColumns(columnList);
  return decodedResult; 
}

////////////////////////////////////////////////////////
// uses the decode vector field on each column to reconstruct
// a lossy full matrix  
////////////////////////////////////////////////////////
MatrixXd DecodeFullMatrixFast(MATRIX_COMPRESSION_DATA& data) {
  TIMER functionTimer(__FUNCTION__);

  DECOMPRESSION_DATA decompression_dataX = data.get_decompression_dataX();
  int numCols = decompression_dataX.get_numCols();

  vector<VectorXd> columnList(numCols);
  VECTOR3_FIELD_3D decodedV;
  for (int col = 0; col < numCols; col++) {
    cout << "Column: " << col << endl;
    DecodeVectorFieldFast(data, col, decodedV);
    VectorXd flattenedV_eigen = decodedV.flattenedEigen(); 
    columnList[col] = flattenedV_eigen;
  }
  MatrixXd decodedResult = EIGEN::buildFromColumns(columnList);
  return decodedResult; 
}
//////////////////////////////////////////////////////////////////////
// unproject the reduced coordinate into the peeled cells in this field 
// using compression data
//////////////////////////////////////////////////////////////////////
void PeeledCompressedUnproject(VECTOR3_FIELD_3D& V, MATRIX_COMPRESSION_DATA& U_data, const VectorXd& q) {
  TIMER functionTimer(__FUNCTION__);

  int xRes = V.xRes();
  int yRes = V.yRes();
  int zRes = V.zRes();
  DECOMPRESSION_DATA dataX = U_data.get_decompression_dataX();
  int totalColumns = dataX.get_numCols();
  const VEC3I& dims = dataX.get_dims();
  
  /*
  cout << "xRes: " << xRes << endl;
  cout << "yRes: " << yRes << endl;
  cout << "zRes: " << zRes << endl;
  cout << "dims[0]: " << dims[0] << endl;
  cout << "dims[1]: " << dims[1] << endl;
  cout << "dims[2]: " << dims[2] << endl;
  */

  // verify that the (peeled) dimensions match
  assert( xRes - 2 == dims[0] && yRes - 2 == dims[1] && zRes - 2 == dims[2] );
  assert ( totalColumns == q.size() );

  // times 3 since it is a vec3 field
  const int numRows = 3 * (xRes - 2) * (yRes - 2) * (zRes - 2);

  VectorXd result(numRows);
  result.setZero();
  VECTOR3_FIELD_3D decodedVecField;

  for (int col = 0; col < totalColumns; col++) {

    DecodeVectorFieldFast(U_data, col, decodedVecField);
    VectorXd decodedEigen = decodedVecField.flattenedEigen(); 
    double coeff = q[col];
    result += (coeff * decodedEigen);
  }

  V.setWithPeeled(result);

}


double GetDotProductSum(vector<VectorXd> Vlist, vector<VectorXd> Wlist) {

  assert(Vlist.size() == Wlist.size());

  int size = Vlist.size();

  double dotProductSum = 0.0;
  for (int i = 0; i < size; i++) {
    double dotProduct_i = Vlist[i].dot(Wlist[i]);
    dotProductSum += dotProduct_i;
  }

  return dotProductSum;

}


VectorXd PeeledCompressedProject(VECTOR3_FIELD_3D& V, MATRIX_COMPRESSION_DATA& U_data)
{
  TIMER functionTimer(__FUNCTION__);

  DECOMPRESSION_DATA dataX = U_data.get_decompression_dataX();
  int* allDataX = U_data.get_dataX();
  DECOMPRESSION_DATA dataY = U_data.get_decompression_dataY();
  int* allDataY = U_data.get_dataY();
  DECOMPRESSION_DATA dataZ = U_data.get_decompression_dataZ();
  int* allDataZ = U_data.get_dataZ();

  const VEC3I& dims = dataX.get_dims();
  const int xRes = dims[0];
  const int yRes = dims[1];
  const int zRes = dims[2];
  const int totalColumns = dataX.get_numCols();
  VectorXd result(totalColumns);

  // move to the peeled coordinates
  VECTOR3_FIELD_3D V_peeled = V.peelBoundary();
  FIELD_3D V_X, V_Y, V_Z;
  // fill V_X, V_Y, V_Z
  GetScalarFields(V_peeled, V_X, V_Y, V_Z);
  
  // GetBlocksEigen zero-pads as not to disturb the integrity of the matrix-vector multiply
  vector<VectorXd> Xpart = GetBlocksEigen(V_X);
  vector<VectorXd> Ypart = GetBlocksEigen(V_Y);
  vector<VectorXd> Zpart = GetBlocksEigen(V_Z);
  
  vector<VectorXd> blocks;
  for (int col = 0; col < totalColumns; col++) {

    TIMER columnLoopTimer("peeled project column loop");
    
    double totalSum = 0.0;
    DecodeScalarFieldEigenFast(dataX, allDataX, col, blocks);
    totalSum += GetDotProductSum(blocks, Xpart);
   
    DecodeScalarFieldEigenFast(dataY, allDataY, col, blocks);
    totalSum += GetDotProductSum(blocks, Ypart); 
 
    DecodeScalarFieldEigenFast(dataZ, allDataZ, col, blocks);
    totalSum += GetDotProductSum(blocks, Zpart);

    result[col] = totalSum;

  }

  return result;

}
  
  
