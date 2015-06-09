#include <iostream>
#include <sys/stat.h>
#include "COMPRESSION_REWRITE.h"

using std::vector;
using std::cout;
using std::endl;

const double DCT_NORMALIZE = 1.0 / sqrt( 8 * BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE );
const double SQRT_ONEHALF = 1.0/sqrt(2.0);
const double SQRT_TWO = sqrt(2.0);

////////////////////////////////////////////////////////
// Function Implementations
////////////////////////////////////////////////////////


////////////////////////////////////////////////////////
// cast a FIELD_3D to an INTEGER_FIELD_3D by rounding 
// *NOTE* an in-place method for doing the same operation
// is now supported in FIELD_3D
////////////////////////////////////////////////////////
void RoundFieldToInt(const FIELD_3D& F, INTEGER_FIELD_3D* castedField) {
  TIMER functionTimer(__FUNCTION__);

  assert( F.totalCells() == castedField->totalCells() );

  for (int i = 0; i < F.totalCells(); i++) {
    (*castedField)[i] = rint(F[i]);
  }
}

////////////////////////////////////////////////////////
// cast an INTEGER_FIELD_3D to a FIELD_3D
////////////////////////////////////////////////////////
void CastIntFieldToDouble(const INTEGER_FIELD_3D& F, FIELD_3D* castedField) {
  TIMER functionTimer(__FUNCTION__);
  
  assert( F.totalCells() == castedField->totalCells() );

  for (int i = 0; i < F.totalCells(); i++) {
    (*castedField)[i] = (double)F[i];
  }
}

////////////////////////////////////////////////////////
// operates on a VectorXi and fills another VectorXi
// with its cumulative sum starting at zero and omitting the last
// entry. e.g. if the input vector was (1, 2, 3, 4),
// the result would be (0, 1, 3, 6)
////////////////////////////////////////////////////////
void ModifiedCumSum(const VectorXi& V, VectorXi* sum) 
{
  TIMER functionTimer(__FUNCTION__);
  
  // wipe the output and set it to the appropriate size
  sum->setZero(V.size());
  int accumulation = 0;

  // note the loop starts offset at 1
  for (int i = 1; i < V.size(); i++) {

    // accumulate the previous value
    accumulation += V[i - 1];
    (*sum)[i] = accumulation;
  }

}


////////////////////////////////////////////////////////
// returns the 3 component scalar fields 
// from a passed in vector field 
////////////////////////////////////////////////////////
void GetScalarFields(const VECTOR3_FIELD_3D& V, FIELD_3D* X, FIELD_3D* Y, FIELD_3D* Z) {
  TIMER functionTimer(__FUNCTION__);
  *X = V.scalarField(0);
  *Y = V.scalarField(1);
  *Z = V.scalarField(2);
}

// ZigzagFlattned/Unflatten


////////////////////////////////////////////////////////
// Given a passed in buffer and a 'direction' 
// (1 for forward, -1 for inverse),
// we return an fftw plan for doing an in-place 3d dct 
// which is linked to the in buffer
////////////////////////////////////////////////////////
void Create_DCT_Plan(double* in, int direction, fftw_plan* plan) {
  TIMER functionTimer(__FUNCTION__);

  // direction is 1 for a forward transform, -1 for a backward transform
  assert( direction == 1 || direction == -1 );

  int xRes = BLOCK_SIZE;
  int yRes = BLOCK_SIZE;
  int zRes = BLOCK_SIZE;

  fftw_r2r_kind kind;
  if (direction == 1) {
    kind = FFTW_REDFT10;
  }
  else {
    kind = FFTW_REDFT01;
  }
  // 'in' appears twice since it is in-place
  *plan = fftw_plan_r2r_3d(zRes, yRes, xRes, in, in, kind, kind, kind, FFTW_MEASURE);
}

////////////////////////////////////////////////////////
// perform a unitary normalization on the passed in 
// buffer of a field
////////////////////////////////////////////////////////
void DCT_Unitary_Normalize(double* buffer)
{
  TIMER functionTimer(__FUNCTION__);

  int totalCells = BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE;
  int slabSize = BLOCK_SIZE * BLOCK_SIZE;

  for (int i = 0; i < totalCells; i++) {
    buffer[i] *= DCT_NORMALIZE;
  }
  
  for (int z = 0; z < BLOCK_SIZE; z++) {
    for (int y = 0; y < BLOCK_SIZE; y++) {
      buffer[z * slabSize + y * BLOCK_SIZE] *= SQRT_ONEHALF;
    }
  }

  for (int y = 0; y < BLOCK_SIZE; y++) {
    for (int x = 0; x < BLOCK_SIZE; x++) {
      buffer[y * BLOCK_SIZE + x] *= SQRT_ONEHALF;
    }
  }

  for (int z = 0; z < BLOCK_SIZE; z++) {
    for (int x = 0; x < BLOCK_SIZE; x++) {
      buffer[z * slabSize + x] *= SQRT_ONEHALF;
    }
  }
} 

////////////////////////////////////////////////////////
// undo the unitary normalization prior to doing an
// fftw-style idct
////////////////////////////////////////////////////////
void UndoNormalize(FIELD_3D* F)
{
  TIMER functionTimer(__FUNCTION__);
  assert( F->xRes() == BLOCK_SIZE && F->yRes() == BLOCK_SIZE && F->zRes() == BLOCK_SIZE );
  
  int totalCells = BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE;
  int slabSize = BLOCK_SIZE * BLOCK_SIZE;
  double* buffer = F->data();

  for (int i = 0; i < totalCells; i++) {
    buffer[i] *= DCT_NORMALIZE;
  }
  
  for (int z = 0; z < BLOCK_SIZE; z++) {
    for (int y = 0; y < BLOCK_SIZE; y++) {
      buffer[z * slabSize + y * BLOCK_SIZE] *= SQRT_TWO;
    }
  }

  for (int y = 0; y < BLOCK_SIZE; y++) {
    for (int x = 0; x < BLOCK_SIZE; x++) {
      buffer[y * BLOCK_SIZE + x] *= SQRT_TWO;
    }
  }

  for (int z = 0; z < BLOCK_SIZE; z++) {
    for (int x = 0; x < BLOCK_SIZE; x++) {
      buffer[z * slabSize + x] *= SQRT_TWO;
    }
  }
}
////////////////////////////////////////////////////////
// undo the unitary normalization on a flattened
// FIELD_3D  prior to doing an fftw-style idct
////////////////////////////////////////////////////////
void UndoNormalizeEigen(VectorXd* F)
{
  TIMER functionTimer(__FUNCTION__);
  
  int totalCells = BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE;
  assert( F->size() == totalCells ); 

  int slabSize = BLOCK_SIZE * BLOCK_SIZE;

  for (int i = 0; i < totalCells; i++) {
    (*F)[i] *= DCT_NORMALIZE;
  }
  
  for (int z = 0; z < BLOCK_SIZE; z++) {
    for (int y = 0; y < BLOCK_SIZE; y++) {
      (*F)[z * slabSize + y * BLOCK_SIZE] *= SQRT_TWO;
    }
  }

  for (int y = 0; y < BLOCK_SIZE; y++) {
    for (int x = 0; x < BLOCK_SIZE; x++) {
      (*F)[y * BLOCK_SIZE + x] *= SQRT_TWO;
    }
  }

  for (int z = 0; z < BLOCK_SIZE; z++) {
    for (int x = 0; x < BLOCK_SIZE; x++) {
      (*F)[z * slabSize + x] *= SQRT_TWO;
    }
  }
}
////////////////////////////////////////////////////////
// given a passed in FIELD_3D, fftw plan, and 
// corresponding 'in' buffer, performs the corresponding
// transform on the field. this one is *unitary normalization* 
////////////////////////////////////////////////////////
void DCT_Smart_Unitary(const fftw_plan& plan, int direction, double* in, FIELD_3D* F)
{
  TIMER functionTimer(__FUNCTION__);

  int xRes = F->xRes();
  int yRes = F->yRes();
  int zRes = F->zRes();
  int totalCells = xRes * yRes * zRes;

  assert ( xRes == BLOCK_SIZE && yRes == BLOCK_SIZE && zRes == BLOCK_SIZE );

  if (direction == -1) { // inverse transform; need to pre-normalize!
    UndoNormalize(F);
  }

  // fill the 'in' buffer
  memcpy(in, F->data(), totalCells * sizeof(double));
 
  
  TIMER fftTimer("fftw execute");
  fftw_execute(plan);
  fftTimer.stop();
  
  // 'in' is now overwritten to the result of the transform

  if (direction == 1) { // forward transform; need to post-normalize!
    DCT_Unitary_Normalize(in);
  }

  // rewrite F's data with the new contents of in 
  memcpy(F->data(), in, totalCells * sizeof(double));
}
////////////////////////////////////////////////////////
// given a passed in flattened-out FIELD_3D, fftw plan, and 
// corresponding 'in' buffer, performs the corresponding
// transform on the field. this one is *unitary normalization* 
////////////////////////////////////////////////////////
void DCT_Smart_Unitary_Eigen(const fftw_plan& plan, int direction, double* in, VectorXd* F)
{
  TIMER functionTimer(__FUNCTION__);

  int totalCells = F->size(); 

  assert ( totalCells == BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE ); 

  if (direction == -1) { // inverse transform; need to pre-normalize!
    UndoNormalizeEigen(F);
  }

  // fill the 'in' buffer
  memcpy(in, F->data(), totalCells * sizeof(double));
 
  TIMER fftTimer("fftw execute");
  fftw_execute(plan);
  fftTimer.stop();
  // 'in' is now overwritten to the result of the transform

  if (direction == 1) { // forward transform; need to post-normalize!
    DCT_Unitary_Normalize(in);
  }

  // rewrite F's data with the new contents of in 
  memcpy(F->data(), in, totalCells * sizeof(double));
}

/*
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
*/

////////////////////////////////////////////////////////
// performs 8 x 8 x 8 3D DCT block compression on the
// passed in FIELD_3D, returning a lossy version 
////////////////////////////////////////////////////////
//
//
/*
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
*/

////////////////////////////////////////////////////////
// given passed in dimensions, computes how much we
// have to pad by in each dimension to reach the next
// multiple of BLOCK_SIZE for even block subdivision 
////////////////////////////////////////////////////////

void GetPaddings(const VEC3I& v, VEC3I* paddings)
{ 
  TIMER functionTimer(__FUNCTION__);
  int xRes = v[0];
  int yRes = v[1];
  int zRes = v[2];
  int xPadding = (BLOCK_SIZE - (xRes % BLOCK_SIZE)) % BLOCK_SIZE;     // how far are you from the next multiple of 8?
  int yPadding = (BLOCK_SIZE - (yRes % BLOCK_SIZE)) % BLOCK_SIZE;
  int zPadding = (BLOCK_SIZE - (zRes % BLOCK_SIZE)) % BLOCK_SIZE;
  (*paddings)[0] = xPadding;
  (*paddings)[1] = yPadding;
  (*paddings)[2] = zPadding;
}

////////////////////////////////////////////////////////
// given a passed in FIELD_3D, pad it and  parse it 
// into a vector of 8 x 8 x 8 blocks (listed in row-major order)
////////////////////////////////////////////////////////

void GetBlocks(const FIELD_3D& F, vector<FIELD_3D>* blocks)
{
  TIMER functionTimer(__FUNCTION__);

  int xRes = F.xRes();
  int yRes = F.yRes();
  int zRes = F.zRes();
  VEC3I v(xRes, yRes, zRes);
 
  VEC3I paddings(0, 0, 0);
  // fill these in with the appropriate paddings
  GetPaddings(v, &paddings);

  FIELD_3D F_padded = F.pad_xyz(paddings);
  
  // update the resolutions to the padded ones 
  xRes = F_padded.xRes();
  yRes = F_padded.yRes();
  zRes = F_padded.zRes();

  // sanity check that our padder had the desired effect
  assert(xRes % BLOCK_SIZE == 0);
  assert(yRes % BLOCK_SIZE == 0);
  assert(zRes % BLOCK_SIZE == 0);

  for (int z = 0; z < zRes/BLOCK_SIZE; z++) {
    for (int y = 0; y < yRes/BLOCK_SIZE; y++) {
      for (int x = 0; x < xRes/BLOCK_SIZE; x++) {
        blocks->push_back(F_padded.subfield(BLOCK_SIZE*x, BLOCK_SIZE*(x+1), 
            BLOCK_SIZE*y, BLOCK_SIZE*(y+1), BLOCK_SIZE*z, BLOCK_SIZE*(z+1)));
      }
    }
  }
}
////////////////////////////////////////////////////////
// given a passed in FIELD_3D, pad it with zeros and  parse it 
// into a vector of flattened 8 x 8 x 8 blocks (listed in row-major order)
////////////////////////////////////////////////////////

void GetBlocksEigen(const FIELD_3D& F, vector<VectorXd>* blocks)
{
  TIMER functionTimer(__FUNCTION__);

  int xRes = F.xRes();
  int yRes = F.yRes();
  int zRes = F.zRes();
  VEC3I v(xRes, yRes, zRes);
 
  VEC3I paddings(0, 0, 0);
  // fill these in with the appropriate paddings
  GetPaddings(v, &paddings);

  // call zero pad rather than continuous value pad
  FIELD_3D F_padded = F.zeroPad_xyz(paddings);
  
  // update the resolutions to the padded ones 
  xRes = F_padded.xRes();
  yRes = F_padded.yRes();
  zRes = F_padded.zRes();

  // sanity check that our padder had the desired effect
  assert(xRes % BLOCK_SIZE == 0);
  assert(yRes % BLOCK_SIZE == 0);
  assert(zRes % BLOCK_SIZE == 0);

  // resize blocks appropriately
  blocks->resize(xRes/BLOCK_SIZE * yRes/BLOCK_SIZE * zRes/BLOCK_SIZE);

  int index = 0;
  for (int z = 0; z < zRes/BLOCK_SIZE; z++) {
    for (int y = 0; y < yRes/BLOCK_SIZE; y++) {
      for (int x = 0; x < xRes/BLOCK_SIZE; x++, index++) {
        // add the flattened out block to the list
        (*blocks)[index] = (F_padded.subfield(BLOCK_SIZE*x, BLOCK_SIZE*(x+1), 
            BLOCK_SIZE*y, BLOCK_SIZE*(y+1), BLOCK_SIZE*z, BLOCK_SIZE*(z+1)).flattenedEigen());
      }
    }
  }
}
////////////////////////////////////////////////////////
// reconstruct a FIELD_3D with the passed in dims
// from a list of 8 x 8 x 8 blocks 
////////////////////////////////////////////////////////

void AssimilateBlocks(const VEC3I& dims, const vector<FIELD_3D>& V, FIELD_3D* assimilatedField)
{
  TIMER functionTimer(__FUNCTION__);

  const int xRes = dims[0];
  const int yRes = dims[1];
  const int zRes = dims[2];

  assert( xRes % BLOCK_SIZE == 0 && yRes % BLOCK_SIZE == 0 && zRes % BLOCK_SIZE == 0 );
  assert( xRes == assimilatedField->xRes() && yRes == assimilatedField->yRes() && zRes == assimilatedField->zRes() );

  for (int z = 0; z < zRes; z++) {
    for (int y = 0; y < yRes; y++) {
      for (int x = 0; x < xRes; x++) {
        int index = (x/BLOCK_SIZE) + (y/BLOCK_SIZE) * (xRes/BLOCK_SIZE) + (z/BLOCK_SIZE) * (xRes/BLOCK_SIZE) * (yRes/BLOCK_SIZE);     // warning, evil integer division happening!
        (*assimilatedField)(x, y, z) = V[index](x % BLOCK_SIZE, y % BLOCK_SIZE, z % BLOCK_SIZE);             
      }
    }
  }

}

////////////////////////////////////////////////////////
// performs a UNITARY dct/idct on each individual block of a passed in
// vector of blocks. direction 1 is dct, -1 is idct.
////////////////////////////////////////////////////////

void UnitaryBlockDCT(int direction, vector<FIELD_3D>* blocks) 
{
  TIMER functionTimer(__FUNCTION__);
  
  // allocate a buffer for the size of an block-size block block
  double* in = (double*) fftw_malloc(BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE * sizeof(double));

  // make the appropriate plan
  fftw_plan plan;
  Create_DCT_Plan(in, direction, &plan);
  
  for (auto itr = blocks->begin(); itr != blocks->end(); ++itr) {
    // take the transform at *itr (which is a FIELD_3D)
    // and overwrite its contents
    DCT_Smart_Unitary(plan, direction, in, &(*itr));
  }

  fftw_free(in);
  fftw_destroy_plan(plan);
  fftw_cleanup();
}
////////////////////////////////////////////////////////
// performs a UNITARY dct/idct on each individual flattened
// block of a passed in vector of blocks. direction 1 is dct, -1 is idct.
////////////////////////////////////////////////////////

void UnitaryBlockDCTEigen(int direction, vector<VectorXd>* blocks) 
{
  TIMER functionTimer(__FUNCTION__);
  
  // allocate a buffer for the size of an block-size block
  double* in = (double*) fftw_malloc(BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE * sizeof(double));

  // make the appropriate plan
  fftw_plan plan;
  Create_DCT_Plan(in, direction, &plan);
  
  for (auto itr = blocks->begin(); itr != blocks->end(); ++itr) {
    // take the transform at *itr (which is a FIELD_3D)
    // and overwrite its contents
    DCT_Smart_Unitary_Eigen(plan, direction, in, &(*itr));
  }

  fftw_free(in);
  fftw_destroy_plan(plan);
  fftw_cleanup();
}

////////////////////////////////////////////////////////
// build a block diagonal matrix with repeating copies of the passed
// in matrix A along the diagonal. very poor memory usage!
////////////////////////////////////////////////////////
void BlockDiagonal(const MatrixXd& A, int count, MatrixXd* B)
{
  TIMER functionTimer(__FUNCTION__);

  *B = MatrixXd::Zero(A.rows() * count, A.cols() * count);

  // **************************************************   
  // check the memory consumption
  double GB = A.rows() * count * A.cols() * count * sizeof(double) / pow(2.0, 30);
  cout << "B is consuming " << GB << " GB of memory!" << endl; 
  // **************************************************   

  for (int i = 0; i < count; i++) {
    B->block(i * A.rows(), i * A.cols(), A.rows(), A.cols()) = A;
  }
}

////////////////////////////////////////////////////////
// build a block diagonal matrix with repeating copies of the passed
// in matrix A along the diagonal. assumes for simplicity
// that A is 3 x 3 for the SVD case
////////////////////////////////////////////////////////
void SparseBlockDiagonal(const MatrixXd& A, int count, SparseMatrix<double>* B)
{
  TIMER functionTimer(__FUNCTION__);
  
  assert( A.rows() == 3 && A.cols() == 3 );

  typedef Eigen::Triplet<double> T;
  vector<T> tripletList;

  int nonzeros = A.rows() * A.cols() * count;
  tripletList.reserve(nonzeros);
  for (int i = 0; i < A.rows() * count; i++) {
    if (i % 3 == 0) {
      tripletList.push_back(T(i, i,     A(0, 0)));
      tripletList.push_back(T(i, i + 1, A(0, 1)));
      tripletList.push_back(T(i, i + 2, A(0, 2)));
      }
    else if (i % 3 == 1) {
      tripletList.push_back(T(i, i - 1, A(1, 0)));
      tripletList.push_back(T(i, i,     A(1, 1)));
      tripletList.push_back(T(i, i + 1, A(1, 2)));
    }
    else { // i % 3 == 2 
      tripletList.push_back(T(i, i - 2, A(2, 0)));
      tripletList.push_back(T(i, i - 1, A(2, 1)));
      tripletList.push_back(T(i, i,     A(2, 2)));
    }
  }

  *B = SparseMatrix<double>(A.rows() * count, A.cols() * count);
  B->setFromTriplets(tripletList.begin(), tripletList.end());

}

////////////////////////////////////////////////////////
// given a passed in vec3 field, build a matrix with 3 columns,
// one for each of the x, y, and z components
////////////////////////////////////////////////////////
void BuildXYZMatrix(const VECTOR3_FIELD_3D& V, MatrixXd* A) 
{
  TIMER functionTimer(__FUNCTION__);

  int N = V.xRes() * V.yRes() * V.zRes();
  *A = MatrixXd::Zero(N, 3);

  FIELD_3D Vx, Vy, Vz;
  GetScalarFields(V, &Vx, &Vy, &Vz);

  A->col(0) = Vx.flattenedEigen();
  A->col(1) = Vy.flattenedEigen();
  A->col(2) = Vz.flattenedEigen();
}

////////////////////////////////////////////////////////
// find a new 3d coordinate system for a vector field using svd and transform into it
// fill s with the singular values and v^T with the coordinate transform matrix
////////////////////////////////////////////////////////
void TransformVectorFieldSVD(VectorXd* s, MatrixXd* v, VECTOR3_FIELD_3D* V)
{
  TIMER functionTimer(__FUNCTION__);

  MatrixXd xyzMatrix;
  BuildXYZMatrix(*V, &xyzMatrix);
  JacobiSVD<MatrixXd> svd(xyzMatrix, ComputeThinU | ComputeThinV);
  *s = svd.singularValues();
  *v = svd.matrixV();

  int count = V->xRes() * V->yRes() * V->zRes();
  
  SparseMatrix<double> B;
  SparseBlockDiagonal(v->transpose(), count, &B);

  VectorXd transformProduct = B * V->flattenedEigen();
  memcpy(V->data(), transformProduct.data(), 3 * count * sizeof(double));
}

////////////////////////////////////////////////////////
// perform the coordinate transform without having
// to recompute the SVD by reading in from a cached 3 x 3
// v matrix
////////////////////////////////////////////////////////
void TransformVectorFieldSVDCached(Matrix3d* v, VECTOR3_FIELD_3D* V)
{
  TIMER functionTimer(__FUNCTION__);

  // collapse V into 3 columns by coordinate
  MatrixXd xyzMatrix;
  BuildXYZMatrix(*V, &xyzMatrix);

  // build the sparse block diagonal matrix with V^T on the diagonal
  int count = V->xRes() * V->yRes() * V->zRes();
  SparseMatrix<double> B;
  SparseBlockDiagonal(v->transpose(), count, &B);

  // compute the product and copy it into the result
  VectorXd transformProduct = B * V->flattenedEigen();
  memcpy(V->data(), transformProduct.data(), 3 * count * sizeof(double));
}
////////////////////////////////////////////////////////
// find a new 3d coordinate system for a vector field using svd and transform into it
// fill s with the singular values and v^T with the coordinate transform matrix.
// update the compression data to account for the transform matrix and its
// corresponding singular values.
////////////////////////////////////////////////////////
void TransformVectorFieldSVDCompression(VECTOR3_FIELD_3D* V, COMPRESSION_DATA* data)
{
  TIMER functionTimer(__FUNCTION__);

  // build the N x 3 matrix from V
  MatrixXd xyzMatrix;
  BuildXYZMatrix(*V, &xyzMatrix);
  
  // compute the thin svd
  JacobiSVD<MatrixXd> svd(xyzMatrix, ComputeThinU | ComputeThinV);

  // fetch the data to be updated
  int numCols = data->get_numCols();
  vector<Vector3d>* singularList = data->get_singularList();
  vector<Matrix3d>* vList = data->get_vList();

  // if it's the first time calling TransformVectorFieldSVD in a chain, 
  // preallocate
  if (singularList->size() <= 0 && vList->size() <= 0) {
    singularList->reserve(numCols);
    vList->reserve(numCols);
  }
  
  // update the compression data
  singularList->push_back(svd.singularValues());
  vList->push_back(svd.matrixV());

  // build the sparse block diagonal transform matrix
  int count = V->xRes() * V->yRes() * V->zRes();
  SparseMatrix<double> B;
  SparseBlockDiagonal(svd.matrixV().transpose(), count, &B);

  // compute the transformation using a matrix-vector multiply
  VectorXd transformProduct = B * V->flattenedEigen();

  // copy the result into V (in-place)
  memcpy(V->data(), transformProduct.data(), 3 * count * sizeof(double));
}

////////////////////////////////////////////////////////
// undo the effects of a previous svd coordinate transformation using the passed
// in v matrix 
////////////////////////////////////////////////////////
void UntransformVectorFieldSVD(const MatrixXd& v, VECTOR3_FIELD_3D* transformedV)
{
  TIMER functionTimer(__FUNCTION__);

  MatrixXd xyzMatrix;
  BuildXYZMatrix(*transformedV, &xyzMatrix);

  int count = transformedV->xRes() * transformedV->yRes() * transformedV->zRes();

  SparseMatrix<double> B;
  SparseBlockDiagonal(v, count, &B);

  VectorXd transformProduct = B * transformedV->flattenedEigen();
  memcpy(transformedV->data(), transformProduct.data(), 3 * count * sizeof(double));
}

////////////////////////////////////////////////////////
// Normalize the block contents to a resolution of
// nBits based on the DC component. Update the sList.
////////////////////////////////////////////////////////
void PreprocessBlock(FIELD_3D* F, int blockNumber, int col, COMPRESSION_DATA* data)
{
  TIMER functionTimer(__FUNCTION__);
  int nBits = data->get_nBits();

  // normalize so that the DC component is at 2 ^ {nBits - 1} - 1
  double s = (pow(2, nBits - 1) - 1) / (*F)[0];
  (*F) *= s;

  // fetch data for updating sList 
  MatrixXd* sListMatrix = data->get_sListMatrix();
  int numBlocks = data->get_numBlocks();
  int numCols = data->get_numCols();

  // if it's the first time PreprocessBlock is called in a chain, resize
  if (sListMatrix->cols() <= 0) { 
    sListMatrix->setZero(numBlocks, numCols);
  }

  // update sList
  (*sListMatrix)(blockNumber, col) = s;
}

////////////////////////////////////////////////////////
// Binary search to find the appropriate gamma given
// desired percent threshold within maxIterations. Prints
// out information as it goes.
////////////////////////////////////////////////////////
void TuneGammaVerbose(const FIELD_3D& F, int blockNumber, int col, 
    COMPRESSION_DATA* data, FIELD_3D* damp)
{
  TIMER functionTimer(__FUNCTION__);

  // fetch parameters from data
  int nBits = data->get_nBits();
  int maxIterations = data->get_maxIterations();
  double percent = data->get_percent();

  double lower = 0.0;
  // QUESTION: how should we define upper?
  double upper = nBits;
  cout << "Upper: " << upper << endl;
  // arbitrarily set epsilon to be 0.5%
  double epsilon = 0.005;
  double gamma = 0.5 * (upper + lower);
  cout << "Initial gamma: " << gamma << endl;
  damp->toPower(gamma);
  cout << "Initial damping array: " << endl;
  cout << damp->flattened() << endl;
  
  cout << "F: " << endl;
  cout << F.flattened() << endl;
  // the total amount of energy in the Fourier space
  double totalEnergy = F.sumSq();
  FIELD_3D damped = ( F / (*damp) );
  damped.roundInt();
  cout << "Damped block: " << endl;
  cout << damped.flattened() << endl;

  cout << "Undamped block: " << endl;
  cout << ((*damp) * damped).flattened() << endl;

  double energyDiff = abs(totalEnergy - ( (*damp) * damped ).sumSq());
  cout << "Absolute energy difference: " << energyDiff << endl;
  double percentEnergy = 1.0 - (energyDiff / totalEnergy);
  cout << "Initial percent: " << percentEnergy << endl;
  int iterations = 0;
   
  while ( abs( percent - percentEnergy ) > epsilon && iterations < maxIterations) {

    if (percentEnergy < percent) { // too much damping; need to lower gamma
      upper = gamma;
      gamma = 0.5 * (upper + lower);

      // to the power of 1 / upper brings it back to the vanilla state, 
      // from which we raise it to the new gamma
      damp->toPower(gamma / upper);
    }

    else { // not enough damping; need to increase gamma
      lower = gamma;
      gamma = 0.5 * (upper + lower);

      // to the power of 1 / lower brings it back to the vanilla state, 
      // from which we raise it to the new gamma
      damp->toPower(gamma / lower);
    }

    cout << "New damping array: " << endl;
    cout << damp->flattened() << endl;
    // update percentEnergy
    damped = ( F / (*damp) );
    damped.roundInt();
    cout << "New damped block: " << endl;
    cout << damped.flattened() << endl;
    cout << "New undamped block: " << endl;
    cout << ((*damp) * damped).flattened() << endl;
    energyDiff = abs(totalEnergy - ( (*damp) * damped ).sumSq());
    percentEnergy =  1.0 - (energyDiff / totalEnergy);
    cout << "New percent energy: " << percentEnergy << endl;
    cout << "New gamma: " << gamma << endl;
    iterations++;
    cout << "Next iteration: " << iterations << endl;
  }
  
  cout << "Took " << iterations << " iterations to compute gamma!\n";
  cout << "Percent Energy ended up at : " << percentEnergy << endl;
  cout << "Gamma ended up at: " << gamma << endl;

  // fetch data to update gammaList
  MatrixXd* gammaListMatrix = data->get_gammaListMatrix();
  int numBlocks = data->get_numBlocks();
  int numCols = data->get_numCols();

  // if it's the first time TuneGamma is called in a chain, resize
  if (gammaListMatrix->cols() <= 0) { 
    gammaListMatrix->setZero(numBlocks, numCols);
  }

  (*gammaListMatrix)(blockNumber, col) = gamma;

}

////////////////////////////////////////////////////////
// Binary search to find the appropriate gamma given
// desired percent threshold within maxIterations. 
///////////////////////////////////////////////////////
void TuneGamma(const FIELD_3D& F, int blockNumber, int col, 
    COMPRESSION_DATA* data, FIELD_3D* damp)
{
  TIMER functionTimer(__FUNCTION__);

  // fetch parameters from data
  int nBits = data->get_nBits();
  int maxIterations = data->get_maxIterations();
  double percent = data->get_percent();

  double lower = 0.0;
  // QUESTION: how should we define upper?
  double upper = nBits;
  // arbitrarily set epsilon to be 0.5%
  double epsilon = 0.005;
  double gamma = 0.5 * (upper + lower);
  damp->toPower(gamma);
  
  // the total amount of energy in the Fourier space
  double totalEnergy = F.sumSq();
  FIELD_3D damped = ( F / (*damp) );
  damped.roundInt();

  double energyDiff = abs(totalEnergy - ( (*damp) * damped ).sumSq());
  double percentEnergy = 1.0 - (energyDiff / totalEnergy);
  int iterations = 0;
   
  while ( abs( percent - percentEnergy ) > epsilon && iterations < maxIterations) {

    if (percentEnergy < percent) { // too much damping; need to lower gamma
      upper = gamma;
      gamma = 0.5 * (upper + lower);

      // to the power of 1 / upper brings it back to the vanilla state, 
      // from which we raise it to the new gamma
      damp->toPower(gamma / upper);
    }

    else { // not enough damping; need to increase gamma
      lower = gamma;
      gamma = 0.5 * (upper + lower);

      // to the power of 1 / lower brings it back to the vanilla state, 
      // from which we raise it to the new gamma
      damp->toPower(gamma / lower);
    }

    // update percentEnergy
    damped = ( F / (*damp) );
    damped.roundInt();
    energyDiff = abs(totalEnergy - ( (*damp) * damped ).sumSq());
    percentEnergy =  1.0 - (energyDiff / totalEnergy);
    iterations++;
  }
  

  // fetch data to update gammaList
  MatrixXd* gammaListMatrix = data->get_gammaListMatrix();
  int numBlocks = data->get_numBlocks();
  int numCols = data->get_numCols();

  // if it's the first time TuneGamma is called in a chain, resize
  if (gammaListMatrix->cols() <= 0) { 
    gammaListMatrix->setZero(numBlocks, numCols);
  }

  (*gammaListMatrix)(blockNumber, col) = gamma;

}

////////////////////////////////////////////////////////
// takes a passed in FIELD_3D (which is intended to be
// the result of a DCT post-preprocess). calculates the best gamma value
// for a damping array. then damps by that array and
// quantizes the result to an integer. stores the 
// value of gamma for the damping.
////////////////////////////////////////////////////////

void EncodeBlock(const FIELD_3D& F, int blockNumber, int col, COMPRESSION_DATA* data, 
    INTEGER_FIELD_3D* quantized) 
{

  TIMER functionTimer(__FUNCTION__);

  // size the return value appropriately
  quantized->resizeAndWipe(F.xRes(), F.yRes(), F.zRes());

  // grab the pre-cached vanila damping array
  const FIELD_3D& dampingArray = data->get_dampingArray();

  // make a copy for modification during TuneGamma
  FIELD_3D damp = dampingArray;

  int numBlocks = data->get_numBlocks();
  assert(blockNumber >= 0 && blockNumber < numBlocks);
 
  // finds best gamma given the percent. updates gammaList
  // and updates damp 
  TuneGamma(F, blockNumber, col, data, &damp);

  // fill the return value with rounded damped entries
  RoundFieldToInt( (F / damp), quantized );

}

////////////////////////////////////////////////////////
// takes a passed in INTEGER_FIELD_3D (which is inteneded to
// be run-length decoded and unzigzagged) corresponding to
// a particulary blockNumber and column of the matrix. undoes
// the effects of damping and quantization as best it can.
////////////////////////////////////////////////////////
void DecodeBlock(const INTEGER_FIELD_3D& intBlock, int blockNumber, int col, 
    const DECOMPRESSION_DATA& decompression_data, FIELD_3D* decoded) 
{

  TIMER functionTimer(__FUNCTION__);

  int numBlocks = decompression_data.get_numBlocks();
  // make sure we are not accessing an invalid block
  assert( (blockNumber >= 0) && (blockNumber < numBlocks) );

  // we use u, v, w rather than x, y , z to indicate the spatial frequency domain
  const int uRes = intBlock.xRes();
  const int vRes = intBlock.yRes();
  const int wRes = intBlock.zRes();
  // size the decoded block appropriately and fill it with the block data
  decoded->resizeAndWipe(uRes, vRes, wRes);
  CastIntFieldToDouble(intBlock, decoded);

  // use the appropriate scale factor to decode
  const MatrixXd& sListMatrix = decompression_data.get_sListMatrix();
  double s = sListMatrix(blockNumber, col);
  double sInv = 1.0 / s;

  const MatrixXd& gammaListMatrix = decompression_data.get_gammaListMatrix();
  double gamma = gammaListMatrix(blockNumber, col);
  
  // dequantize by inverting the scaling by s and contracting by the 
  // appropriate gamma-modulated damping array
  const FIELD_3D& dampingArray = decompression_data.get_dampingArray();
  FIELD_3D damp = dampingArray;
  damp.toPower(gamma);

  // undo the dampings and preprocess
  (*decoded) *= dampingArray;
  (*decoded) *= sInv;
 
}


////////////////////////////////////////////////////////
// does the same operations as DecodeBlock, but with a passed
// in compression data parameter rather than decompression data 
// due to const poisoning, compression data cannot be marked const,
// but nonetheless it is not modified.
////////////////////////////////////////////////////////

void DecodeBlockWithCompressionData(const INTEGER_FIELD_3D& intBlock, 
  int blockNumber, int col, COMPRESSION_DATA* data, FIELD_3D* decoded) 
{ 
  TIMER functionTimer(__FUNCTION__);

  int numBlocks = data->get_numBlocks();
  // make sure we are not accessing an invalid block
  assert( (blockNumber >= 0) && (blockNumber < numBlocks) );

  // we use u, v, w rather than x, y , z to indicate the spatial frequency domain
  const int uRes = intBlock.xRes();
  const int vRes = intBlock.yRes();
  const int wRes = intBlock.zRes();
  // size the decoded block appropriately and fill it with the block data
  decoded->resizeAndWipe(uRes, vRes, wRes);
  CastIntFieldToDouble(intBlock, decoded);

  // use the appropriate scale factor to decode
  MatrixXd* sList = data->get_sListMatrix();
  double s = (*sList)(blockNumber, col);
  double sInv = 1.0 / s;
  MatrixXd* gammaList = data->get_gammaListMatrix();
  double gamma = (*gammaList)(blockNumber, col);
    
  // dequantize by inverting the scaling by s and contracting by the 
  // appropriate gamma-modulated damping array
  const FIELD_3D& dampingArray = data->get_dampingArray();
  FIELD_3D damp = dampingArray;
  damp.toPower(gamma);

  // undo the dampings and preprocess
  (*decoded) *= damp;
  (*decoded) *= sInv;

}

////////////////////////////////////////////////////////
// given a zigzagged integer buffer, write it to a binary
// file via run-length encoding. updates the blockLengthsMatrix. 
////////////////////////////////////////////////////////
void RunLengthEncodeBinary(const char* filename, int blockNumber, int col, 
    const VectorXi& zigzaggedArray, COMPRESSION_DATA* compression_data)
{ 
  TIMER functionTimer(__FUNCTION__);

  FILE* pFile;
  // open a file in append mode since we will call this function repeatedly
  pFile = fopen(filename, "ab+");    
  if (pFile == NULL) {
    perror ("Error opening file.");
  }
  else {

    // we use a C++ vector container for our data since we don't know
    // a priori how many entries it will have once encoded
    vector<int> dataList;          
    // reserve plenty of space just to be on the safe side
    dataList.reserve(2 * BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE);

    // initialize variables
    int data = 0;
    int runLength = 0;
    int encodedLength = 0;

    // assuming BLOCK_SIZE blocks
    int length = BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE;

    for (int i = 0; i < length; i++) {

      data = zigzaggedArray[i];

      // add the value once no matter what
      dataList.push_back(data);
      encodedLength++;

      // we already wrote one value, so runLength starts from 1
      runLength = 1;

      // if the current value and the next value agree, increment the run length.
      // don't allow the next value to go out of bounds
      while ( (i + 1 < length) && (zigzaggedArray[i] == zigzaggedArray[i + 1]) ) {
        runLength++;
        i++;
      }
      
      // we don't bother to write run lengths for singletons
      if (runLength > 1) {
        // use a single repeated value as an 'escape' to indicate a run
        dataList.push_back(data);
        
        // push the runLength to the data vector
        dataList.push_back(runLength);

        encodedLength += 2;
      }


    }
 
    // the size of the dataList vector is how long the encoded block will be
    // int encodedLength = dataList.size();

    // fetch the blockLengthsMatrix for updating
    MatrixXi* blockLengthsMatrix = compression_data->get_blockLengthsMatrix();
    int numBlocks = compression_data->get_numBlocks();
    int numCols = compression_data->get_numCols();

    // if the matrix isn't yet allocated, prellocate
    if (blockLengthsMatrix->cols() <= 0) {
      blockLengthsMatrix->setZero(numBlocks, numCols);
    }

    // update the appropriate entry
    (*blockLengthsMatrix)(blockNumber, col) = encodedLength;

    fwrite(dataList.data(), sizeof(int), encodedLength, pFile);
    // this write assumes that C++ vectors are stored in contiguous memory!

    fclose(pFile);
  }
}

////////////////////////////////////////////////////////
// decode a run-length encoded binary file and fill 
// a VectorXi with the contents.
////////////////////////////////////////////////////////

void RunLengthDecodeBinary(int* allData, int blockNumber, int col, 
    COMPRESSION_DATA* compression_data, VectorXi* parsedData)
{

  TIMER functionTimer(__FUNCTION__);

  int nBits = compression_data->get_nBits();
   
  MatrixXi* blockLengthsMatrix = compression_data->get_blockLengthsMatrix(); 
  int compressedBlockSize = (*blockLengthsMatrix)(blockNumber, col);
  assert(compressedBlockSize >= 0 && 
      compressedBlockSize <= 2 * BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE);


  MatrixXi* blockIndicesMatrix = compression_data->get_blockIndicesMatrix();
  int blockIndex = (*blockIndicesMatrix)(blockNumber, col);


  /* 
  // ***************************************************
  // for debugging
  
    cout << "compressedBlockSize: " << compressedBlockSize << endl;

    VECTOR block(compressedBlockSize);
    for (int i = 0; i < block.size(); i++) {
      block[i] = allData[blockIndex + i];
    }
    cout << "blockNumber " << blockNumber << ", column " << col << ": " << endl;
    cout << block << endl;

  // ***************************************************
  */


  int i = 0;
  int runLength = 1;
 
  parsedData->resize(BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE);
  int* data = parsedData->data();
  int j = 0;
  while (i < compressedBlockSize) {
    
    // write the value once
    data[j] = allData[blockIndex + i];

    /*
    // *************************************************** 
    // purely for debugging!
    if (j == 0) {
      if (data[j] != pow(2, nBits - 1) - 1) {
        cout << "Error: DC term parsed wrong!" << endl;
        cout << "DC term was thought to be: " << data[j] << endl;
        cout << "Block number is: " << blockNumber << endl;
        cout << "Column is: " << col << endl;
      }
    }
    // *************************************************** 
    */
    
    if ( (i + 1 < compressedBlockSize) && 
        allData[blockIndex + i] == allData[blockIndex + i + 1]) {
      i += 2;
      runLength = allData[blockIndex + i];

      assert(runLength > 1 && runLength <= BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE);

      std::fill(data + j + 1, data + j + 1 + runLength - 1, allData[blockIndex + i - 2]);
      j += (runLength - 1);

    }

    i++;
    j++;
  }
} 



////////////////////////////////////////////////////////
// Flattends an INTEGER_FIELD_3D through a zig-zag scan
// into a VectorXi. Since the scan always follows the same order,
// we precompute the zigzag scan array, pass it
// as a parameter, and then just do an index lookup
////////////////////////////////////////////////////////
void ZigzagFlatten(const INTEGER_FIELD_3D& F, const INTEGER_FIELD_3D& zigzagArray, 
    VectorXi* zigzagged) 
{

  TIMER functionTimer(__FUNCTION__);

  int xRes = F.xRes();
  int yRes = F.yRes();
  int zRes = F.zRes();
  assert(xRes == BLOCK_SIZE && yRes == BLOCK_SIZE && zRes == BLOCK_SIZE);

  zigzagged->resize(xRes * yRes * zRes);
  int totalCells = BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE;
  
  for (int i = 0; i < totalCells; i++) {
    int index = zigzagArray[i];
    (*zigzagged)[index] = F[i];
  }

}

////////////////////////////////////////////////////////
// Unflattens a VectorXi into an INTEGER_FIELD_3D. 
// uses precomputed zigzagArray and simple lookups. 
////////////////////////////////////////////////////////
void ZigzagUnflatten(const VectorXi& V, const INTEGER_FIELD_3D& zigzagArray, 
    INTEGER_FIELD_3D* unflattened) 
{

  TIMER functionTimer(__FUNCTION__);

  assert(V.size() == BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE);

  unflattened->resizeAndWipe(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
  
  int totalCells = BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE;
  for (int i = 0; i < totalCells; i++) {
    int index = zigzagArray[i];
    (*unflattened)[i] = V[index];
  }

}

////////////////////////////////////////////////////////
// reads from a binary file the SVD data and sets the 
// values inside compression data
////////////////////////////////////////////////////////
void ReadSVDData(const char* filename, COMPRESSION_DATA* data)
{
  
  FILE* pFile;
  pFile = fopen(filename, "rb");

  if (pFile == NULL) {
    perror("Error opening SVD data");
    exit(EXIT_FAILURE);
  }

  // fetch number of columns to preallocate the lists
  int numCols = data->get_numCols();
  vector<Vector3d>* singularList = data->get_singularList();
  vector<Matrix3d>* vList        = data->get_vList();

  // preallocate the correct size
  singularList->resize(numCols);
  vList->resize(numCols);

  // for each column, first read the singular values, then read 
  // the values in the V matrix. there are 3 singular values,
  // and the V matrices are all 3 x 3.
  for (int col = 0; col < numCols; col++) {
    fread((*singularList)[col].data(), sizeof(double), 3, pFile);
    fread((*vList)[col].data(), sizeof(double), 3 * 3, pFile);
  }
  
  fclose(pFile);
}

////////////////////////////////////////////////////////
// reads from a binary file into a buffer, and sets
// important initializations inside compression data
////////////////////////////////////////////////////////

int* ReadBinaryFileToMemory(const char* filename, 
    COMPRESSION_DATA* data) 
{
  TIMER functionTimer(__FUNCTION__);

  // initialize what we will return
  int* allData = NULL;

  FILE* pFile;

  pFile = fopen(filename, "rb");
  if (pFile == NULL) {
    perror("Error opening file.");
    exit(EXIT_FAILURE);
  }

  else {
    
    // build the damping array and zigzag arrays 
    data->set_dampingArray();
    data->set_zigzagArray();
 
    // read nBits and set it
    int nBits;
    fread(&nBits, 1, sizeof(int), pFile);
    data->set_nBits(nBits);
    cout << "nBits: " << nBits << endl;

    // read dims, numCols, and numBlocks
    int xRes, yRes, zRes;
    fread(&xRes, 1, sizeof(int), pFile);
    fread(&yRes, 1, sizeof(int), pFile);
    fread(&zRes, 1, sizeof(int), pFile);
    VEC3I dims(xRes, yRes, zRes);
    data->set_dims(dims);
    cout << "dims: " << dims << endl;

    int numCols, numBlocks;
    fread(&numCols, 1, sizeof(int), pFile);
    fread(&numBlocks, 1, sizeof(int), pFile);
    // set the decompression data accordingly
    data->set_numCols(numCols);
    data->set_numBlocks(numBlocks);
    cout << "numCols: " << numCols << endl;
    cout << "numBlocks: " << numBlocks << endl;
    
    // read in the sListMatrix and set the data
    int blocksXcols = numBlocks * numCols;
    MatrixXd* sListMatrix = data->get_sListMatrix();
    sListMatrix->resize(numBlocks, numCols);
    fread(sListMatrix->data(), blocksXcols, sizeof(double), pFile);
    
    // do the same for gammaListMatrix
    MatrixXd* gammaListMatrix = data->get_gammaListMatrix();
    gammaListMatrix->resize(numBlocks, numCols);
    fread(gammaListMatrix->data(), blocksXcols, sizeof(double), pFile);

    // do the same for the blockLengthsMatrix, except the data are ints
    MatrixXi* blockLengthsMatrix = data->get_blockLengthsMatrix();
    blockLengthsMatrix->resize(numBlocks, numCols);
    fread(blockLengthsMatrix->data(), blocksXcols, sizeof(int), pFile);

    // cout << "blockLengthsMatrix, column 0: " << endl;
    VectorXi blockLengths0 = blockLengthsMatrix->col(0);
    // cout << EIGEN::convertInt(blockLengths0) << endl;

    // store the total length of all blocks to be able to 
    // read in the full compressed data later
    int totalLength = blockLengthsMatrix->sum();
    // cout << "totalLength: " << totalLength << endl;

    // read in blockIndicesMatrix
    MatrixXi* blockIndicesMatrix = data->get_blockIndicesMatrix();
    blockIndicesMatrix->resize(numBlocks, numCols);
    fread(blockIndicesMatrix->data(), blocksXcols, sizeof(int), pFile);

    // cout << "blockIndicesMatrix, column 0: " << endl;
    // cout << EIGEN::convertInt(blockIndicesMatrix->col(0)) << endl; 

    // finally, read in the full compressed data
    allData = (int*) malloc(totalLength * sizeof(int));
    if (allData == NULL) {
      perror("Malloc failed to allocate allData!");
      exit(EXIT_FAILURE);
    }

    fread(allData, totalLength, sizeof(int), pFile);
  }

  return allData;
}

////////////////////////////////////////////////////////
// deletes a file if it already exists
////////////////////////////////////////////////////////
void DeleteIfExists(const char* filename)
{
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
// takes an input FIELD_3D at a particular matrix column
// which is the result of an SVD coordinate transform, compresses
// it according to the general scheme, and writes it to a binary file.
// meant to be called in a chain so that the binary file
// continues to grow. 
////////////////////////////////////////////////////////

void CompressAndWriteField(const char* filename, const FIELD_3D& F, int col,
    COMPRESSION_DATA* compression_data)
{
  TIMER functionTimer(__FUNCTION__);
 
  // fetch some compression data 
  int numBlocks = compression_data->get_numBlocks();
  int numCols = compression_data->get_numCols();

  const INTEGER_FIELD_3D& zigzagArray = compression_data->get_zigzagArray();
  MatrixXi* blockLengthsMatrix = compression_data->get_blockLengthsMatrix();

  // if it's the first time calling this routine in a chain, preallocate
  // the matrix
  if (blockLengthsMatrix->cols() <= 0) {
    blockLengthsMatrix->resize(numBlocks, numCols);
  }

  // subdivide F into blocks
  vector<FIELD_3D> blocks;
  GetBlocks(F, &blocks);

  // do the forward transform 
  UnitaryBlockDCT(1, &blocks);

  // initialize the relevant variables before looping through all the blocks
  VectorXi blockLengths(numBlocks);
  INTEGER_FIELD_3D intEncoded_i;
  VectorXi zigzagged_i;

  // loop through the blocks and apply the encoding procedure
  for (int i = 0; i < numBlocks; i++) {

    // rescales data and updates sList
    PreprocessBlock(&(blocks[i]), i, col, compression_data);

    // performs quantization and damping. updates gammaList
    EncodeBlock(blocks[i], i, col, compression_data, &intEncoded_i);

    // do the zigzag scan for run-length encoding
    ZigzagFlatten(intEncoded_i, zigzagArray, &zigzagged_i);

    // performs run-length encoding. updates blockLengthsMatrix. since
    // it opens 'filename' in append mode, it can be called in a chain
    RunLengthEncodeBinary(filename, i, col, zigzagged_i, compression_data);  
  }
  
}


////////////////////////////////////////////////////////
// build the block indices matrix from the block lengths matrix
////////////////////////////////////////////////////////
void BuildBlockIndicesMatrix(COMPRESSION_DATA* data)
{
  MatrixXi* blockLengths = data->get_blockLengthsMatrix();
  MatrixXi* blockIndices = data->get_blockIndicesMatrix();

  // copy the block lengths to start
  *blockIndices = (*blockLengths);

  // flatten column-wise into a vector
  blockIndices->resize(blockLengths->rows() * blockLengths->cols(), 1);

  // compute the cumulative sum
  VectorXi sum;
  ModifiedCumSum(blockIndices->col(0), &sum);

  // copy back into block indices
  memcpy(blockIndices->data(), sum.data(), sum.size() * sizeof(int));

  // reshape appropriately
  blockIndices->resize(blockLengths->rows(), blockLengths->cols());

}

////////////////////////////////////////////////////////
// build the block indices matrix from the block lengths matrix.
// uses explicitly passed in matrices for debugging!
////////////////////////////////////////////////////////
void BuildBlockIndicesMatrixDebug(const MatrixXi blockLengths, MatrixXi* blockIndices)
{
  TIMER functionTimer(__FUNCTION__);

  // copy the block lengths to start
  *blockIndices = blockLengths;

  // flatten column-wise into a vector
  blockIndices->resize(blockLengths.rows() * blockLengths.cols(), 1);

  // compute the cumulative sum 
  VectorXi sum; 
  ModifiedCumSum(blockIndices->col(0), &sum);

  // copy back into block indices
  memcpy(blockIndices->data(), sum.data(), sum.size() * sizeof(int));

  // reshape appropriately
  blockIndices->resize(blockLengths.rows(), blockLengths.cols()); 

}

////////////////////////////////////////////////////////
// given a row number and the dimensions, computes
// which block number we need for the decoder. populates
// blockIndex with the corresponding value as well.
////////////////////////////////////////////////////////
/*
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

    int u = x % BLOCK_SIZE;
    int v = y % BLOCK_SIZE;
    int w = z % BLOCK_SIZE;
    blockIndex = u + BLOCK_SIZE * v + BLOCK_SIZE * BLOCK_SIZE * w;
    
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
    int blockNumber = x/BLOCK_SIZE + (y/BLOCK_SIZE * (xRes/BLOCK_SIZE)) + (z/BLOCK_SIZE * (xRes/BLOCK_SIZE) * (yRes/BLOCK_SIZE));

    return blockNumber;
  }
*/
  
////////////////////////////////////////////////////////
// given a (row, col), decode it from the lossy matrix
////////////////////////////////////////////////////////
/*
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

  vector<int> decoded_runLength(BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE);
  VECTOR decoded_runLengthVector(BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE);
  INTEGER_FIELD_3D unzigzagged(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
  FIELD_3D decoded_block(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
 

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
*/

   
////////////////////////////////////////////////////////
// given a start row and numRows (typically 3), computes
// the submatrix given the compression data
////////////////////////////////////////////////////////
/*
void GetSubmatrixFast(int startRow, int numRows, MATRIX_COMPRESSION_DATA& data, MatrixXd& matrixToFill) {
     
    
  TIMER functionTimer(__FUNCTION__);
  
  // only leave this commented out if you don't mind skipping an assertion! 
  
  const DECOMPRESSION_DATA& decompression_dataX = data.get_decompression_dataX();
  int numCols = decompression_dataX.get_numCols();
  assert( matrixToFill.rows() == numRows && matrixToFill.cols() == numCols );

  for (int i = 0; i < numRows; i++) {
    GetRowFast(startRow + i, i, data, matrixToFill);
  }    
}
*/


// get a whole row from a compressed matrix 
/*
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
*/
 

////////////////////////////////////////////////////////
// generates the header information in the binary file
// need to include:
//
////////////////////////////////////////////////////////

void WriteMetaData(const char* filename, COMPRESSION_DATA& compression_data)
{ 

  TIMER functionTimer(__FUNCTION__);

    FILE* pFile;
    pFile = fopen(filename, "wb");
    if (pFile == NULL) {
      perror ("Error opening file.");
    }
    else {
      
      // write nBits to the binary file
      int nBits = compression_data.get_nBits();
      fwrite(&nBits, sizeof(int), 1, pFile);

      // write dims, numCols, and numBlocks
      const VEC3I& dims = compression_data.get_dims();
      int xRes = dims[0];
      int yRes = dims[1];
      int zRes = dims[2];
      fwrite(&xRes, sizeof(int), 1, pFile);
      fwrite(&yRes, sizeof(int), 1, pFile);
      fwrite(&zRes, sizeof(int), 1, pFile);
      int numCols = compression_data.get_numCols();
      int numBlocks = compression_data.get_numBlocks();
      int blocksXcols = numBlocks * numCols;
      fwrite(&numCols, sizeof(int), 1, pFile);
      fwrite(&numBlocks, sizeof(int), 1, pFile);
       
      MatrixXd* sListMatrix = compression_data.get_sListMatrix();
      assert( sListMatrix->rows() * sListMatrix->cols() == blocksXcols );

      // write the matrix data for sList, blockLengths, and blockIndices.
      // note that Eigen uses column-major format!
      fwrite(sListMatrix->data(), sizeof(double), blocksXcols, pFile); 

      MatrixXd* gammaListMatrix = compression_data.get_gammaListMatrix();
      assert( gammaListMatrix->rows() * gammaListMatrix->cols() == blocksXcols );

      fwrite(gammaListMatrix->data(), sizeof(double), blocksXcols, pFile);

      MatrixXi* blockLengthsMatrix = compression_data.get_blockLengthsMatrix();
      assert( blockLengthsMatrix->rows() * blockLengthsMatrix->cols() == blocksXcols);

      fwrite(blockLengthsMatrix->data(), sizeof(int), blocksXcols, pFile);
      
      MatrixXi* blockIndicesMatrix = compression_data.get_blockIndicesMatrix();
      assert( blockIndicesMatrix->rows() * blockIndicesMatrix->cols() == blocksXcols);

      fwrite(blockIndicesMatrix->data(), sizeof(int), blocksXcols, pFile);

      fclose(pFile);
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
// write the singular values and V matrices to a binary file
////////////////////////////////////////////////////////
void WriteSVDData(const char* filename, COMPRESSION_DATA* data)
{
  FILE* pFile;
  pFile = fopen(filename, "wb");    
  if (pFile == NULL) {
    perror ("Error opening file.");
  }
  else {
    vector<Vector3d>* singularList = data->get_singularList();
    vector<Matrix3d>* vList        = data->get_vList();

    int numCols = data->get_numCols();

    assert(singularList->size() == numCols && vList->size() == numCols);

    // for each column, first write the singular values, then write
    // the values in the V matrix. there are 3 singular values,
    // and the V matrices are all 3 x 3.
    for (int col = 0; col < numCols; col++) {
      fwrite((*singularList)[col].data(), sizeof(double), 3, pFile); 
      fwrite((*vList)[col].data(), sizeof(double), 3 * 3, pFile);
    }
  }
  fclose(pFile);
}

////////////////////////////////////////////////////////
// compress all of the scalar field components
// of a matrix (which represents a vector field) and write them to
// a binary file. applies svd coordinate transform first.
////////////////////////////////////////////////////////

// *****************************************************
// Still in the process of editing this!
// *****************************************************

void CompressAndWriteMatrixComponents(const char* filename, const MatrixXd& U,  
      COMPRESSION_DATA* data0, COMPRESSION_DATA* data1, COMPRESSION_DATA* data2) 
{
  TIMER functionTimer(__FUNCTION__);
  
  // wipe any pre-existing binary file of the same name, since we will be opening
  // in append mode! 
  DeleteIfExists(filename);

  // write to component X, Y, and Z accordingly
  // initialize strings 0, 1, and 2 for the final result
  string filenameX(filename); filenameX += 'X';
  string filename0(filename); filename0 += '0';
  DeleteIfExists(filenameX.c_str());
  DeleteIfExists(filename0.c_str());
  
  string filenameY(filename); filenameY += 'Y';
  string filename1(filename); filename1 += '1';
  DeleteIfExists(filenameY.c_str());
  DeleteIfExists(filename1.c_str());
  
  string filenameZ(filename); filenameZ += 'Z';
  string filename2(filename); filename2 += '2';
  DeleteIfExists(filenameZ.c_str());
  DeleteIfExists(filename2.c_str());

  // get useful compression data. it should be the same across all three,
  // so just fetch it from data0
  const VEC3I& dims = data0->get_dims();
  int xRes = dims[0];
  int yRes = dims[1];
  int zRes = dims[2];
  int numCols = U.cols();

  for (int col = 0; col < numCols; col++) {  

    // for each column, grab the vector field that it corresponds to
    VECTOR3_FIELD_3D V(U.col(col), xRes, yRes, zRes);
    
    // do the svd coordinate transform in place and update the data for 
    // vList and singularList. only data0 contains these!  
    TransformVectorFieldSVDCompression(&V, data0);
    
   
    // write the components to an (appended) binary file 

    CompressAndWriteField(filenameX.c_str(), V.scalarField(0), col, data0);
    CompressAndWriteField(filenameY.c_str(), V.scalarField(1), col, data1);
    CompressAndWriteField(filenameZ.c_str(), V.scalarField(2), col, data2);

    // progress printout for the impatient user
    PrintProgress(col, numCols);
  }
  
  // once we've gone through each column, we can write the full SVD data
  WriteSVDData("U.preadvect.SVD.data", data0);

  // we can also build the block indices matrix for each component
  BuildBlockIndicesMatrix(data0);
  BuildBlockIndicesMatrix(data1);
  BuildBlockIndicesMatrix(data2);

  // write the metadata for each component one at a time
  const char* metafile = "metadata.bin"; 
  WriteMetaData(metafile, *data0);

  // appends the metadata as a header to the main binary file and pipes them into final_string
  PrefixBinary(metafile, filenameX, filename0);
 
  // removes the now-redundant files 
  CleanUpPrefix(metafile, filenameX.c_str());

  // do the same for component 1
  WriteMetaData(metafile, *data1);
  PrefixBinary(metafile, filenameY, filename1);
  CleanUpPrefix(metafile, filenameY.c_str());

  // do the same for component 2
  WriteMetaData(metafile, *data2);
  PrefixBinary(metafile, filenameZ, filename2);
  CleanUpPrefix(metafile, filenameZ.c_str());
}

////////////////////////////////////////////////////////
// print four different percents for how far along each
// column we are
////////////////////////////////////////////////////////
void PrintProgress(int col, int numCols) 
{
  // percent total progress
  double percent = col / (double) numCols;

  // four checkpoints
  int checkPoint1 = (numCols - 1) / 4;
  int checkPoint2 = (numCols - 1) / 2;
  int checkPoint3 = (3 * (numCols - 1)) / 4;
  int checkPoint4 = numCols - 1;

  // if we're at any of the checkpoints, print the progress
  if (col == checkPoint1 || col == checkPoint2 || col == checkPoint3 || col == checkPoint4) {
    cout << "    Percent: " << percent << flush;
    if (col == checkPoint4) {
      cout << endl;
    }
  } 
}


////////////////////////////////////////////////////////
// decode an entire scalar field of a particular column from matrix compression data
////////////////////////////////////////////////////////
void DecodeScalarField(COMPRESSION_DATA* compression_data, int* allData, 
    int col, FIELD_3D* decoded)
{
  TIMER functionTimer(__FUNCTION__);

  // get the dims from data and construct a field of the appropriate size
  const VEC3I& dims = compression_data->get_dims();

  // fill in the paddings
  VEC3I newDims;
  GetPaddings(dims, &newDims);

  // update the resolutions
  newDims += dims;

  // fetch numBlocks and precomputed zigzag table
  int numBlocks = compression_data->get_numBlocks();
  const INTEGER_FIELD_3D& zigzagArray = compression_data->get_zigzagArray();

  // container for the encoded blocks
  vector<FIELD_3D> blocks(numBlocks);

  // variable to store decoded run-length blocks
  VectorXi runLengthDecoded;

  for (int blockNumber = 0; blockNumber < numBlocks; blockNumber++) {

    // decode the run length scheme
    RunLengthDecodeBinary(allData, blockNumber, col, 
        compression_data, &runLengthDecoded);

    // undo the zigzag scan
    INTEGER_FIELD_3D unflattened;
    ZigzagUnflatten(runLengthDecoded, zigzagArray, &unflattened);

    // undo the scaling from the quantizer and push to the block container
    FIELD_3D decodedBlock;
    DecodeBlockWithCompressionData(unflattened, blockNumber, col,
        compression_data, &decodedBlock);
    blocks[blockNumber] = decodedBlock; 
  }
  
  // perform the IDCT on each block
  // -1 <-- inverse
  UnitaryBlockDCT(-1, &blocks);
  
  // reassemble the blocks into one large scalar field
  FIELD_3D padded_result(newDims[0], newDims[1], newDims[2]);
  AssimilateBlocks(newDims, blocks, &padded_result);

  // strip the padding
  *decoded = padded_result.subfield(0, dims[0], 0, dims[1], 0, dims[2]); 
 
}

////////////////////////////////////////////////////////
// decode an entire scalar field of a particular column from matrix compression data
// *without* going back to the spatial domain (or the SVD transform). leave them
// in a list of blocks as well.
////////////////////////////////////////////////////////
void DecodeScalarFieldEigen(COMPRESSION_DATA* compression_data, int* allData, 
    int col, vector<VectorXd>* decoded)
{
  TIMER functionTimer(__FUNCTION__);

  // get the dims from data and construct a field of the appropriate size
  const VEC3I& dims = compression_data->get_dims();

  // fill in the paddings
  VEC3I newDims;
  GetPaddings(dims, &newDims);

  // update the resolutions
  newDims += dims;

  // fetch numBlocks and precomputed zigzag table
  int numBlocks = compression_data->get_numBlocks();
  const INTEGER_FIELD_3D& zigzagArray = compression_data->get_zigzagArray();

  // resize the container we will return
  decoded->resize(numBlocks);

  // variable to store decoded run-length blocks
  VectorXi runLengthDecoded;

  for (int blockNumber = 0; blockNumber < numBlocks; blockNumber++) {

    // decode the run length scheme
    RunLengthDecodeBinary(allData, blockNumber, col, 
        compression_data, &runLengthDecoded);

    // undo the zigzag scan
    INTEGER_FIELD_3D unflattened;
    ZigzagUnflatten(runLengthDecoded, zigzagArray, &unflattened);

    // undo the scaling from the quantizer and push to the block container
    FIELD_3D decodedBlock;
    DecodeBlockWithCompressionData(unflattened, blockNumber, col,
        compression_data, &decodedBlock);
    (*decoded)[blockNumber] = decodedBlock.flattenedEigen(); 
  }
}

////////////////////////////////////////////////////////
// reconstructs a lossy original of a scalar field
// which had been written to a binary file and now loaded 
// into a int buffer. returns it in a vector of its blocks,
// each flattened out into a VectorXd. this is for use
// in projection/unprojection
////////////////////////////////////////////////////////
/*
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
  INTEGER_FIELD_3D unzigzagged(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
  vector<int> runLengthDecoded(BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE);
  VECTOR runLengthDecodedVec(BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE);

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
  // DoSmartBlockDCT(blocks, -1);
  // ***NEW: do the unitary inverse here!***
  // ******************************************
  DoSmartUnitaryBlockDCT(blocks, -1);


  // this part is inefficient---we should implement a DCT that operates on VectorXd so that
  // we don't have to loop through all the blocks again
  for (int blockNumber = 0; blockNumber < numBlocks; blockNumber++) {
    VectorXd blockEigen= blocks[blockNumber].flattenedEigen();
    toFill[blockNumber] = blockEigen;
  } 
} 
*/

////////////////////////////////////////////////////////
// reconstructs a lossy original of a scalar field
// *BUT STILL IN THE FOURIER SPACE*
// which had been written to a binary file and now loaded 
// into a int buffer. returns it in a vector of its blocks,
// each flattened out into a VectorXd. this is for use
// in projection/unprojection
////////////////////////////////////////////////////////
/*
void DecodeScalarFieldWithoutTransformEigenFast(const DECOMPRESSION_DATA& decompression_data, int* const& allData, int col, vector<VectorXd>& toFill) {
  TIMER functionTimer(__FUNCTION__);

  // get the dims from data and construct a field of the appropriate size
  const VEC3I& dims = decompression_data.get_dims();
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
  INTEGER_FIELD_3D unzigzagged(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
  vector<int> runLengthDecoded(BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE);
  VECTOR runLengthDecodedVec(BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE);

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
  
  // DO NOT perform the IDCT on each block

  // this part is inefficient---we should implement a DCT that operates on VectorXd so that
  // we don't have to loop through all the blocks again
  for (int blockNumber = 0; blockNumber < numBlocks; blockNumber++) {
    toFill[blockNumber] = blocks[blockNumber].flattenedEigen();
  } 
} 
*/

////////////////////////////////////////////////////////
// reconstructs a lossy original of a scalar field
// which had been written to a binary file and now loaded 
// into a int buffer. returns it in a vector of its blocks,
// each flattened out into a VectorXd. this is for use
// in projection/unprojection
////////////////////////////////////////////////////////
/*
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
    INTEGER_FIELD_3D unzigzagged(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
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
*/


////////////////////////////////////////////////////////
// uses DecodeScalarField three times to reconstruct
// a lossy vector field
////////////////////////////////////////////////////////

void DecodeVectorField(MATRIX_COMPRESSION_DATA* data, int col, 
    VECTOR3_FIELD_3D* decoded) 
{
  TIMER functionTimer(__FUNCTION__);
  
  COMPRESSION_DATA* compression_dataX = data->get_compression_dataX();
  COMPRESSION_DATA* compression_dataY = data->get_compression_dataY();
  COMPRESSION_DATA* compression_dataZ = data->get_compression_dataZ();

  const VEC3I& dims = compression_dataX->get_dims();
  int xRes = dims[0];
  int yRes = dims[1];
  int zRes = dims[2];

  int* allDataX = data->get_dataX();
  int* allDataY = data->get_dataY();
  int* allDataZ = data->get_dataZ();

  FIELD_3D scalarX, scalarY, scalarZ; 
  DecodeScalarField(compression_dataX, allDataX, col, &scalarX);
  DecodeScalarField(compression_dataY, allDataY, col, &scalarY);
  DecodeScalarField(compression_dataZ, allDataZ, col, &scalarZ);

  // copy the data into the resuting vector field data structure
  (*decoded) = VECTOR3_FIELD_3D(scalarX.data(), scalarY.data(), scalarZ.data(), 
      xRes, yRes, zRes);

  vector<Matrix3d>* vList = compression_dataX->get_vList();
  UntransformVectorFieldSVD((*vList)[col], decoded);

}



////////////////////////////////////////////////////////
// uses the decode vector field on each column to reconstruct
// a lossy full matrix  
////////////////////////////////////////////////////////
/*
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
*/

//////////////////////////////////////////////////////////////////////
// unproject the reduced coordinate into the peeled cells in this field 
// using compression data
//////////////////////////////////////////////////////////////////////
/*
void PeeledCompressedUnproject(VECTOR3_FIELD_3D& V, MATRIX_COMPRESSION_DATA& U_data, const VectorXd& q) {
  TIMER functionTimer(__FUNCTION__);

  int xRes = V.xRes();
  int yRes = V.yRes();
  int zRes = V.zRes();
  DECOMPRESSION_DATA dataX = U_data.get_decompression_dataX();
  int totalColumns = dataX.get_numCols();
  const VEC3I& dims = dataX.get_dims();

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
*/

//////////////////////////////////////////////////////////////////////
// compute the block-wise dot product between two lists and sum them into one
// large dot product
//////////////////////////////////////////////////////////////////////
double GetDotProductSum(const vector<VectorXd>& Vlist, const vector<VectorXd>& Wlist) {

  assert(Vlist.size() == Wlist.size());

  int size = Vlist.size();

  double dotProductSum = 0.0;
  for (int i = 0; i < size; i++) {
    double dotProduct_i = Vlist[i].dot(Wlist[i]);
    dotProductSum += dotProduct_i;
  }

  return dotProductSum;

}


/*
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
*/ 
 
//////////////////////////////////////////////////////////////////////
// helper function for frequency domain projection. transforms 
// V first by the SVD and then DCT. fills up the three vectors
// with the three components.
//////////////////////////////////////////////////////////////////////
void TransformSVDAndDCT(int col, VECTOR3_FIELD_3D* V, 
    MATRIX_COMPRESSION_DATA* U_data, 
    vector<VectorXd>* Xpart, vector<VectorXd>* Ypart, vector<VectorXd>* Zpart) 
{
  COMPRESSION_DATA* dataX = U_data->get_compression_dataX();
  vector<Matrix3d>* vList = dataX->get_vList();

  TransformVectorFieldSVDCached(&(*vList)[col], V);

  FIELD_3D V_X, V_Y, V_Z;
  GetScalarFields(V->peelBoundary(), &V_X, &V_Y, &V_Z);

  GetBlocksEigen(V_X, Xpart);
  GetBlocksEigen(V_Y, Ypart);
  GetBlocksEigen(V_Z, Zpart);
  
  UnitaryBlockDCTEigen(1, Xpart);
  UnitaryBlockDCTEigen(1, Ypart);
  UnitaryBlockDCTEigen(1, Zpart);

}

//////////////////////////////////////////////////////////////////////
// projection, implemented in the frequency domain
//////////////////////////////////////////////////////////////////////
void PeeledCompressedProjectTransform(const VECTOR3_FIELD_3D& V, 
    MATRIX_COMPRESSION_DATA* U_data, VectorXd* q)
{
  TIMER functionTimer(__FUNCTION__);

  // fetch the compression data and the full data buffer for each component
  COMPRESSION_DATA* dataX = U_data->get_compression_dataX();
  int* allDataX = U_data->get_dataX();
  COMPRESSION_DATA* dataY = U_data->get_compression_dataY();
  int* allDataY = U_data->get_dataY();
  COMPRESSION_DATA* dataZ = U_data->get_compression_dataZ();
  int* allDataZ = U_data->get_dataZ();

  // preallocate the resulting vector
  int totalColumns = dataX->get_numCols();
  q->resize(totalColumns);

  vector<VectorXd> Xpart, Ypart, Zpart;
  vector<VectorXd> blocks;
  VECTOR3_FIELD_3D Vcopy = V;

  for (int col = 0; col < totalColumns; col++) {
    // transform V with an SVD and a DCT to mimic compression
    TransformSVDAndDCT(col, &Vcopy, U_data, &Xpart, &Ypart, &Zpart); 

    double totalSum = 0.0;
    DecodeScalarFieldEigen(dataX, allDataX, col, &blocks);
    totalSum += GetDotProductSum(blocks, Xpart);

    DecodeScalarFieldEigen(dataY, allDataY, col, &blocks);
    totalSum += GetDotProductSum(blocks, Ypart);

    DecodeScalarFieldEigen(dataZ, allDataZ, col, &blocks);
    totalSum += GetDotProductSum(blocks, Zpart);

    (*q)[col] = totalSum;
 
  }

}

