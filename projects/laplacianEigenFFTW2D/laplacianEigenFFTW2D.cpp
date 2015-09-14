#include "EIGEN.h"

#include <cmath>

#include "FIELD_2D.h"
#include "FIELD_3D.h"
#include "VECTOR3_FIELD_3D.h"
#include "VECTOR3_FIELD_2D.h"
#include "VEC3.h"
#include "MATRIX.h"
#include "TENSOR3.h"
#include "SPARSE_MATRIX.h"

#ifdef __APPLE__
#include <GLUT/glut.h>
//#include <GL/freeglut.h>
#else
#include <GL/gl.h> // OpenGL itself.
#include <GL/glu.h> // GLU support library.
#include <GL/glut.h> // GLUT support library.
#endif

#include <iostream>
#include <QUICKTIME_MOVIE.h>
#include "MERSENNETWISTER.h"
#include "SIMPLE_PARSER.h"

using namespace std;

enum VIEWING {SPATIAL, REAL, IM};
VIEWING whichViewing = SPATIAL;

// the original field data
//VECTOR3_FIELD_3D vectorField;
//VECTOR3_FIELD_3D vectorFieldNormalized;

// the distance field data
FIELD_2D distanceField;

// z slice currently viewing viewed
int zSlice = -1;

string windowLabel("FIELD_3D Viewer");

// scaled, biased and manipulated field
FIELD_2D viewingField;

// FFT of the original field
FIELD_2D fftReal;
FIELD_2D fftIm;

int xScreenRes = 800;
int yScreenRes = 800;
int xMouse, yMouse;
int mouseButton;
int mouseState;
int xField = -1;
int yField = -1;
float zoom = 1.0;

float scale = 1.0;
float bias = 0.0;

bool drawingGrid = false;
bool useAbsolute = false;
bool useLog = false;
bool normalized = false;
float oldScale = 1.0;
float oldBias = 0.0;

VEC3F eyeCenter(0.5, 0.5, 1);
VECTOR3_FIELD_3D centers;
VECTOR3_FIELD_3D displayCenters;

VECTOR3_FIELD_2D vectorField2D; 
vector<VEC3F> particles;

string path("./data/temp/");

//int basisRank = 16;
int basisRank = 9;
//int basisRank = 17;
Real viscosity = 0;
int xRes = 64;
int yRes = 64;

MATRIX velocityU;
MATRIX vorticityU;
TENSOR3 C;
VECTOR w;
VECTOR wDot;
Real dt = 0.0001;
//Real dt = 0.01;
vector<pair<int, int> > ijPairs;
map<pair<int, int>, int > ijPairsReverse;

vector<FIELD_2D> velocityFFTX;
vector<FIELD_2D> velocityFFTY;

vector<int> xDeltaIndices;
vector<int> yDeltaIndices;
vector<Real> xMagnitudes;
vector<Real> yMagnitudes;

void updateViewingTexture();

FIELD_2D xHat;
FIELD_2D yHat;

fftw_plan xPlanIDCT;
fftw_plan yPlanIDCT;
fftw_plan xPlanIDST;
fftw_plan yPlanIDST;
fftw_plan xPlanIDSTyIDCT;
fftw_plan yPlanIDCTyIDST;

int frames = 0;

enum RECONSTRUCTION_METHOD { FFT_RECON, MATRIX_VECTOR_RECON, ANALYTIC_RECON };

RECONSTRUCTION_METHOD reconstructionMethod = FFT_RECON;

vector<SPARSE_MATRIX> sparseC;
VECTOR3_FIELD_2D eigenfield;

QUICKTIME_MOVIE movie;
bool captureMovie = false;
//bool captureMovie = true;

///////////////////////////////////////////////////////////////////////
// reconstruct the velocity field using FFT
///////////////////////////////////////////////////////////////////////
void reconstructSparseFFT()
{
  //TIMER functionTimer(__FUNCTION__);
  
  xHat.clear();
  yHat.clear();

  // build a frequency-domain x velocity
  for (int i = 0; i < w.size(); i++)
    xHat[xDeltaIndices[i]] = xMagnitudes[i] * w[i];
  
  // build a frequency-domain y velocity
  for (int i = 0; i < w.size(); i++)
    yHat[yDeltaIndices[i]] = yMagnitudes[i] * w[i];

  //TIMER FFTTimer("FFT calls");
  // build the spatial version

  // try it without plans
  //xHat.xIDST();
  //xHat.yIDCT();

  // try it with plans
  //xHat.xIDST(xPlanIDST);
  //xHat.yIDCT(yPlanIDCT);

  // try both at once
  //xHat.xIDSTyIDCT();
  xHat.xIDSTyIDCT(xPlanIDSTyIDCT);

  //yHat.xIDCT();
  //yHat.yIDST();
  //yHat.xIDCT(xPlanIDCT);
  //yHat.yIDST(yPlanIDST);
  //yHat.xIDCTyIDST();
  yHat.xIDCTyIDST(yPlanIDCTyIDST);

  //TIMER setter("Set timer");
  vectorField2D.setX(xHat);
  vectorField2D.setY(yHat);
}

///////////////////////////////////////////////////////////////////////
// reconstruct the velocity field using FFT
///////////////////////////////////////////////////////////////////////
void reconstructFFT()
{
  TIMER functionTimer(__FUNCTION__);

  // build a frequency-domain x velocity
  FIELD_2D xHat(xRes, yRes);
  for (int x = 0; x < velocityFFTX.size(); x++)
    xHat += velocityFFTX[x] * w[x];
  
  // build a frequency-domain x velocity
  FIELD_2D yHat(xRes, yRes);
  for (int x = 0; x < velocityFFTY.size(); x++)
    yHat += velocityFFTY[x] * w[x];

  // build the spatial version
  xHat.xIDST();
  xHat.yIDCT();

  yHat.xIDCT();
  yHat.yIDST();

  vectorField2D.setX(xHat);
  vectorField2D.setY(yHat);
}

///////////////////////////////////////////////////////////////////////
// step the system forward in time
///////////////////////////////////////////////////////////////////////
void stepEigenfunctions()
{
  TIMER functionTimer(__FUNCTION__);
  Real e1 = w.dot(w);

  //TIMER tensorMultiply("Tensor multiply");
#if 0
  for (int k = 0; k < basisRank; k++)
  {
    MATRIX& slab = C.slab(k);
    wDot[k] = w.dot(slab * w);
  }
#else
  for (int k = 0; k < basisRank; k++)
  {
    SPARSE_MATRIX& slab = sparseC[k];
    //wDot[k] = w.dot(slab * w);
    wDot[k] = w.dot(slab.staticMultiply(w));
  }
#endif
  //tensorMultiply.stop();

  w += dt * wDot;

  Real e2 = w.dot(w);

  if (e2 > 0)
    w *= sqrt(e1 / e2);

  int k = 0;
  for (int k = 0; k < basisRank; k++)
  {
    int i = ijPairs[k].first;
    int j = ijPairs[k].second;

    Real lambda = -(i * i + j * j);

    //const Real viscosity = 1.0;
    //const Real viscosity = 0.0;

    // diffuse
    w[k] *= exp(lambda * dt * viscosity);
  }

  // goose the lowest mode?
  //cout << "w0: " << w[0] << endl;
  //w[0] += 1;

  TIMER reconstructTimer("Velocity Reconstruction");

  switch (reconstructionMethod)
  {
    case FFT_RECON:
      //reconstructFFT();
      reconstructSparseFFT();
      break;
    case MATRIX_VECTOR_RECON:
      {
        VECTOR final = velocityU * w;
        vectorField2D.unflatten(final);
      }
      break;
    case ANALYTIC_RECON:
      {
        // build directly analytically, so takes up zero memory
        vectorField2D.clear();
        for (int x = 0; x < ijPairs.size(); x++)
        {
          int i = ijPairs[x].first;
          int j = ijPairs[x].second;
          eigenfield.eigenfunctionFFTW(i,j);
          eigenfield *= w[x];

          vectorField2D += eigenfield;
        }
      }
      break;
  } 
  vectorField2D.stompBorder();
  reconstructTimer.stop();
 
  // visualize...
  //distanceField = vectorField2D.LIC();
  //updateViewingTexture();

  frames++;
}

///////////////////////////////////////////////////////////////////////
// build a list of the i,j pairs used to build the eigenfunctions
///////////////////////////////////////////////////////////////////////
void buildIJPairs(int rank)
{
  ijPairs.clear();
  cout << " Building pairs " << flush;

  int columnsMade = 0;
  for (int i = 1; i < rank; i++)
    for (int j = 1; j <= i; j++)
    {
      cout << " (" << i << "," << j << ") " << flush;
      pair<int, int> ij(i,j);
      ijPairs.push_back(ij);
      ijPairsReverse[ij] = ijPairs.size() - 1;
      columnsMade++;
      
      if (columnsMade == rank)
      {
        i = rank;
        break;
      }
      if (i == j) continue;

      pair<int, int> ji(j,i);
      ijPairs.push_back(ji);
      ijPairsReverse[ji] = ijPairs.size() - 1;
      columnsMade++;

      if (columnsMade == rank)
      {
        i = rank;
        break;
      }
    }
  cout << " done. " << endl;
}

///////////////////////////////////////////////////////////////////////
// build a list of the i,j pairs used to build the eigenfunctions
///////////////////////////////////////////////////////////////////////
void buildIJPairsPerfectSquare(int rank)
{
  ijPairs.clear();

  int columnsMade = 0;
  for (int i = 1; i <= (int)sqrt(rank); i++)
    for (int j = 1; j <= (int)sqrt(rank); j++)
    {
      pair<int, int> ij(i,j);
      ijPairs.push_back(ij);

      ijPairsReverse[ij] = ijPairs.size() - 1;
      columnsMade++;

      /*
      if (columnsMade == rank)
      {
        i = 2 * rank;
        j = 2 * rank;
      }
      */
    }
}

///////////////////////////////////////////////////////////////////////
// Build up the velocity basis matrix
///////////////////////////////////////////////////////////////////////
void buildVelocityBasis(int rank)
{
  vector<VECTOR> columns;
  cout << " Making eigenfunctions " << flush;
  for (int x = 0; x < ijPairs.size(); x++)
  {
    int i = ijPairs[x].first;
    int j = ijPairs[x].second;
    //vectorField2D.eigenfunction(i,j);
    cout << "(" << i << "," << j << ") " << flush;
    vectorField2D.eigenfunctionFFTW(i,j);
    VECTOR column = vectorField2D.flatten();
    columns.push_back(column);
  }
  velocityU = MATRIX(columns);
  velocityU.write("./data/velocityU.matrix");
  cout << "done." << endl;
}

///////////////////////////////////////////////////////////////////////
// Build up the velocity basis matrix
///////////////////////////////////////////////////////////////////////
void buildVelocityBasisPerfectSquare(int rank)
{
  vector<VECTOR> columns;
  int columnsMade = 0;
  for (int i = 1; i < rank; i++)
    for (int j = 1; j < rank; j++)
    {
      cout << " Making eigenfunction (" << i << ", " << j << ")" << endl;
      vectorField2D.eigenfunction(i,j);
      VECTOR column = vectorField2D.flatten();
      columns.push_back(column);
      columnsMade++;

      /*
      if (columnsMade == rank)
      {
        i = 2 * rank;
        j = 2 * rank;
      }
      */
    }
  velocityU = MATRIX(columns);
  velocityU.write("./data/velocityU.matrix");
}

///////////////////////////////////////////////////////////////////////
// Build up the vorticity basis matrix
///////////////////////////////////////////////////////////////////////
void buildVorticityBasis(int rank)
{
  vector<VECTOR> columns;
  for (int x = 0; x < ijPairs.size(); x++)
  {
    int i = ijPairs[x].first;
    int j = ijPairs[x].second;
    cout << " Making vorticity function (" << i << ", " << j << ")" << endl;
    vectorField2D.vorticity(i,j);
    VECTOR column = vectorField2D.flatten();
    columns.push_back(column);
  }
  vorticityU = MATRIX(columns);
  vorticityU.write("./data/vorticityU.matrix");
}

///////////////////////////////////////////////////////////////////////
// Build up the vorticity basis matrix
///////////////////////////////////////////////////////////////////////
void buildVorticityBasisPerfectSquare(int rank)
{
  vector<VECTOR> columns;
  int columnsMade = 0;
  for (int i = 1; i <= sqrt(rank); i++)
    for (int j = 1; j <= sqrt(rank); j++)
    {
      cout << " Making vorticity function (" << i << ", " << j << ")" << endl;
      vectorField2D.vorticity(i,j);
      VECTOR column = vectorField2D.flatten();
      columns.push_back(column);
      columnsMade++;

      /*
      if (columnsMade == rank)
      {
        i = 2 * rank;
        j = 2 * rank;
      }
      */
    }
  vorticityU = MATRIX(columns);
  vorticityU.write("./data/vorticityU.matrix");
}

///////////////////////////////////////////////////////////////////////
// advect vorticity using the velocity basis
///////////////////////////////////////////////////////////////////////
VECTOR advect(pair<int, int> iVelocity, pair<int, int> jVorticity)
{
  VECTOR3_FIELD_2D advected(vectorField2D);
  advected.clear(); 

  const int i1 = iVelocity.first;
  const int i2 = iVelocity.second;
  const int j1 = jVorticity.first;
  const int j2 = jVorticity.second;
  const Real iLambda = -(i1 * i1 + i2 * i2);
  const Real jLambda = -(j1 * j1 + j2 * j2);

  const int xRes = advected.xRes();
  const int yRes = advected.yRes();
  int index = 0;
  for (int y = 0; y < yRes; y++)
    for (int x = 0; x < xRes; x++, index++)
    {
      VEC3F center = advected.cellCenter(x,y);

      Real xReal = center[0];
      Real yReal = center[1];

      Real first = i1 * j2 * cos(i1 * xReal) * cos(j2 * yReal) * sin(j1 * xReal) * sin(i2 * yReal);
      first *= 1.0 / iLambda;
      Real second = i2 * j1 * cos(j1 * xReal) * cos(i2 * yReal) * sin(i1 * xReal) * sin(j2 * yReal);
      second *= 1.0 / iLambda;

      advected(x,y)[2] = first - second;
    }

  return advected.flatten();
}

///////////////////////////////////////////////////////////////////////
// advect vorticity using the velocity basis
///////////////////////////////////////////////////////////////////////
VECTOR advectVorticityVelocity(const VECTOR& vorticity, const VECTOR& velocity)
{
  // take the cross-product
  assert(vorticity.size() % 3 == 0);
  const int size = vorticity.size();
  VECTOR cross(size);
  for (int x = 0; x < size; x += 3)
  {
    VEC3F currentVorticity;
    currentVorticity[0] = vorticity[x];
    currentVorticity[1] = vorticity[x + 1];
    currentVorticity[2] = vorticity[x + 2];

    VEC3F currentVelocity;
    currentVelocity[0] = velocity[x];
    currentVelocity[1] = velocity[x + 1];
    currentVelocity[2] = velocity[x + 2];

    MATRIX3 crossProduct = MATRIX3::cross(currentVorticity);
    VEC3F currentCross = crossProduct * currentVelocity;

    cross[x] = currentCross[0];
    cross[x + 1] = currentCross[1];
    cross[x + 2] = currentCross[2];
  }

  // take the curl
  VECTOR3_FIELD_2D crossField(vectorField2D);
  crossField.unflatten(cross);

  FIELD_2D Fx = crossField.scalarField(0); 
  FIELD_2D Fy = crossField.scalarField(1); 
  FIELD_2D Fz = Fx * 0; 

  FIELD_2D FzDy = Fz.Dy();
  FIELD_2D FyDz = Fy.Dz();
  
  FIELD_2D FxDz = Fx.Dz();
  FIELD_2D FzDx = Fz.Dx();
  
  FIELD_2D FyDx = Fy.Dx();
  FIELD_2D FxDy = Fx.Dy();

  VECTOR3_FIELD_2D curlField(vectorField2D);
  for (int y = 0; y < curlField.yRes(); y++)
    for (int x = 0; x < curlField.xRes(); x++)
    {
      curlField(x,y)[0] = FzDy(x,y) - FyDz(x,y);
      curlField(x,y)[1] = FxDz(x,y) - FzDx(x,y);
      curlField(x,y)[2] = FyDx(x,y) - FxDy(x,y);
    }

  // DEBUG
  FIELD_2D zField = curlField.scalarField(2);
  FIELDVIEW2D(zField);

  return curlField.flatten();
}

///////////////////////////////////////////////////////////////////////
// advect vorticity using the velocity basis
///////////////////////////////////////////////////////////////////////
VECTOR advectVelocityVelocity(const VECTOR& velocity0, const VECTOR& velocity1)
{
  // take the cross-product
  assert(velocity0.size() % 3 == 0);
  const int size = velocity0.size();
  VECTOR cross(size);
  for (int x = 0; x < size; x += 3)
  {
    VEC3F currentVelocity0;
    currentVelocity0[0] = velocity0[x];
    currentVelocity0[1] = velocity0[x + 1];
    currentVelocity0[2] = velocity0[x + 2];

    VEC3F currentVelocity1;
    currentVelocity1[0] = velocity1[x];
    currentVelocity1[1] = velocity1[x + 1];
    currentVelocity1[2] = velocity1[x + 2];

    MATRIX3 crossProduct = MATRIX3::cross(currentVelocity0);
    VEC3F currentCross = crossProduct * currentVelocity1;
    /*
    MATRIX3 c;
    c *= 0;
    c(0,1) = -currentVelocity1[2];
    c(0,2) =  currentVelocity1[1];
    c(1,0) =  currentVelocity1[2];
    c(1,2) = -currentVelocity1[0];
    c(2,0) = -currentVelocity1[1];
    c(2,1) =  currentVelocity1[0];

    VEC3F currentCross = c.transpose() * currentVelocity0;
    */

    cross[x] = currentCross[0];
    cross[x + 1] = currentCross[1];
    cross[x + 2] = currentCross[2];
  }

  return cross;
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
double coefdensity(int a1, int b1, int a2,int b2,int c,int tt)
{
  if (tt==0) {
            //SS x SS
      if (c==0) return 0.25 * -(a1*b2 - a2*b1); // --
      if (c==1) return 0.25 * (a1*b2 + a2*b1);  // -+
      if (c==2) return 0.25 * -(a1*b2 + a2*b1); // +-
      if (c==3) return 0.25 * (a1*b2 - a2*b1);  // ++
  } else if (tt==1) {
      //SC x SS
      if (c==0) return 0.25 * -(a1*b2 - a2*b1); // --
      if (c==1) return 0.25 * -(a1*b2 + a2*b1);  // -+
      if (c==2) return 0.25 * (a1*b2 + a2*b1); // +-
      if (c==3) return 0.25 * (a1*b2 - a2*b1);  // ++
        } else if (tt==2) {
      //CS x SS
      if (c==0) return 0.25 * -(a1*b2 - a2*b1); // --
      if (c==1) return 0.25 * -(a1*b2 + a2*b1);  // -+
      if (c==2) return 0.25 * (a1*b2 + a2*b1); // +-
      if (c==3) return 0.25 * (a1*b2 - a2*b1);  // ++
  } else if (tt==3) {
      //CS x SS
      if (c==0) return 0.25 * -(a1*b2 - a2*b1); // --
      if (c==1) return 0.25 * -(a1*b2 + a2*b1);  // -+
      if (c==2) return 0.25 * (a1*b2 + a2*b1); // +-
      if (c==3) return 0.25 * (a1*b2 - a2*b1);  // ++
  } 

  return 0;
}

///////////////////////////////////////////////////////////////////////
// Build the structure coefficients
///////////////////////////////////////////////////////////////////////
void buildAnalyticC(int rank)
{
  float checkRoot = sqrt(rank);

  float remainder = checkRoot - floor(checkRoot);

  if (fabs(remainder > 1e-4))
  {
    cout << __FILE__ << " " << __FUNCTION__ << " " << __LINE__ << " : " << endl;
    cout << " Analytic basis only works for perfect squares!" << endl;
    exit(0);
  }

  C = TENSOR3(rank, rank, rank);
  for (int d1 = 0; d1 < rank; d1++)
  {
    int a1 = ijPairs[d1].first;
    int a2 = ijPairs[d1].second;
    int a1_2 = a1*a1;
    
    double lambda_a = -(a1*a1 + a2*a2);
    double inv_lambda_a = -1.0/(a1*a1 + a2*a2);
    
    for (int d2 = 0; d2 < rank; d2++) 
    {
      int b1 = ijPairs[d2].first;
      int b2 = ijPairs[d2].second;

      double lambda_b = -(b1*b1 + b2*b2);
      double inv_lambda_b = -1.0/(b1*b1 + b2*b2);

      int k1 = ijPairsReverse[pair<int,int>(a1,a2)];
      int k2 = ijPairsReverse[pair<int,int>(b1,b2)];

      int antipairs[4][2];
      antipairs[0][0] = a1-b1; antipairs[0][1] = a2-b2;
      antipairs[1][0] = a1-b1; antipairs[1][1] = a2+b2;
      antipairs[2][0] = a1+b1; antipairs[2][1] = a2-b2;
      antipairs[3][0] = a1+b1; antipairs[3][1] = a2+b2;
      
      for (int c = 0;c < 4;c++) 
      {
        int i = antipairs[c][0];
        int j = antipairs[c][1];

        if (ijPairsReverse.find(pair<int,int>(i,j)) == ijPairsReverse.end())
          continue;

        int index = ijPairsReverse[pair<int,int>(i,j)];
          
        double coef = - coefdensity(a1,a2,b1,b2,c,0) * inv_lambda_b;
        C(k2,k1,index) = coef;
        C(k1,k2,index) = coef * -lambda_b / lambda_a;
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////
// Build the structure coefficients
///////////////////////////////////////////////////////////////////////
void buildSparseAnalyticC(int rank)
{
  float checkRoot = sqrt(rank);

  float remainder = checkRoot - floor(checkRoot);

  if (fabs(remainder > 1e-4))
  {
    cout << __FILE__ << " " << __FUNCTION__ << " " << __LINE__ << " : " << endl;
    cout << " Analytic basis only works for perfect squares!" << endl;
    exit(0);
  }

  sparseC.clear();

  for (int x = 0; x < rank; x++)
    sparseC.push_back(SPARSE_MATRIX(rank, rank));

  for (int d1 = 0; d1 < rank; d1++)
  {
    int a1 = ijPairs[d1].first;
    int a2 = ijPairs[d1].second;
    int a1_2 = a1*a1;
    
    double lambda_a = -(a1*a1 + a2*a2);
    double inv_lambda_a = -1.0/(a1*a1 + a2*a2);
    
    for (int d2 = 0; d2 < rank; d2++) 
    {
      int b1 = ijPairs[d2].first;
      int b2 = ijPairs[d2].second;

      double lambda_b = -(b1*b1 + b2*b2);
      double inv_lambda_b = -1.0/(b1*b1 + b2*b2);

      int k1 = ijPairsReverse[pair<int,int>(a1,a2)];
      int k2 = ijPairsReverse[pair<int,int>(b1,b2)];

      int antipairs[4][2];
      antipairs[0][0] = a1-b1; antipairs[0][1] = a2-b2;
      antipairs[1][0] = a1-b1; antipairs[1][1] = a2+b2;
      antipairs[2][0] = a1+b1; antipairs[2][1] = a2-b2;
      antipairs[3][0] = a1+b1; antipairs[3][1] = a2+b2;
      
      for (int c = 0;c < 4;c++) 
      {
        int i = antipairs[c][0];
        int j = antipairs[c][1];

        if (ijPairsReverse.find(pair<int,int>(i,j)) == ijPairsReverse.end())
          continue;

        int index = ijPairsReverse[pair<int,int>(i,j)];
          
        double coef = - coefdensity(a1,a2,b1,b2,c,0) * inv_lambda_b;
        sparseC[index](k2,k1) = coef;
        sparseC[index](k1,k2) = coef * -lambda_b / lambda_a;

        if (fabs(coef) > 0) {
          cout << " analytic: " << endl;
          cout << " a1: " << a1 << " a2: " << a2 << endl;
          cout << " b1: " << b1 << " b2: " << b2 << endl;
          cout << " k1: " << k1 << " k2: " << k2 << endl;
          cout << " coef: " << coef << endl;
        }
      }
    }
  }

  // build arrays for fast static multiplies
  for (int x = 0; x < rank; x++)
    sparseC[x].buildStatic();
}

///////////////////////////////////////////////////////////////////////
// Print a string to the GL window
///////////////////////////////////////////////////////////////////////
void printGlString(string output)
{
  //glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
  glColor4f(1.0f, 0.0f, 0.0f, 1.0f);
  for (unsigned int x = 0; x < output.size(); x++)
    glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, output[x]);
}

///////////////////////////////////////////////////////////////////////
// initialize the GL texture
///////////////////////////////////////////////////////////////////////
void initTexture(FIELD_2D& texture)
{
  float* rgb = new float[3 * texture.totalCells()];

  for (int x = 0; x < texture.totalCells(); x++)
  {
    rgb[3 * x] = rgb[3 * x + 1] = rgb[3 * x + 2] = texture[x];

    //if (distanceField[x] > 0)
    //  rgb[3 * x] = texture[x];
    //else
    //  rgb[3 * x + 2] = texture[x];
  }

  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glTexImage2D(GL_TEXTURE_2D, 0, 3, 
      texture.xRes(), 
      texture.yRes(), 0, 
      GL_RGB, GL_FLOAT, 
      rgb);

  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
  glEnable(GL_TEXTURE_2D);

  delete[] rgb;
}

///////////////////////////////////////////////////////////////////////
// update the texture being viewed
///////////////////////////////////////////////////////////////////////
void updateViewingTexture()
{
  viewingField = distanceField;

  if (useAbsolute)
    viewingField.abs();

  viewingField += bias;
  viewingField *= scale;

  if (useLog)
    viewingField.log(10.0);

  // DEBUG
  viewingField = 1;

  initTexture(viewingField);
}

/*
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
void glutReshape(int w, int h)
{
  // Make ensuing transforms affect the projection matrix
  glMatrixMode(GL_PROJECTION);

  // set the projection matrix to an orthographic view
  glLoadIdentity();
  glOrtho(-0.5, 0.5, -0.5, 0.5, -10, 10);

  // set the matric mode back to modelview
  glMatrixMode(GL_MODELVIEW);

  // set the lookat transform
  glLoadIdentity();
  gluLookAt(0, 0, 1,    // eye
            0, 0, 0.0,  // center 
            0, 1, 0);   // up
}
*/

///////////////////////////////////////////////////////////////////////
// draw a grid over everything
///////////////////////////////////////////////////////////////////////
void drawGrid()
{
  glColor4f(0.1, 0.1, 0.1, 1.0);

  int xRes = distanceField.xRes();
  int yRes = distanceField.yRes();

  float dx = 1.0 / xRes;
  float dy = 1.0 / yRes;

  if (xRes < yRes)
    dx *= (Real)xRes / yRes;
  if (xRes > yRes)
    dy *= (Real)yRes / xRes;

  glBegin(GL_LINES);
  for (int x = 0; x < distanceField.xRes() + 1; x++)
  {
    glVertex3f(x * dx, 0, 1);
    glVertex3f(x * dx, 1, 1);
  }
  for (int y = 0; y < distanceField.yRes() + 1; y++)
  {
    glVertex3f(0, y * dy, 1);
    glVertex3f(1, y * dy, 1);
  }
  glEnd();
}

///////////////////////////////////////////////////////////////////////
// draw the velocity field
///////////////////////////////////////////////////////////////////////
void drawVectorField()
{
  // cache the cell centers
  //Real xLength = 1.0;
  //Real yLength = 1.0;
  Real xLength = vectorField2D.lengths()[0];
  Real yLength = vectorField2D.lengths()[1];

  int xRes = distanceField.xRes();
  int yRes = distanceField.yRes();

  if (xRes < yRes)
    xLength = (Real)xRes / yRes;
  if (yRes < xRes)
    yLength = (Real)yRes / xRes;
  
  glPushMatrix();
    glTranslatef(0.5 - (1 - xLength) * 0.5, 0.5 - (1 - yLength) * 0.5, 0);
    //vectorField.drawZSlice(displayCenters, zSlice, 0.1, 1);
  glPopMatrix();

}

///////////////////////////////////////////////////////////////////////
// GL and GLUT callbacks
///////////////////////////////////////////////////////////////////////
void glutDisplay()
{
  // Make ensuing transforms affect the projection matrix
  glMatrixMode(GL_PROJECTION);

  // set the projection matrix to an orthographic view
  glLoadIdentity();
  float halfZoom = zoom * 0.5;

  glOrtho(-halfZoom, halfZoom, -halfZoom, halfZoom, -10, 10);

  // set the matric mode back to modelview
  glMatrixMode(GL_MODELVIEW);

  // set the lookat transform
  glLoadIdentity();
  gluLookAt(eyeCenter[0], eyeCenter[1], 1,  // eye
            eyeCenter[0], eyeCenter[1], 0,  // center 
            0, 1, 0);   // up

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  Real xLength = vectorField2D.lengths()[0];
  Real yLength = vectorField2D.lengths()[1];

  int xRes = distanceField.xRes();
  int yRes = distanceField.yRes();
  if (xRes < yRes)
    xLength = (Real)xRes / yRes;
  if (yRes < xRes)
    yLength = (Real)yRes / xRes;

  glColor4f(1,1,1,1);
  glEnable(GL_TEXTURE_2D); 
  glBegin(GL_QUADS);
    glTexCoord2f(0.0, 0.0); glVertex3f(0.0, 0.0, 0.0);
    glTexCoord2f(0.0, 1.0); glVertex3f(0.0, yLength, 0.0);
    glTexCoord2f(1.0, 1.0); glVertex3f(xLength, yLength, 0.0);
    glTexCoord2f(1.0, 0.0); glVertex3f(xLength, 0.0, 0.0);
  glEnd();
  glDisable(GL_TEXTURE_2D);

  if (drawingGrid)
    drawGrid();

  //vectorField2D.draw();

  glColor4f(1,0,0,0.5);
  glPointSize(5);
  glBegin(GL_POINTS);
  for (unsigned int x = 0; x < particles.size(); x++)
  {
    glVertex3f(particles[x][0], particles[x][1], 0);
  }
  glEnd();

  // if there's a valid field index, print it
  if (xField >= 0 && yField >= 0 &&
      xField < distanceField.xRes() && yField < distanceField.yRes())
  {
    glLoadIdentity();

    // must set color before setting raster position, otherwise it won't take
    glColor4f(0.0f, 1.0f, 0.0f, 1.0f);

    // normalized screen coordinates (-0.5, 0.5), due to the glLoadIdentity
    float halfZoom = 0.5 * zoom;
    glRasterPos3f(-halfZoom* 0.95, -halfZoom* 0.95, 0);

    // build the field value string
    char buffer[256];

    // add the integer index
    int index = xField + yField * distanceField.xRes();
    string fieldValue;
    sprintf(buffer, "(");
    fieldValue = fieldValue + string(buffer);
    sprintf(buffer, "%i", xField);
    fieldValue = fieldValue + string(buffer);
    sprintf(buffer, "%i", yField);
    fieldValue = fieldValue + string(", ") + string(buffer) + string(") = ");

    // add the vector field value
    VEC3F vectorValue = vectorField2D[index];
    sprintf(buffer, "%f", vectorValue[0]);
    fieldValue = fieldValue + string("(") + string(buffer);
    sprintf(buffer, "%f", vectorValue[1]);
    fieldValue = fieldValue + string(",") + string(buffer);
    sprintf(buffer, "%f", vectorValue[2]);
    fieldValue = fieldValue + string(",") + string(buffer) + string(")");

    // add the global position
    /*
    VEC3F position = distanceField.cellCenter(xField, yField, zSlice);
    sprintf(buffer, "%f", position[0]);
    fieldValue = fieldValue + string("(") + string(buffer);
    sprintf(buffer, "%f", position[1]);
    fieldValue = fieldValue + string(",") + string(buffer);
    sprintf(buffer, "%f", position[2]);
    fieldValue = fieldValue + string(",") + string(buffer) + string(") = ");

    switch (whichViewing)
    {
      case REAL:
        sprintf(buffer, "%f", fftReal(xField, yField));
        break;
      case IM:
        sprintf(buffer, "%f", fftIm(xField, yField));
        break;
      default:
        {
          Real value = distanceField(xField, yField, zSlice);
          if (isnan(value))
            sprintf(buffer, "nan");
          else
            sprintf(buffer, "%f", value);
        }
        break;
    }
    fieldValue = fieldValue + string(buffer);
    */

    printGlString(fieldValue);
  }

  if (captureMovie)
    movie.addFrameGL();

  glutSwapBuffers();
}

///////////////////////////////////////////////////////////////////////
// throw down some particles
///////////////////////////////////////////////////////////////////////
void seedParticles()
{
  static MERSENNETWISTER twister(123456);
  const VEC3F lengths = vectorField2D.lengths();

  //for (int x = 0; x < 10000; x++)
  for (int x = 0; x < 40000; x++)
  {
    VEC3F particle = lengths;
    particle[0] *= twister.rand();
    particle[1] *= twister.rand();

    particles.push_back(particle);
  }    
}

///////////////////////////////////////////////////////////////////////
// push around some particles
///////////////////////////////////////////////////////////////////////
void advectParticles()
{
  TIMER functionTimer(__FUNCTION__);
  for (unsigned int x = 0; x < particles.size(); x++)
  {
    VEC3F velocity = vectorField2D(particles[x]);
    particles[x] += dt * velocity;
  }
}

///////////////////////////////////////////////////////////////////////
// animate and display new result
///////////////////////////////////////////////////////////////////////
void glutIdle()
{
  stepEigenfunctions();
  advectParticles();
  //seedParticles();

  glutDisplay();
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
void printCommands()
{
  cout << " q           - quits" << endl;
  cout << " left mouse  - pan around" << endl;
  cout << " right mouse - zoom in and out " << endl;
  cout << " right arrow - increase bias " << endl;
  cout << " left arrow  - decrease bias " << endl;
  cout << " up arrow    - increase scale " << endl;
  cout << " down arrow  - decrease scale " << endl;
  cout << " n           - normalize (auto-set scale and bias) " << endl;
  cout << " g           - throw a grid over the pixels " << endl;
  cout << " a           - take absolute value of cells " << endl;
  cout << " l           - take log of cells " << endl;
  cout << " r           - look at real component of FFT" << endl;
  cout << " i           - look at imaginary component of FFT" << endl;
  cout << " s           - look at spatial (non-FFT)" << endl;
}

///////////////////////////////////////////////////////////////////////
// normalize the texture
///////////////////////////////////////////////////////////////////////
void normalize()
{
  switch (whichViewing)
  {
    case REAL:
      viewingField = fftReal;
      break;
    case IM:
      viewingField = fftIm;
      break;
    default:
      viewingField = distanceField;
      break;
  }

  if (useAbsolute)
    viewingField.abs();

  float minFound = viewingField.min();
  float maxFound = viewingField.max();

  // cache the values in case we want to undo the normalization
  oldScale = scale;
  oldBias = bias;

  // go ahead and compute the normalized version
  bias = -minFound;
  scale = 1.0 / (maxFound - minFound);

  updateViewingTexture();
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
void glutSpecial(int key, int x, int y)
{
  switch (key)
  {
    case GLUT_KEY_LEFT:
      bias -= 0.01;
      break;
    case GLUT_KEY_RIGHT:
      bias += 0.01;
      break;
    case GLUT_KEY_UP:
      scale += 0.01;
      break;
    case GLUT_KEY_DOWN:
      scale -= 0.01;
      break;
  }
  cout << " scale: " << scale << " bias: " << bias << endl;

  updateViewingTexture();
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
void glutKeyboard(unsigned char key, int x, int y)
{
  switch (key)
  {
    case ' ':
      //TIMER::printTimings();
      //TIMER::printTimingsPerFrame(frames);
      TIMER::printRawTimingsPerFrame(frames);
      break;
    case 'a':
      useAbsolute = !useAbsolute;
      updateViewingTexture();
      break;
    case 'l':
      useLog = !useLog;
      updateViewingTexture();
      break;
    case 'g':
      drawingGrid = !drawingGrid;
      break;
    case 'n':
      if (normalized)
      {
        scale = oldScale;
        bias = oldBias;
        updateViewingTexture();
      }
      else
        normalize();

      normalized = !normalized;
      break;
    case '?':
      printCommands();
      break;
    case 'r':
      whichViewing = REAL;
      updateViewingTexture();
      break;
    case 'i':
      whichViewing = IM;
      updateViewingTexture();
      break;
    case 's':
      whichViewing = SPATIAL;
      updateViewingTexture();
      break;
    case 'm':
      if (captureMovie)
        movie.writeMovie("movie.mov");
      captureMovie = !captureMovie;
      break;
    case 'q':
      if (captureMovie)
        movie.writeMovie("movie.mov");
      exit(0);
      break;
    default:
      break;
  }
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
void glutMouseClick(int button, int state, int x, int y)
{
  xMouse = x;  
  yMouse = y;

  mouseButton = button;
  mouseState = state;
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
void glutMouseMotion(int x, int y)
{
  float xDiff = x - xMouse;
  float yDiff = y - yMouse;
  float speed = 0.001;
  
  if (mouseButton == GLUT_LEFT_BUTTON) 
  {
    eyeCenter[0] -= xDiff * speed;
    eyeCenter[1] += yDiff * speed;
  }
  if (mouseButton == GLUT_RIGHT_BUTTON)
  {
    zoom -= yDiff * speed;
  }

  xMouse = x;
  yMouse = y;
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
void glutPassiveMouseMotion(int x, int y)
{
  // make the lower left the origin
  y = yScreenRes - y;

  float xNorm = (float)x / xScreenRes;
  float yNorm = (float)y / yScreenRes;

  float halfZoom = 0.5 * zoom;
  float xWorldMin = eyeCenter[0] - halfZoom;
  float xWorldMax = eyeCenter[0] + halfZoom;

  // get the bounds of the field in screen coordinates
  //
  // if non-square textures are ever supported, change the 0.0 and 1.0 below
  float xMin = (0.0 - xWorldMin) / (xWorldMax - xWorldMin);
  float xMax = (1.0 - xWorldMin) / (xWorldMax - xWorldMin);

  float yWorldMin = eyeCenter[1] - halfZoom;
  float yWorldMax = eyeCenter[1] + halfZoom;

  float yMin = (0.0 - yWorldMin) / (yWorldMax - yWorldMin);
  float yMax = (1.0 - yWorldMin) / (yWorldMax - yWorldMin);

  int xRes = distanceField.xRes();
  int yRes = distanceField.yRes();

  Real xScale = 1.0;
  Real yScale = 1.0;

  if (xRes < yRes)
    xScale = (Real)yRes / xRes;
  if (xRes > yRes)
    yScale = (Real)xRes / yRes;

  // index into the field after normalizing according to screen
  // coordinates
  xField = xScale * xRes * ((xNorm - xMin) / (xMax - xMin));
  yField = yScale * yRes * ((yNorm - yMin) / (yMax - yMin));
}

//////////////////////////////////////////////////////////////////////////////
// open the GLVU window
//////////////////////////////////////////////////////////////////////////////
int glvuWindow()
{
  glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE| GLUT_RGBA);
  glutInitWindowSize(xScreenRes, yScreenRes); 
  glutInitWindowPosition(10, 10);
  glutCreateWindow(windowLabel.c_str());

  // set the viewport resolution (w x h)
  glViewport(0, 0, (GLsizei) xScreenRes, (GLsizei) yScreenRes);

  glClearColor(0.1, 0.1, 0.1, 0);
  glutDisplayFunc(&glutDisplay);
  glutIdleFunc(&glutIdle);
  glutKeyboardFunc(&glutKeyboard);
  glutSpecialFunc(&glutSpecial);
  glutMouseFunc(&glutMouseClick);
  glutMotionFunc(&glutMouseMotion);
  glutPassiveMotionFunc(&glutPassiveMouseMotion);

  glEnable (GL_BLEND);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glShadeModel(GL_SMOOTH);

  updateViewingTexture();

  glutMainLoop();

  // Control flow will never reach here
  return EXIT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////
// run some tests on the eigenfunctions
///////////////////////////////////////////////////////////////////////
void tests()
{
  /*
  int a1 = 2;
  int a2 = 3;
  int b1 = 1;
  int b2 = 2;
  */
  int a1 = 2;
  int a2 = 4;

  int b1 = 1;
  int b2 = 3;
  int xRes = 256;
  int yRes = 256;
  VEC3F center(M_PI / 2.0, M_PI / 2.0, 0);
  VEC3F lengths(M_PI, M_PI, 0);
  vectorField2D = VECTOR3_FIELD_2D(xRes, yRes, center, lengths);
  VECTOR3_FIELD_2D dotCurlCross(xRes, yRes, center, lengths);
  dotCurlCross.dotCurledCrossed(a1 ,a2, b1, b2, 1, 1);
  VECTOR3_FIELD_2D curlCross(xRes, yRes, center, lengths);
  curlCross.curlCrossedVorticity(a1 ,a2, b1, b2);

  /*
  {
    Real x = M_PI / 4;
    Real y = M_PI / 8;
    Real cross = a1 * b2 * cos(a1 * x) * cos(b2 * y) * sin(a2 * y) * sin(b1 * x) -
                 a2 * b1 * cos(a2 * y) * cos(b1 * x) * sin(a1 * x) * sin(b2 * y);

    cout << " center value, curl crossed: " << cross << endl;
  }
  */
  VECTOR3_FIELD_2D vorticityk1(xRes, yRes, center, lengths);
  vorticityk1.vorticity(a1, a2);
  VECTOR3_FIELD_2D vorticityk2(xRes, yRes, center, lengths);
  vorticityk2.vorticity(b1, b2);

  //FIELDVIEW2D(vorticityk1.scalarField(2));
  //FIELDVIEW2D(vorticityk2.scalarField(2));
  //FIELDVIEW2D(dotCurlCross.scalarField(2));
  //FIELDVIEW2D(curlCross.scalarField(2));
  //exit(0);

  //vectorField2D.curlCrossedVorticity(a1 ,a2, b1, b2);

  Real dxSq = vectorField2D.dx() * vectorField2D.dx();

  FIELD_2D zComponent = dotCurlCross.scalarField(2);
  //FIELDVIEW2D(zComponent);
  cout << " direct sum: " << zComponent.sum() * dxSq << endl;

  vectorField2D.vorticity(1,1);
  FIELD_2D z11 = vectorField2D.scalarField(2);
  //FIELDVIEW2D(z11);

  VECTOR3_FIELD_2D velocityFieldi = vectorField2D;
  velocityFieldi.eigenfunctionUnscaled(a1,a2);
  VECTOR velocityi = velocityFieldi.flatten();
  VECTOR3_FIELD_2D curli = VECTOR3_FIELD_2D::curl(vorticityk1.scalarField(2));

  VECTOR3_FIELD_2D velocityFieldj = vectorField2D;
  velocityFieldj.eigenfunctionUnscaled(b1,b2);
  VECTOR velocityj = velocityFieldj.flatten();
  VECTOR3_FIELD_2D curlj = VECTOR3_FIELD_2D::curl(vorticityk2.scalarField(2));

  /*
  {
    Real x = M_PI / 4;
    Real y = M_PI / 8;

    VEC3F iVec;
    iVec[0] =  a2 * sin(a1 * x) * cos(a2 * y);
    iVec[1] = -a1 * cos(a1 * x) * sin(a2 * y);
    iVec[2] = 0;
    
    VEC3F jVec;
    jVec[0] =  b2 * sin(b1 * x) * cos(b2 * y);
    jVec[1] = -b1 * cos(b1 * x) * sin(b2 * y);
    jVec[2] = 0;

    cout << " iVec: " << iVec << endl;
    cout << " jVec: " << jVec << endl;

    VEC3F crossed = cross(iVec, jVec);
    cout << " crossed: " << crossed << endl;
  }
  */

  //advectVorticityVelocity(vorticity23, velocity12);

  //FIELDVIEW2D(curli.magnitudeField());
  //FIELDVIEW2D(velocityFieldi.magnitudeField());
  //FIELDVIEW2D(curlj.magnitudeField());
  //FIELDVIEW2D(velocityFieldj.magnitudeField());
  //exit(0);

  VECTOR cross = advectVelocityVelocity(velocityi, velocityj);
  VECTOR3_FIELD_2D crossField(vectorField2D);
  crossField.unflatten(cross);
  //FIELD_2D z2312 = crossField.scalarField(2);
  //FIELDVIEW2D(crossField.scalarField(0));
  //FIELDVIEW2D(crossField.scalarField(1));
  FIELDVIEW2D(crossField.scalarField(2));
  //FIELDVIEW2D(velocityFieldi.magnitudeField());
  //FIELDVIEW2D(velocityFieldj.scalarField(0));
  //FIELDVIEW2D(velocityFieldj.scalarField(1));
  //FIELDVIEW2D(velocityFieldj.scalarField(2));
  //exit(0);
  VECTOR3_FIELD_2D vorticity11 = vectorField2D;
  vorticity11.vorticity(1,1);

  FIELD_2D dotDirect = crossField.scalarField(2);
  dotDirect *= vorticity11.scalarField(2);

  //FIELDVIEW2D(dotDirect);
  cout << " numerical sum: " << dotDirect.sum() * dxSq << endl;
  cout << " static version: " << VECTOR3_FIELD_2D::structureCoefficient(a1,a2,b1,b2,1,1) << endl;
  //cout << " ground: " << -M_PI * M_PI / 16 << endl;
  //cout << " ground: " << -M_PI * M_PI / 8 << endl;
  cout << " ground: " << -(M_PI * M_PI / 8) << endl;

  Real ground = -(M_PI * M_PI / 8);
  cout << " scaled ground: " << ground * 4 / (M_PI * M_PI) / (b1 * b1 + b2 * b2) << endl;

  exit(0);
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
void test2()
{
  buildIJPairsPerfectSquare(basisRank);
  buildAnalyticC(basisRank);

  MATRIX newC(basisRank, basisRank);

  for (int d1 = 0; d1 < basisRank; d1++)
  {
    int a1 = ijPairs[d1].first;
    int a2 = ijPairs[d1].second;
    for (int d2 = 0; d2 < basisRank; d2++) 
    {
      int b1 = ijPairs[d2].first;
      int b2 = ijPairs[d2].second;

      int a = ijPairsReverse[pair<int,int>(a1,a2)];
      int b = ijPairsReverse[pair<int,int>(b1,b2)];

      Real coef = VECTOR3_FIELD_2D::structureCoefficient(a1, a2, b1, b2, 1,1);

      //newC(k1, k2) = coef;
      newC(b, a) = coef;
      //cout << " k1: " << k1 << " k2: " << k2 << " a1: " << a1 << " a2: " << a2 << " b1: " << b1 << " b2: " << b2 << " coef: " << coef << endl;
    }
    cout << d1 << endl;
  }

  cout << " newC = " << newC << endl;
  cout << " groundC = " << C.slab(0) << endl; 
  exit(0);
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
void buildDirectAnalyticC(int basisRank)
{
  C = TENSOR3(basisRank, basisRank, basisRank);
  for (int d1 = 0; d1 < basisRank; d1++)
  {
    int a1 = ijPairs[d1].first;
    int a2 = ijPairs[d1].second;
    for (int d2 = 0; d2 < basisRank; d2++) 
    {
      int b1 = ijPairs[d2].first;
      int b2 = ijPairs[d2].second;

      int a = ijPairsReverse[pair<int,int>(a1,a2)];
      int b = ijPairsReverse[pair<int,int>(b1,b2)];

      for (int d3 = 0; d3 < basisRank; d3++)
      {
        int k1 = ijPairs[d3].first;
        int k2 = ijPairs[d3].second;
        int k = ijPairsReverse[pair<int,int>(k1,k2)];
        //cout << " k1: " << k1 << " k2: " << k2 << " k: " << k << endl;
        Real coef = VECTOR3_FIELD_2D::structureCoefficientAnalytic(a1, a2, b1, b2, k1, k2);

        //newC(k1, k2) = coef;
        C(b, a, k) = coef;

        /*
        if (fabs(coef) > 1e-5)
        {
          cout << " numerical: " << endl;
          cout << " a1: " << a1 << " a2: " << a2 << endl;
          cout << " b1: " << b1 << " b2: " << b2 << endl;
          cout << " k1: " << k1 << " k2: " << k2 << endl;
          cout << " coef: " << coef << endl;
        }
        */
      }
    }
    cout << d1 << endl;
  }

  for (int x = 0; x < basisRank; x++)
    cout << " Analytic slice " << x << ": " << C.slab(x) << endl;
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
void buildDirectSparseAnalyticC(int rank)
{
  sparseC.clear();

  for (int x = 0; x < rank; x++)
    sparseC.push_back(SPARSE_MATRIX(rank, rank));

  cout << " Building C: " << flush;
  for (int d1 = 0; d1 < rank; d1++)
  {
    int a1 = ijPairs[d1].first;
    int a2 = ijPairs[d1].second;
    for (int d2 = 0; d2 < rank; d2++) 
    {
      int b1 = ijPairs[d2].first;
      int b2 = ijPairs[d2].second;

      int a = ijPairsReverse[pair<int,int>(a1,a2)];
      int b = ijPairsReverse[pair<int,int>(b1,b2)];

      for (int d3 = 0; d3 < rank; d3++)
      {
        int k1 = ijPairs[d3].first;
        int k2 = ijPairs[d3].second;
        int k = ijPairsReverse[pair<int,int>(k1,k2)];
        Real coef = VECTOR3_FIELD_2D::structureCoefficientAnalytic(a1, a2, b1, b2, k1, k2);

        if (fabs(coef) > 1e-8)
          //sparseC[k](b, a) = coef;
          sparseC[k](b, a) = -coef;
      }
    }
    cout << d1 << " " << flush;
  }
  cout << " done. " << endl;

  // build arrays for fast static multiplies
  for (int x = 0; x < rank; x++)
    sparseC[x].buildStatic();
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
void buildC()
{
  C = TENSOR3(basisRank, basisRank, basisRank);
  for (int d1 = 0; d1 < basisRank; d1++)
  {
    int a1 = ijPairs[d1].first;
    int a2 = ijPairs[d1].second;
    for (int d2 = 0; d2 < basisRank; d2++) 
    {
      int b1 = ijPairs[d2].first;
      int b2 = ijPairs[d2].second;

      int a = ijPairsReverse[pair<int,int>(a1,a2)];
      int b = ijPairsReverse[pair<int,int>(b1,b2)];

      for (int d3 = 0; d3 < basisRank; d3++)
      {
        int k1 = ijPairs[d3].first;
        int k2 = ijPairs[d3].second;
        int k = ijPairsReverse[pair<int,int>(k1,k2)];
        //cout << " k1: " << k1 << " k2: " << k2 << " k: " << k << endl;
        Real coef = VECTOR3_FIELD_2D::structureCoefficient(a1, a2, b1, b2, k1, k2);

        //newC(k1, k2) = coef;
        C(b, a, k) = coef;

        /*
        if (fabs(coef) > 1e-5)
        {
          cout << " numerical: " << endl;
          cout << " a1: " << a1 << " a2: " << a2 << endl;
          cout << " b1: " << b1 << " b2: " << b2 << endl;
          cout << " k1: " << k1 << " k2: " << k2 << endl;
          cout << " coef: " << coef << endl;
        }
        */
      }
    }
    cout << d1 << endl;
  }

  for (int y = 0; y < basisRank; y++)
  {
    MATRIX slice = C.slab(y);
    for (int x = 0; x < slice.rows() * slice.cols(); x++)
      slice.data()[x] = (fabs(slice.data()[x]) < 1e-7) ? 0 : slice.data()[x];
    cout << " Numerical slice " << y << ": " << slice << endl;
  }
}

///////////////////////////////////////////////////////////////////////
// build inverse plans
///////////////////////////////////////////////////////////////////////
void computePlans()
{
  cout << " Computing plans ..." << flush;
  //TIMER functionTimer(__FUNCTION__);
  xHat.resizeAndWipe(xRes, yRes);
  yHat.resizeAndWipe(xRes, yRes);
  
  xHat.xPlanIDST(xPlanIDST);
  xHat.yPlanIDCT(yPlanIDCT);
  xHat.xPlanIDSTyIDCT(xPlanIDSTyIDCT);
  
  yHat.xPlanIDCT(xPlanIDCT);
  yHat.yPlanIDST(yPlanIDST);
  yHat.xPlanIDCTyIDST(yPlanIDCTyIDST);

  cout << " done." << endl;
}

///////////////////////////////////////////////////////////////////////
// compute the FFT version of each velocity field
///////////////////////////////////////////////////////////////////////
void computeFieldFFTs()
{
  cout << " Computing FFTs ... " << flush;
  VECTOR3_FIELD_2D velocityField(xRes, yRes); 

  velocityFFTX.clear();
  velocityFFTY.clear();

  xDeltaIndices.clear();
  xMagnitudes.clear();
  yDeltaIndices.clear();
  yMagnitudes.clear();

  for (int i = 0; i < velocityU.cols(); i++)
  {
    VECTOR column = velocityU.getColumn(i);
    velocityField.unflatten(column);
    FIELD_2D xField = velocityField.scalarField(0);
    FIELD_2D yField = velocityField.scalarField(1);
    
    xField.yDCT();
    xField.xDST();
    
    yField.xDCT();
    yField.yDST();

    int xIndex = xField.maxAbsIndex();
    xDeltaIndices.push_back(xIndex);
    xMagnitudes.push_back(xField[xIndex]);
    
    int yIndex = yField.maxAbsIndex();
    yDeltaIndices.push_back(yIndex);
    yMagnitudes.push_back(yField[yIndex]);

    velocityFFTX.push_back(xField);
    velocityFFTY.push_back(yField);
  }

  cout << "done. " << endl;
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
  if (argc < 2)
  {
    cout << " USAGE: " << argv[0] << " *.cfg" << endl;
    return 0;
  }

  SIMPLE_PARSER parser(argv[1]);

  //doubleCurl();

  //tests();
  //test2();

  //int xRes = 65;
  //int yRes = 65;
  xRes = parser.getInt("xRes", xRes);
  yRes = parser.getInt("yRes", yRes);
  basisRank = parser.getInt("basis rank", basisRank);
  path = parser.getString("path", path);
  viscosity = parser.getFloat("viscosity", viscosity);
  dt = parser.getFloat("dt", dt);
  int method = parser.getInt("reconstruction method", 0);

  reconstructionMethod = (RECONSTRUCTION_METHOD)method;

  cout << " Using parameters: " << endl;
  cout << "   xRes:      " << xRes << endl;
  cout << "   yRes:      " << yRes << endl;
  cout << "   basisRank: " << basisRank<< endl;
  cout << "   path:      " << path << endl;
  cout << "   viscosity: " << viscosity << endl;
  cout << "   dt:        " << dt << endl;

  switch (reconstructionMethod) {
    case FFT_RECON:
      cout << "   reconstruction: FFT " << endl;
      break;
    case MATRIX_VECTOR_RECON:
      cout << "   reconstruction: Matrix-vector" << endl;
      break;
    case ANALYTIC_RECON:
      cout << "   reconstruction: Analytic" << endl;
      break;
  }

  VEC3F center(M_PI / 2.0, M_PI / 2.0, 0);
  VEC3F lengths(M_PI, M_PI, 0);
  vectorField2D = VECTOR3_FIELD_2D(xRes, yRes, center, lengths);
  vectorField2D.eigenfunction(1,1);
  vectorField2D.writeLIC("1_1.jpg");
  
  // build a temp field for if we want to reconstruct the velocity field
  // analytically
  eigenfield = VECTOR3_FIELD_2D(xRes, yRes, center, lengths);

  char buffer[256];
  sprintf(buffer, "%i", basisRank);
  string filenameC = path + string("C.") + string(buffer) + string(".tensor");

  buildIJPairs(basisRank);
  buildVelocityBasis(basisRank);
  //buildVorticityBasis(basisRank);
  /*
  //C.read("./data/C.tensor");
  bool success = C.read(filenameC);

  if (success)
    cout << " Precomputed tensor " << filenameC.c_str() << " found! " << endl;
  else
  {
    buildC();
    C.write(filenameC);
  }
  */
  //buildC();
  //buildDirectAnalyticC(basisRank);
  //buildAnalyticC(basisRank);
  //buildSparseAnalyticC(basisRank);
  buildDirectSparseAnalyticC(basisRank);

  // compute the FFT version of each velocity field
  computeFieldFFTs();

  // compute the plans for the FFTs
  computePlans();

  // initialize vorticity vectors
  w = VECTOR(basisRank);
  wDot = VECTOR(basisRank);

  // insert a vertical impulse right in the middle
  VECTOR3_FIELD_2D impulse(xRes, yRes, center, lengths);
  //impulse(xRes / 2, yRes / 2)[1] = 1000;
  impulse(xRes / 2, yRes / 2)[0] = 1000;
  //impulse(xRes / 2, yRes / 2)[0] = 10000;

  // do it the FFT way
#if 1
  FIELD_2D xField = impulse.scalarField(0);
  xField.yDCT();
  xField.xDST();
  FIELD_2D yField = impulse.scalarField(1);
  yField.xDCT();
  yField.yDST();

  VECTOR force(w.size());
  for (int x = 0; x < xDeltaIndices.size(); x++)
    force[x] += xField[xDeltaIndices[x]];
  for (int x = 0; x < yDeltaIndices.size(); x++)
    force[x] += -yField[yDeltaIndices[x]];
#else
  // do it the matrix-vector way
  VECTOR force = velocityU ^ impulse.flatten();
#endif
  w += force;

  eyeCenter = center;
  zoom = lengths[0] * 1.1;
  distanceField = vectorField2D.LIC();

  seedParticles();

  glutInit(&argc, argv);

  glvuWindow();
  return 1;
}
