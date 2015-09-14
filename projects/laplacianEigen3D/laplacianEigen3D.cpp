/*
 This file is part of SSFR (Zephyr).
 
 Zephyr is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 
 Zephyr is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with Zephyr.  If not, see <http://www.gnu.org/licenses/>.
 
 Copyright 2013 Theodore Kim
 */
#include "EIGEN.h"

#include <cmath>
#include <glvu.h>
#include <VEC3.h>
#include <iostream>
#if __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#include <GL/glu.h>
#endif
#include <list>
#include "MERSENNETWISTER.h"
#include "FIELD_3D.h"
#include "VECTOR3_FIELD_3D.h"
#include "MATRIX.h"
#include "TENSOR3.h"
#include "SIMPLE_PARSER.h"
#include "SPARSE_MATRIX.h"
#include "SPARSE_TENSOR3.h"

using namespace std;

GLVU glvu;

// is the mouse pressed?
bool mouseClicked = false;

// what's the z coordinate of the last mouse click?
float clickZ;

// animate it this frame?
bool animate = true;

void runEverytime();

// user configuration
string snapshotPath("./data/snapshots.stam.no.vorticity/");
string previewMovie("./data/stam.mov");
int simulationSnapshots = 20;

VEC3F center(0,0,0);
VEC3F lengths(1,1,1);

// these determine the size of the basis that will be used
//int res = 40;
int res = 32;
//int res = 64;
//int dim = 2;
//int res = 100;
//int dim = 3;
int dim = 4;
//int dim = 5;
//int dim = 6;
//int dim = 10;

//int res = 100;
VECTOR3_FIELD_3D velocityField(res, res, res, center, lengths);

vector<VEC3F> particles;
vector<list<VEC3F> > ribbons;
vector<list<VEC3F> > velocityRibbons;

vector<vector<int> > ixyz;
vector<Real> eigenvalues;
map<int, int> ixyzReverse;

MATRIX velocityU;
MATRIX vorticityU;
TENSOR3 C;
//vector<SPARSE_MATRIX> sparseC;
SPARSE_TENSOR3 tensorC;

// use for reverse lookups later
int slabSize;
int xRes;

VECTOR w;
VECTOR wDot;
Real dt = 0.001;
int frames = 0;

vector<int> xDeltaIndices;
vector<Real> xMagnitudes;
vector<Real> xProjection;
vector<int> yDeltaIndices;
vector<Real> yMagnitudes;
vector<Real> yProjection;
vector<int> zDeltaIndices;
vector<Real> zMagnitudes;
vector<Real> zProjection;

FIELD_3D xHat(res,res,res);
FIELD_3D yHat(res,res,res);
FIELD_3D zHat(res,res,res);
fftw_plan xPlanIDSTyIDCTzIDCT;
fftw_plan yPlanIDCTyIDSTzIDCT;
fftw_plan zPlanIDCTyIDCTzIDST;

///////////////////////////////////////////////////////////////////////
// reconstruct the velocity field using FFT
///////////////////////////////////////////////////////////////////////
void reconstructSparseFFT()
{
  xHat.clear();
  yHat.clear();
  zHat.clear();

  assert(xDeltaIndices.size() == w.size());
  assert(yDeltaIndices.size() == w.size());
  assert(zDeltaIndices.size() == w.size());

  // build a frequency-domain x velocity
  for (int i = 0; i < w.size(); i++)
    xHat[xDeltaIndices[i]] += xMagnitudes[i] * w[i];
  
  // build a frequency-domain y velocity
  for (int i = 0; i < w.size(); i++)
    yHat[yDeltaIndices[i]] += yMagnitudes[i] * w[i];
 
  // build a frequency-domain z velocity
  for (int i = 0; i < w.size(); i++)
    zHat[zDeltaIndices[i]] += zMagnitudes[i] * w[i];

  // build the spatial version
  //TIMER FFTTimer("FFT calls");
  xHat.xIDSTyIDCTzIDCT(xPlanIDSTyIDCTzIDCT);
  yHat.xIDCTyIDSTzIDCT(yPlanIDCTyIDSTzIDCT);
  zHat.xIDCTyIDCTzIDST(zPlanIDCTyIDCTzIDST);

  TIMER setter("Set timer");
  velocityField.setX(xHat);
  velocityField.setY(yHat);
  velocityField.setZ(zHat);
}

///////////////////////////////////////////////////////////////////////
// step the system forward in time
///////////////////////////////////////////////////////////////////////
void stepEigenfunctions()
{
  TIMER functionTimer(__FUNCTION__);
  int basisRank = ixyz.size();
  Real e1 = w.dot(w);

#if 0
  for (int k = 0; k < basisRank; k++)
  {
    MATRIX slab = C.slab(k);
    wDot[k] = w.dot(slab * w);
  }
#else
  /*
  for (int k = 0; k < basisRank; k++)
  {
    SPARSE_MATRIX& slab = sparseC[k];
    wDot[k] = w.dot(slab.staticMultiply(w));
  }
  */
  for (int k = 0; k < basisRank; k++)
  {
    SPARSE_MATRIX& slab = tensorC.slab(k);
    wDot[k] = w.dot(slab.staticMultiply(w));
  }
#endif

  w += dt * wDot;

  Real e2 = w.dot(w);

  if (e2 > 0)
    w *= sqrt(e1 / e2);

  for (int k = 0; k < basisRank; k++)
  {
    Real lambda = -(eigenvalues[k]);

    //const Real viscosity = 0.0;
    //
    // this is the one the 2D version uses
    const Real viscosity = 0.1;
    //const Real viscosity = 0.5;
    //const Real viscosity = 1.0;
    //const Real viscosity = 5;
    //const Real viscosity = 10;

    // diffuse
    w[k] *= exp(lambda * dt * viscosity);
  }

  {
  TIMER reconstructionTimer("Velocity Reconstruction");
  //VECTOR final = velocityU * w;
  //velocityField.unflatten(final);

  reconstructSparseFFT();
  }

  frames++;
}

///////////////////////////////////////////////////////////////////////
// build a list of the i,x,y,z indices to build the eigenfunctions
//
// In addition to the 3 zero modes, all the modes (k1,k2,k3)
// corresponding to a (dim x dim x dim) cube will also be added
///////////////////////////////////////////////////////////////////////
void buildTableIXYZ(int dim)
{
  ixyz.clear();
  slabSize = dim * dim * 3;
  xRes = dim * 3;

  // add the cube of modes
  for (int z = 0; z < dim + 1; z++)
    for (int y = 0; y < dim + 1; y++)
      for (int x = 0; x < dim + 1; x++)
        for (int i = 0; i < 3; i++)
        {
          Real eigenvalue = x * x + y * y + z * z;

          int ks[] = {x,y,z}; 
          int k1Zero = (x == 0);
          int k2Zero = (y == 0);
          int k3Zero = (z == 0);
          int sum = k1Zero + k2Zero + k3Zero;

          // if two of the k's are zero, the whole velocity field will resolve to zero
          if (sum >= 2)
            continue;
          // if the component's k is zero, it will end up zeroing out the whole field
          if (ks[i] == 0)
            continue;

          vector<int> mode;
          mode.push_back(i);
          mode.push_back(x);
          mode.push_back(y);
          mode.push_back(z);
          ixyz.push_back(mode);

          eigenvalues.push_back(eigenvalue);

          int reverse = mode[3] * slabSize + mode[2] * xRes + mode[1] * 3 + mode[0];
          ixyzReverse[reverse] = ixyz.size() - 1;
        }

  cout << " Basis rank: " << ixyz.size() << endl;
  vector<int> lastMode = ixyz.back();
  cout << " Last mode (" << lastMode[0] << ", " << lastMode[1] << "," << lastMode[2] << "," << lastMode[3] << ")" << endl;
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
int reverseLookup(const vector<int>& a123)
{
  int reverse = a123[3] * slabSize + a123[2] * xRes + a123[1] * 3 + a123[0];
  if (ixyzReverse.find(reverse) == ixyzReverse.end())
  {
    cout << __FILE__ << " " << __FUNCTION__ << " " << __LINE__ << " : " << endl;
    cout << __FILE__ << " " << __FUNCTION__ << " " << __LINE__ << " : " << endl;
    cout << " REVERSE LOOKUP FAILED " << endl;
    cout << __FILE__ << " " << __FUNCTION__ << " " << __LINE__ << " : " << endl;
    cout << __FILE__ << " " << __FUNCTION__ << " " << __LINE__ << " : " << endl;
    exit(0);
  }

  return ixyzReverse[reverse];
}

///////////////////////////////////////////////////////////////////////
// Build up the velocity basis matrix
///////////////////////////////////////////////////////////////////////
void buildVorticityBasis(const string& filename)
{
  TIMER functionTimer(__FUNCTION__);
  vector<VECTOR> columns(ixyz.size());
#pragma omp parallel
#pragma omp for  schedule(dynamic)
  for (unsigned int x = 0; x < ixyz.size(); x++)
  {
    int i  = ixyz[x][0];
    int k1 = ixyz[x][1];
    int k2 = ixyz[x][2];
    int k3 = ixyz[x][3];
    cout << " Making vorticity function (" << i << ", " << k1 << "," << k2 << "," << k3 << ")" << endl;
    velocityField.vorticity(i,k1,k2,k3);
    VECTOR column = velocityField.flattened();
    columns[x] = column;
  }
  cout << " Built " << ixyz.size() << " vorticity functions " << endl;

  vorticityU = MATRIX(columns);
  //vorticityU.write("./data/vorticityU.matrix");
  vorticityU.write(filename.c_str());
}

///////////////////////////////////////////////////////////////////////
// Build up the velocity basis matrix
///////////////////////////////////////////////////////////////////////
void buildVelocityBasis(const string& filename)
{
  TIMER functionTimer(__FUNCTION__);

  int totalCells = velocityField.totalCells();
  velocityU = MATRIX(totalCells * 3, ixyz.size());

  int totalZeroColumns = 0;
  int zerosSeen = 0;

  cout << " Making eigenfunctions: " << flush;
  //vector<VECTOR> columns(ixyz.size());
//#pragma omp parallel
//#pragma omp for  schedule(dynamic)
  for (unsigned int x = 0; x < ixyz.size(); x++)
  {
    int i  = ixyz[x][0];
    int k1 = ixyz[x][1];
    int k2 = ixyz[x][2];
    int k3 = ixyz[x][3];
    //velocityField.eigenfunctionUnscaled(i,k1,k2,k3);
    velocityField.eigenfunctionFFTW(i,k1,k2,k3);
    VECTOR column = velocityField.flattened();

    assert(column.size() == totalCells * 3);
    for (int y = 0; y < column.size(); y++)
      velocityU(y,x) = column[y];
    //columns[x] = column;

    Real norm2 = column.norm2();

    if (norm2 < 1e-8)
      totalZeroColumns++;

    //cout << " (" << i << ", " << k1 << "," << k2 << "," << k3 << ")" << flush;
    //cout << " Column " << x << " norm: " << column.norm2() << endl;
    
    int k1Zero = k1 == 0;
    int k2Zero = k2 == 0;
    int k3Zero = k3 == 0;
    int sum = k1Zero + k2Zero + k3Zero;
    if (sum >= 2)
    {
      cout << " (" << i << ", " << k1 << "," << k2 << "," << k3 << ")" << flush;
      cout << " Column " << x << " norm: " << column.norm2() << endl;
      zerosSeen++;
    }
    else if (ixyz[x][i+1] == 0)
    {
      cout << " (" << i << ", " << k1 << "," << k2 << "," << k3 << ")" << flush;
      cout << " Column " << x << " norm: " << column.norm2() << endl;
      zerosSeen++;
    }

    /*
    if (norm2 <= 0 && x > 40)
    {
      FIELDVIEW3D(velocityField.scalarField(0));
      exit(0);
    }
    */
  }
  cout << endl;
  cout << " Built " << ixyz.size() << " eigenfunctions " << endl;
  cout << " Saw      " << totalZeroColumns << " zero columns " << endl;
  cout << " Detected " << zerosSeen << " zero columns " << endl;

  /*
  cout << " Building U ..." << flush;
  velocityU = MATRIX(columns);
  cout << " done. " << endl;
  */
  /*
  cout << " Writing to file " << filename.c_str() << " ..." << flush;
  velocityU.write(filename.c_str());
  cout << " done. " << endl;
  */
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
void printSliceC(const int slice)
{
  MATRIX slab;
  cout << " Print C slice " << slice << ": " << endl;
  slab = C.slab(slice);
  for (int x = 0; x < slab.rows() * slab.cols(); x++)
    slab.data()[x] = (fabs(slab.data()[x]) < 1e-6) ? 0 : slab.data()[x];
  cout << slab << endl;
}

///////////////////////////////////////////////////////////////////////
// checking the summation conditions
///////////////////////////////////////////////////////////////////////
inline bool c100(const int a, const int b, const int k) { return ((a - b - k) == 0); }
inline bool c110(const int a, const int b, const int k) { return ((a + b - k) == 0); }
inline bool c101(const int a, const int b, const int k) { return ((a - b + k) == 0); }

///////////////////////////////////////////////////////////////////////
// build the coefficients for each component of vorticity
///////////////////////////////////////////////////////////////////////
void vorticityCoefficients(const vector<int>& i123, Real& kappa1, Real& kappa2, Real& kappa3)
{
  const int iComponent = i123[0];
  const int i1 = i123[1];
  const int i2 = i123[2];
  const int i3 = i123[3];

  if (iComponent == 0)
  {
    kappa1 = 0;
    kappa2 = i3;
    kappa3 = -i2;
    return;
  }
  else if (iComponent == 1)
  {
    kappa1 = i3;
    kappa2 = 0;
    kappa3 = -i1;
    return;
  }
  else if (iComponent == 2)
  {
    kappa1 = i2;
    kappa2 = -i1;
    kappa3 = 0;
  }
}

///////////////////////////////////////////////////////////////////////
// build the coefficients for each component of the eigenfunction
///////////////////////////////////////////////////////////////////////
void eigenfunctionCoefficients(const vector<int>& i123, Real& alpha1, Real& alpha2, Real& alpha3)
{
  const int iComponent = i123[0];
  const int i1 = i123[1];
  const int i2 = i123[2];
  const int i3 = i123[3];

  if (iComponent == 0)
  {
    alpha1 = -(i2 * i2 + i3 * i3);
    alpha2 = i1 * i2;
    alpha3 = i1 * i3;
  }
  else if (iComponent == 1)
  {
    alpha1 = -i1 * i2;
    alpha2 = i1 * i1 + i3 * i3;
    alpha3 = -i2 * i3;
  }
  else if (iComponent == 2)
  {
    alpha1 = i1 * i3;
    alpha2 = i2 * i3;
    alpha3 = -(i1 * i1 + i2 * i2);
  }

#if 0
  Real eig = 1.0 / (i1 * i1 + i2 * i2 + i3 * i3);
  alpha1 *= eig;
  alpha2 *= eig;
  alpha3 *= eig;
#endif
}

///////////////////////////////////////////////////////////////////////
// compute a single structure coefficient analytically
///////////////////////////////////////////////////////////////////////
Real structureCoefficientAnalytic(const vector<int>& a123, const vector<int>& b123, const vector<int>& k123)
{
  const int a1 = a123[1];
  const int a2 = a123[2];
  const int a3 = a123[3];
  
  const int b1 = b123[1];
  const int b2 = b123[2];
  const int b3 = b123[3];
  
  const int k1 = k123[1];
  const int k2 = k123[2];
  const int k3 = k123[3];

  // look at all three conditions for each component
  const bool c100_1 = c100(a1,b1,k1);
  const bool c110_1 = c110(a1,b1,k1);
  const bool c101_1 = c101(a1,b1,k1);
  const bool or_1 = c100_1 || c110_1 || c101_1;
  if (!or_1) return 0;
  
  const bool c100_2 = c100(a2,b2,k2);
  const bool c110_2 = c110(a2,b2,k2);
  const bool c101_2 = c101(a2,b2,k2);
  const bool or_2 = c100_2 || c110_2 || c101_2;
  if (!or_2) return 0;
  
  const bool c100_3 = c100(a3,b3,k3);
  const bool c110_3 = c110(a3,b3,k3);
  const bool c101_3 = c101(a3,b3,k3);
  const bool or_3 = c100_3 || c110_3 || c101_3;
  if (!or_3) return 0;

  // build the coefficients
  Real alpha1, alpha2, alpha3;
  eigenfunctionCoefficients(a123, alpha1, alpha2, alpha3);
  Real beta1, beta2, beta3;
  eigenfunctionCoefficients(b123, beta1, beta2, beta3);
  Real kappa1, kappa2, kappa3;
  vorticityCoefficients(k123, kappa1, kappa2, kappa3);

  // resolve the plus minus products
  const int a3b2k1 = (c100_1 + c110_1 + c101_1) * 
                     (-c100_2 + c110_2 + c101_2) * 
                     (c100_3 + c110_3 - c101_3);
  const int a3b1k2 = (-c100_1 + c110_1 + c101_1) * 
                     (c100_2 + c110_2 + c101_2) * 
                     (c100_3 + c110_3 - c101_3);
  const int a2b2k1 = (c100_1 + c110_1 + c101_1) * 
                     (c100_2 + c110_2 - c101_2) * 
                     (-c100_3 + c110_3 + c101_3);
  const int a2b3k1 = a2b2k1;
  const int a1b2k2 = (c100_1 + c110_1 - c101_1) * 
                     (c100_2 + c110_2 + c101_2) * 
                     (-c100_3 + c110_3 + c101_3);
  const int a1b3k2 = a1b2k2;
  const int a2b1k3 = (-c100_1 + c110_1 + c101_1) * 
                     (c100_2 + c110_2 - c101_2) * 
                     (c100_3 + c110_3 + c101_3);
  const int a1b2k3 = (c100_1 + c110_1 - c101_1) * 
                     (-c100_2 + c110_2 + c101_2) * 
                     (c100_3 + c110_3 + c101_3);

  Real final = 0;

  if (kappa1 != 0)
  {
    final += -alpha3 * beta2 * kappa1 * a3b2k1;
    final +=  alpha2 * beta3 * kappa1 * a2b3k1;
  }
  if (kappa2 != 0)
  {
    final +=  alpha3 * beta1 * kappa2 * a3b1k2;
    final += -alpha1 * beta3 * kappa2 * a1b3k2;
  }
  if (kappa3 != 0)
  {
    final += -alpha2 * beta1 * kappa3 * a2b1k3;
    final +=  alpha1 * beta2 * kappa3 * a1b2k3;
  }

  //final *= (1.0/ 64.0) * M_PI * M_PI * M_PI;
  final *= (1.0/ 64.0);
  final *= 1.0 / (b1 * b1 + b2 * b2 + b3 * b3);

  return final;
}

///////////////////////////////////////////////////////////////////////
// build the structure coefficient tensor using analytic formulae
///////////////////////////////////////////////////////////////////////
void buildSparseAnalyticC_OMP()
{
  TIMER functionTimer(__FUNCTION__);
  cout << " Building C ... " << flush;
  int basisRank = ixyz.size();
  int totalEntries = basisRank * basisRank * basisRank;

  // set up for threaded version
  int threads = omp_get_max_threads();
  vector<SPARSE_TENSOR3> tempC(threads);
  for (int x = 0; x < threads; x++)
    tempC[x].resize(basisRank, basisRank, basisRank);
  vector<int> nonZeros(threads);

#pragma omp parallel
#pragma omp for  schedule(dynamic)
  for (int d1 = 0; d1 < basisRank; d1++)
  {
    int threadID = omp_get_thread_num();

    vector<int> a123 = ixyz[d1];
    for (int d2 = 0; d2 < basisRank; d2++) 
    {
      vector<int> b123 = ixyz[d2];

      int a = reverseLookup(a123);
      int b = reverseLookup(b123);

      for (int d3 = 0; d3 < basisRank; d3++)
      {
        vector<int> k123 = ixyz[d3];
        int k = reverseLookup(k123);
        
        Real coef = structureCoefficientAnalytic(a123, b123, k123);

        if (fabs(coef) > 1e-8)
        {
          tempC[threadID](b,a,k) = -coef;
          nonZeros[threadID]++;
        }
      }
    }

    cout << d1 << " " << flush;
  }
  cout << "done. " << endl;

  // consolidate non-zero counts
  int totalNonZeros = 0;
  for (unsigned int x = 0; x < nonZeros.size(); x++)
    totalNonZeros += nonZeros[x];

  // consolidate tensors
  tensorC.resize(basisRank, basisRank, basisRank);
  for (unsigned int x = 0; x < tempC.size(); x++)
    tensorC += tempC[x];

  float percent = (100.0 * totalNonZeros) / totalEntries;
  cout << " Non-zeros " << totalNonZeros << " out of " << totalEntries << " (" << percent << "%)" << endl;
  
  // build arrays for fast static multiplies
  //for (int x = 0; x < basisRank; x++)
  //  sparseC[x].buildStatic();

  // build arrays for fast static multiplies
  tensorC.buildStatic();
}

///////////////////////////////////////////////////////////////////////
// build the structure coefficient tensor using analytic formulae
///////////////////////////////////////////////////////////////////////
void buildSparseAnalyticC()
{
  TIMER functionTimer(__FUNCTION__);
  cout << " Building C ... " << flush;
  int basisRank = ixyz.size();
  int nonZeros = 0;
  int totalEntries = 0;

  //sparseC.clear();
  tensorC.resize(basisRank, basisRank, basisRank);

#pragma omp parallel
#pragma omp for  schedule(dynamic)
  for (int d1 = 0; d1 < basisRank; d1++)
  {
    vector<int> a123 = ixyz[d1];
    for (int d2 = 0; d2 < basisRank; d2++) 
    {
      vector<int> b123 = ixyz[d2];

      int a = reverseLookup(a123);
      int b = reverseLookup(b123);

      for (int d3 = 0; d3 < basisRank; d3++)
      {
        vector<int> k123 = ixyz[d3];
        int k = reverseLookup(k123);
        
        Real coef = structureCoefficientAnalytic(a123, b123, k123);

        if (fabs(coef) > 1e-8)
        {
#pragma omp critical
          tensorC(b,a,k) = -coef;

          //sparseC[k](b,a) = coef;
#pragma omp atomic
          nonZeros++;
        }
        totalEntries++;
        //cout << " k1: " << k1 << " k2: " << k2 << " a1: " << a1 << " a2: " << a2 << " b1: " << b1 << " b2: " << b2 << " coef: " << coef << endl;
      }
    }
    cout << d1 << " " << flush;
  }
  cout << "done. " << endl;

  float percent = (100.0 * nonZeros) / totalEntries;
  cout << " Non-zeros " << nonZeros << " out of " << totalEntries << " (" << percent << "%)" << endl;
  
  // build arrays for fast static multiplies
  //for (int x = 0; x < basisRank; x++)
  //  sparseC[x].buildStatic();

  // build arrays for fast static multiplies
  tensorC.buildStatic();
}

///////////////////////////////////////////////////////////////////////
// draw coordinate axes
///////////////////////////////////////////////////////////////////////
void drawAxes()
{
  // draw coordinate axes
  glPushMatrix();
  glTranslatef(-0.1f, -0.1f, -0.1f);
  glLineWidth(3.0f);
  glBegin(GL_LINES);
    // x axis is red
    glColor4f(10.0f, 0.0f, 0.0f, 1.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glColor4f(10.0f, 0.0f, 0.0f, 0.0f);
    glVertex3f(1.0f, 0.0f, 0.0f);

    // y axis is green 
    glColor4f(0.0f, 10.0f, 0.0f, 1.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glColor4f(0.0f, 10.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 1.0f, 0.0f);
    
    // z axis is blue
    glColor4f(0.0f, 0.0f, 10.0f, 1.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glColor4f(0.0f, 0.0f, 10.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 1.0f);
  glEnd();
  glLineWidth(1.0f);
  glPopMatrix();
}

//////////////////////////////////////////////////////////////////////////////
// Translate screen space to world space
//
// Adapted from the NeHe page:
// http://nehe.gamedev.net/data/articles/article.asp?article=13
//////////////////////////////////////////////////////////////////////////////
VEC3F unproject(float x, float y, float z)
{
  GLint viewport[4];
	GLdouble modelview[16];
	GLdouble projection[16];

	glGetDoublev(GL_MODELVIEW_MATRIX, modelview);
	glGetDoublev(GL_PROJECTION_MATRIX, projection);
	glGetIntegerv(GL_VIEWPORT, viewport);

  double worldX, worldY, worldZ;
	gluUnProject(x, viewport[3] - y, z,
               modelview, projection, viewport, 
               &worldX, &worldY, &worldZ);

  return VEC3F(worldX, worldY, worldZ);
}

///////////////////////////////////////////////////////////////////////
// GL and GLUT callbacks
///////////////////////////////////////////////////////////////////////
void glutDisplay()
{
  glvu.BeginFrame();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);

    glPushMatrix();
      velocityField.drawBoundingBox();
    glPopMatrix();

    glColor4f(3,0,0,1);
#if 0
    glPointSize(1);
    glBegin(GL_POINTS);
      for (int x = 0; x < particles.size(); x++)
      {
        VEC3F velocity = velocityField(particles[x]);
        glNormal3f(velocity[0], velocity[1], velocity[2]);
        glVertex3f(particles[x][0], particles[x][1], particles[x][2]);
      }
    glEnd();
#else
    for (unsigned int x = 0; x < ribbons.size(); x++)
    {
      //VEC3F velocity = velocityField(particles[x]);
      //glNormal3f(velocity[0], velocity[1], velocity[2]);
      glLineWidth(2);
      glBegin(GL_LINE_STRIP);
        list<VEC3F>::iterator iter;
        list<VEC3F>::iterator iterVelocity = velocityRibbons[x].begin();
        for (iter = ribbons[x].begin(); iter != ribbons[x].end(); iter++)
        {
          VEC3F normalized = *iterVelocity;

          normalized.normalize();

          //glNormal3f(normalized[0], normalized[1], normalized[2]);
          //glColor4f(fabs(normalized[0]), fabs(normalized[1]), fabs(normalized[2]), 1.0);
          glColor4f(fabs(normalized[0]), fabs(normalized[1]), fabs(normalized[2]), 0.25);
          //glNormal3f((*iterVelocity)[0], (*iterVelocity)[1], (*iterVelocity)[2]);
          glVertex3f((*iter)[0], (*iter)[1], (*iter)[2]);
        }
      glEnd();
    }
#endif

    glPushMatrix();
      glTranslatef(-0.5, -0.5, -0.5);
      drawAxes();
    glPopMatrix();
  glvu.EndFrame();
}

///////////////////////////////////////////////////////////////////////
// animate and display new result
///////////////////////////////////////////////////////////////////////
void glutIdle()
{
  if (animate)
    runEverytime();

  glutPostRedisplay();
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
void glutKeyboard(unsigned char key, int x, int y)
{
  switch (key)
  {
    case 'a':
      animate = !animate;
      break;
    case 'q':
      TIMER::printTimings();
      TIMER::printRawTimingsPerFrame(frames);
      exit(0);
      break;
    case 'v':
      {
        Camera* camera = glvu.GetCurrentCam();
        glvuVec3f eye;
        glvuVec3f ref;
        glvuVec3f up;
        camera->GetLookAtParams(&eye, &ref, &up);
        cout << " Eye(" << eye[0] << ", " << eye[1] << ", " << eye[2]  << ");" << endl;
        cout << " LookAtCntr(" << ref[0] << ", " << ref[1] << ", " << ref[2]  << ");" << endl;
        cout << " Up(" << up[0] << ", " << up[1] << ", " << up[2]  << ");" << endl;
      }
      break;
    case ' ':
      TIMER::printRawTimingsPerFrame(frames);
      break;
    default:
      break;
  }
  glvu.Keyboard(key,x,y);
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
void glutSpecial(int key, int x, int y)
{
  switch (key)
  {
    case GLUT_KEY_LEFT:
      break;
    case GLUT_KEY_RIGHT:
      break;
    case GLUT_KEY_UP:
      break;
    case GLUT_KEY_DOWN:
      break;
  }
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
void glutMouseClick(int button, int state, int x, int y)
{
  glvu.Mouse(button,state,x,y);
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
void glutMouseMotion(int x, int y)
{
  glvu.Motion(x,y);
}

//////////////////////////////////////////////////////////////////////////////
// open the GLVU window
//////////////////////////////////////////////////////////////////////////////
int glvuWindow()
{
  char title[] = "3D Viewer";
  glvu.Init(title,
            GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH,
            0, 0, 800, 800);

  glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
  GLfloat lightZeroPosition[] = {10.0, 4.0, 10.0, 1.0};
  GLfloat lightZeroColor[] = {1.0, 1.0, 1.0, 1.0};
  glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, 1);
  glLightfv(GL_LIGHT0, GL_POSITION, lightZeroPosition);
  glLightfv(GL_LIGHT0, GL_DIFFUSE, lightZeroColor);
 
  GLfloat lightColor[] = {1.0, 1.0, 1.0, 1.0};
  for (int x = 0; x < 4; x++)
  {
    GLfloat lightPosition[] = {0,0,0,1.0};
    lightPosition[x] = 10;

    if (x > 2)
      lightPosition[x] *= -1;
    glLightfv(GL_LIGHT1, GL_POSITION, lightPosition);
    glLightfv(GL_LIGHT1, GL_DIFFUSE, lightColor);
  }

  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); 
  //glEnable(GL_LIGHTING);
  //glEnable(GL_LIGHT0);
  glEnable(GL_COLOR_MATERIAL);
  glEnable(GL_DEPTH_TEST);
  glShadeModel(GL_SMOOTH);
  glClearColor(0,0,0,0);

  glutDisplayFunc(&glutDisplay);
  glutIdleFunc(&glutIdle);
  glutKeyboardFunc(&glutKeyboard);
  glutSpecialFunc(&glutSpecial);
  glutMouseFunc(&glutMouseClick);
  glutMotionFunc(&glutMouseMotion);

  glEnable (GL_BLEND);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  glvuVec3f ModelMin(-10,-10,-10), ModelMax(10,10,10), 
        //Eye(1.59445, 1.6929, 2.40504), LookAtCntr(1.15667, 1.21574, 1.64302), Up(-0.150235, 0.874454, -0.461259);
        //Eye(1.39197, 1.37948, 2.14899), LookAtCntr(0.947247, 0.903024, 1.39057), Up(-0.150233, 0.874455, -0.461256);
        Eye(1.00005, 0.994277, 1.81856), LookAtCntr(0.571001, 0.516282, 1.05212), Up(-0.150234, 0.874455, -0.461257);

  float Yfov = 45;
  float Aspect = 1;
  float Near = 0.001f;
  float Far = 10.0f;
  glvu.SetAllCams(ModelMin, ModelMax, Eye, LookAtCntr, Up, Yfov, Aspect, Near, Far);

  //glvuVec3f center(0.25, 0.25, 0.25);
  //glvuVec3f center(0.5, 0.5, 0.5);
  glvuVec3f center(0, 0, 0);
  glvu.SetWorldCenter(center);

  glutMainLoop();

  // Control flow will never reach here
  return EXIT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////
// compute the FFT version of each velocity field
///////////////////////////////////////////////////////////////////////
void computeFieldFFTs()
{
  TIMER functionTimer(__FUNCTION__);
  cout << " Computing FFTs ... " << flush;
  VECTOR3_FIELD_3D columnField(res, res, res);

  int size = ixyz.size();
  xDeltaIndices.resize(size);
  xMagnitudes.resize(size);
  xProjection.resize(size);
  yDeltaIndices.resize(size);
  yMagnitudes.resize(size);
  yProjection.resize(size);
  zDeltaIndices.resize(size);
  zMagnitudes.resize(size);
  zProjection.resize(size);

//#pragma omp parallel
//#pragma omp for  schedule(dynamic)
  for (int x = 0; x < ixyz.size(); x++)
  {
    int i  = ixyz[x][0];
    int k1 = ixyz[x][1];
    int k2 = ixyz[x][2];
    int k3 = ixyz[x][3];
    columnField.eigenfunctionFFTW(i,k1,k2,k3);
    //columnField.eigenfunctionUnscaledFFTW(i,k1,k2,k3);
    FIELD_3D xField = columnField.scalarField(0);
    FIELD_3D yField = columnField.scalarField(1);
    FIELD_3D zField = columnField.scalarField(2);
    
    xField.xDSTyDCTzDCT();
    yField.xDCTyDSTzDCT();
    zField.xDCTyDCTzDST();
 
    int xIndex = xField.maxAbsIndex();
    xDeltaIndices[x] = xIndex;
    xMagnitudes[x] = xField[xIndex];
    //xMagnitudesInv[x] = (xField[xIndex] != 0) ? 1.0 / xField[xIndex] : 0;

    /*
    if (fabs((xField[xIndex]) - 33791) < 1e-3)
    {
      cout << " index: " << xIndex << endl;
      cout << " int: " << (int)xField[xIndex] << endl;
      cout << " fabs: " << fabs(xField[xIndex]) << endl;
      cout << " diff: " << (int)xField[xIndex] - 33791 << endl;
      cout << " magnitude: " << xField[xIndex] << endl;
      FIELDVIEW3D(xField);
      //FIELDVIEW3D(yField);
      //FIELDVIEW3D(zField);
      exit(0);
    }
    */
    
    int yIndex = yField.maxAbsIndex();
    yDeltaIndices[x] = yIndex;
    yMagnitudes[x] = yField[yIndex];
    //yMagnitudesInv[x] = (yField[yIndex] != 0) ? 1.0 / yField[yIndex] : 0;
    
    int zIndex = zField.maxAbsIndex();
    zDeltaIndices[x] = zIndex;
    zMagnitudes[x] = zField[zIndex];
    //zMagnitudesInv[x] = (zField[zIndex] != 0) ? 1.0 / zField[zIndex] : 0;
    cout << x << " " << flush;

    /*
    Real alpha1, alpha2, alpha3;
    eigenfunctionCoefficients(ixyz[x], alpha1, alpha2, alpha3);
    Real inv = 1.0 / (k1 * k1 + k2 * k2 + k3 * k3);

    VEC3I xTriplet;
    VEC3I yTriplet;
    VEC3I zTriplet;
    xField.maxAbsIndex(xTriplet);
    yField.maxAbsIndex(yTriplet);
    zField.maxAbsIndex(zTriplet);
    cout << " i: " << i << " k1: " << k1 << " k2: " << k2 << " k3: " << k3 << endl;
    //cout << " x index: " << xIndex << ", " << xTriplet[0] << " " << xTriplet[1] << " " << xTriplet[2] << " magnitude: " << xMagnitudes[x] << endl;
    //cout << " y index: " << yIndex << ", " << yTriplet[0] << " " << yTriplet[1] << " " << yTriplet[2] << " magnitude: " << yMagnitudes[x] << endl;
    //cout << " z index: " << zIndex << ", " << zTriplet[0] << " " << zTriplet[1] << " " << zTriplet[2] << " magnitude: " << zMagnitudes[x] << endl;
    cout << " x index: " << xTriplet[0] << " " << xTriplet[1] << " " << xTriplet[2] << " magnitude: " << xMagnitudes[x] << endl;
    cout << " y index: " << yTriplet[0] << " " << yTriplet[1] << " " << yTriplet[2] << " magnitude: " << yMagnitudes[x] << endl;
    cout << " z index: " << zTriplet[0] << " " << zTriplet[1] << " " << zTriplet[2] << " magnitude: " << zMagnitudes[x] << endl;
    //cout << " res ^2 * (res + 1): " << res * res * (res + 1) << endl;
    Real xGuess = alpha1 * res * res * (res + 1)  * inv;
    Real yGuess = alpha2 * res * res * (res + 1)  * inv;
    Real zGuess = alpha3 * res * res * (res + 1)  * inv;

    if (k1 == 0)
    {
      xGuess = 0;
      yGuess *= 2;
      zGuess *= 2;
    }
    if (k2 == 0)
    {
      xGuess *= 2;
      yGuess = 0;
      zGuess *= 2;
    }
    if (k3 == 0)
    {
      xGuess *= 2;
      yGuess *= 2;
      zGuess = 0;
    }
    cout << " x guess: " << xGuess << endl;
    cout << " y guess: " << yGuess << endl;
    cout << " z guess: " << zGuess << endl;

    assert(fabs(xGuess - xMagnitudes[x]) < 1e-7);
    assert(fabs(yGuess - yMagnitudes[x]) < 1e-7);
    assert(fabs(zGuess - zMagnitudes[x]) < 1e-7);

    Real alpha = k1;
    Real beta = k2;
    Real gamma = k3;
    int slabSize = res * res;
    if (fabs(xMagnitudes[x]) > 0)
    {
      cout << " alpha: " << alpha - 1 << " triplet: " << xTriplet[0] << endl;
      cout << " beta:  " << beta  << " triplet: " << xTriplet[1] << endl;
      cout << " gamma: " << gamma << " triplet: " << xTriplet[2] << endl;
      cout << " x index:       " << xDeltaIndices[x] << endl;
      cout << " x index guess: " << alpha + beta * res + gamma * slabSize << endl;
      cout << " field dims: " << xField.xRes() << " " << xField.slabSize() << endl;
      assert(xTriplet[0] == alpha - 1);
      assert(xTriplet[1] == beta);
      assert(xTriplet[2] == gamma);
    }
    if (fabs(yMagnitudes[x]) > 0)
    {
      assert(yTriplet[0] == alpha);
      assert(yTriplet[1] == beta - 1);
      assert(yTriplet[2] == gamma);
    }
    if (fabs(zMagnitudes[x]) > 0)
    {
      assert(zTriplet[0] == alpha);
      assert(zTriplet[1] == beta);
      assert(zTriplet[2] == gamma - 1);
    }
    cout << endl;
    */
  }

  cout << "done. " << endl;

  /*
  cout << __FILE__ << " " << __FUNCTION__ << " " << __LINE__ << " : " << endl;
  cout << __FILE__ << " " << __FUNCTION__ << " " << __LINE__ << " : " << endl;
  exit(0);
  */
}

///////////////////////////////////////////////////////////////////////
// compute the delta and magnitude of each eigenfunction directly
///////////////////////////////////////////////////////////////////////
void computeDeltas()
{
  TIMER functionTimer(__FUNCTION__);
  cout << " Computing deltas ... " << flush;

  int size = ixyz.size();
  xDeltaIndices.resize(size);
  xMagnitudes.resize(size);
  xProjection.resize(size);
  yDeltaIndices.resize(size);
  yMagnitudes.resize(size);
  yProjection.resize(size);
  zDeltaIndices.resize(size);
  zMagnitudes.resize(size);
  zProjection.resize(size);

  Real projectionFactor = 1.0 / (res * res * (res + 1)) / 8.0;

//#pragma omp parallel
//#pragma omp for  schedule(dynamic)
  for (int x = 0; x < ixyz.size(); x++)
  {
    int i  = ixyz[x][0];
    int k1 = ixyz[x][1];
    int k2 = ixyz[x][2];
    int k3 = ixyz[x][3];

    Real alpha1, alpha2, alpha3;
    eigenfunctionCoefficients(ixyz[x], alpha1, alpha2, alpha3);
    Real inv = 1.0 / (k1 * k1 + k2 * k2 + k3 * k3);

    Real xGuess = alpha1 * res * res * (res + 1) * inv;
    Real yGuess = alpha2 * res * res * (res + 1) * inv;
    Real zGuess = alpha3 * res * res * (res + 1) * inv;

    if (k1 == 0)
    {
      xGuess = 0;
      yGuess *= 2;
      zGuess *= 2;
    }
    if (k2 == 0)
    {
      xGuess *= 2;
      yGuess = 0;
      zGuess *= 2;
    }
    if (k3 == 0)
    {
      xGuess *= 2;
      yGuess *= 2;
      zGuess = 0;
    }

    int slabSize = res * res;
    int xIndex = (k1 - 1) + k2 * res + k3 * slabSize;
    int yIndex = k1 + (k2 - 1) * res + k3 * slabSize;
    int zIndex = k1 + k2 * res + (k3 - 1) * slabSize;
   
    // take the product since we want to see later if one was zero
    int kProduct = k1 * k2 * k3;

    xDeltaIndices[x] = xIndex;
    xMagnitudes[x] = xGuess;
    xProjection[x] = xGuess * projectionFactor;
    xProjection[x] *= (kProduct) ? 1.0 : 0.5;

    yDeltaIndices[x] = yIndex;
    yMagnitudes[x] = yGuess;
    yProjection[x] = yGuess * projectionFactor;
    yProjection[x] *= (kProduct) ? 1.0 : 0.5;
    
    zDeltaIndices[x] = zIndex;
    zMagnitudes[x] = zGuess;
    zProjection[x] = zGuess * projectionFactor;
    zProjection[x] *= (kProduct) ? 1.0 : 0.5;
  }
  cout << "done. " << endl;
}

///////////////////////////////////////////////////////////////////////
// build inverse plans
///////////////////////////////////////////////////////////////////////
void computePlans()
{
  cout << " Computing plans ..." << flush;
  xHat.resizeAndWipe(res,res,res);
  yHat.resizeAndWipe(res,res,res);
  zHat.resizeAndWipe(res,res,res);
  
  xHat.planISICIC(xPlanIDSTyIDCTzIDCT);
  yHat.planICISIC(yPlanIDCTyIDSTzIDCT);
  zHat.planICICIS(zPlanIDCTyIDCTzIDST);

  cout << " done." << endl;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
void runEverytime()
{
  stepEigenfunctions();
  
  TIMER advectionTimer("Particle advection");
#pragma omp parallel
#pragma omp for  schedule(dynamic)
  for (unsigned int x = 0; x < particles.size(); x++)
  {
    particles[x] += dt * velocityField(particles[x]);

    if (ribbons[x].size() > 20)
    //if (ribbons[x].size() > 2)
      ribbons[x].pop_front();
    ribbons[x].push_back(particles[x]);

    if (velocityRibbons[x].size() > 20)
    //if (velocityRibbons[x].size() > 2)
      velocityRibbons[x].pop_front();
    velocityRibbons[x].push_back(velocityField(particles[x]));
  }
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
  //debugCoeff();

  buildTableIXYZ(dim);

  char buffer[256];
  bool success = true;

  sprintf(buffer, "./data/C.res.%i.dim.%i.tensor", res, dim);
  /*
  success = C.read(buffer);
  if (!success)
    buildC(buffer);
  else
    cout << buffer << " found! " << endl;
    */

  //printSliceC(26);
  //buildAnalyticC();
  //buildSparseAnalyticC();
  buildSparseAnalyticC_OMP();
  TIMER::printTimings();

  // this is not actually needed at runtime
  /*
  sprintf(buffer, "./data/vorticity.res.%i.dim.%i.matrix", res, dim);
  success = vorticityU.read(buffer);
  if (!success)
    buildVorticityBasis(buffer);
  else
    cout << buffer << " found! " << endl;
    */

  //buildVelocityBasis(buffer);
  /*
  sprintf(buffer, "./data/velocity.res.%i.dim.%i.matrix", res, dim);
  success = velocityU.read(buffer);
  if (!success)
    buildVelocityBasis(buffer);
  else
    cout << buffer << " found! " << endl;
    */

  computePlans();
  //computeFieldFFTs();
  computeDeltas();

  w = VECTOR(ixyz.size());
  wDot = VECTOR(ixyz.size());

  VECTOR3_FIELD_3D impulse(velocityField);
  impulse = 0;
  int half = impulse.xRes() / 2;
  impulse(half, half, half)[0] = 0;
  impulse(half, half, half)[1] = 1;
  impulse(half, half, half)[2] = 0;
  //impulse(half, half, half)[1] = 0.05;
  //impulse(half, half, half)[0] = 0.01;
  //impulse(half, half, half)[0] = 0.001;
  //impulse(half, half, half)[1] = 0.01;
  //impulse(half, half, half)[1] = 1;

  FIELD_3D xField = impulse.scalarField(0);
  FIELD_3D yField = impulse.scalarField(1);
  FIELD_3D zField = impulse.scalarField(2);
  xField.xDSTyDCTzDCT();
  yField.xDCTyDSTzDCT();
  zField.xDCTyDCTzDST();

  VECTOR force(w.size());
  for (int x = 0; x < xDeltaIndices.size(); x++)
    force[x] += xProjection[x] * xField[xDeltaIndices[x]];
  for (int x = 0; x < yDeltaIndices.size(); x++)
    force[x] += yProjection[x] * yField[yDeltaIndices[x]];
  for (int x = 0; x < zDeltaIndices.size(); x++)
    force[x] += zProjection[x] * zField[zDeltaIndices[x]];

  /*
  // sanity check the projected force
  VECTOR impulseFlattened = impulse.flattened();
  VECTOR projected = velocityU ^ impulseFlattened;
  VECTOR diff = projected - force;
  cout << " Projection diff: " << diff.norm2() << endl;
  */
  
  // set equal to the force (should scale by dt ...)
  w = force;

  // fire down some random particles
  MERSENNETWISTER twister(123456);
  //int totalParticles = 40000;
  int totalParticles = 10000;
  //int totalParticles = 1000;
  for (int x = 0; x < totalParticles; x++)
  {
    VEC3F particle;
    //particle[0] = 0.5;
    particle[0] = twister.rand();
    //particle[1] = 0.5;
    particle[1] = twister.rand();
    //particle[2] = 0.5;
    particle[2] = twister.rand();
    particle -= VEC3F(0.5, 0.5, 0.5);

    particles.push_back(particle);
  }
  ribbons.resize(totalParticles);
  velocityRibbons.resize(totalParticles);

  glutInit(&argc, argv);
  glvuWindow();

  return 1;
}
