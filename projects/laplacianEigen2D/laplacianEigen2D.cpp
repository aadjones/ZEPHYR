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

int basisRank = 16;
//int basisRank = 9;
//int basisRank = 17;
MATRIX velocityU;
MATRIX vorticityU;
TENSOR3 C;
VECTOR w;
VECTOR wDot;
Real dt = 0.0001;
//Real dt = 0.01;
vector<pair<int, int> > ijPairs;
map<pair<int, int>, int > ijPairsReverse;

void updateViewingTexture();

///////////////////////////////////////////////////////////////////////
// step the system forward in time
///////////////////////////////////////////////////////////////////////
void stepEigenfunctions()
{
  TIMER functionTimer(__FUNCTION__);
  Real e1 = w.dot(w);

  for (int k = 0; k < basisRank; k++)
  {
    MATRIX slab = C.slab(k);
    wDot[k] = w.dot(slab * w);
  }

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
    const Real viscosity = 0.0;

    // diffuse
    w[k] *= exp(lambda * dt * viscosity);
  }

  // goose the lowest mode?
  //cout << "w0: " << w[0] << endl;
  //w[0] += 1;

  TIMER reconstructTimer("Velocity Reconstruction");
  VECTOR final = velocityU * w;
  vectorField2D.unflatten(final);
  vectorField2D.stompBorder();
  reconstructTimer.stop();
 
  // visualize...
  //distanceField = vectorField2D.LIC();
  //updateViewingTexture();
}

///////////////////////////////////////////////////////////////////////
// build a list of the i,j pairs used to build the eigenfunctions
///////////////////////////////////////////////////////////////////////
void buildIJPairs(int rank)
{
  ijPairs.clear();

  int columnsMade = 0;
  for (int i = 1; i < rank; i++)
    for (int j = 1; j <= i; j++)
    {
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
  for (int x = 0; x < ijPairs.size(); x++)
  {
    int i = ijPairs[x].first;
    int j = ijPairs[x].second;
    cout << " Making eigenfunction (" << i << ", " << j << ")" << endl;
    //vectorField2D.eigenfunction(i,j);
    vectorField2D.eigenfunctionFFTW(i,j);
    VECTOR column = vectorField2D.flatten();
    columns.push_back(column);
  }
  velocityU = MATRIX(columns);
  velocityU.write("./data/velocityU.matrix");
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
  C = TENSOR3(rank, rank, rank);

  /*
  for (int i = 0; i < rank; i++)
    for (int j = 0; j < rank; j++)
    {
      //VECTOR p = advect(vorticityU.getColumn(i), velocityU.getColumn(j));
      //VECTOR p = advect(vorticityU.getColumn(j), velocityU.getColumn(i));
      VECTOR p = advect(ijPairs[i], ijPairs[j]);
      cout << " Building C(" << i << ", " << j << ")" << endl;

      for (int k = 0; k < rank; k++)
        C(i,j,k) = p.dot(vorticityU.getColumn(k));
    }
  C.write("./data/C.tensor");
  */
  for (int d1 = 0; d1 < rank; d1++)
  {
    //int a1 = this.basis_lookup(d1,0);
    int a1 = ijPairs[d1].first;
    //int a2 = this.basis_lookup(d1,1);
    int a2 = ijPairs[d1].second;
    //cout << "a1: " << a1 << " a2: " << a2 << endl;
    int a1_2 = a1*a1;
    
    double lambda_a = -(a1*a1 + a2*a2);
    double inv_lambda_a = -1.0/(a1*a1 + a2*a2);
    
    for (int d2 = 0; d2 < rank; d2++) 
    {
      //int b1 = this.basis_lookup(d2,0);
      //int b2 = this.basis_lookup(d2,1);
      int b1 = ijPairs[d2].first;
      int b2 = ijPairs[d2].second;
      //cout << "b1: " << b1 << " b2: " << b2 << endl;

      double lambda_b = -(b1*b1 + b2*b2);
      double inv_lambda_b = -1.0/(b1*b1 + b2*b2);

      //int k1 = this.basis_rlookup(a1,a2);
      int k1 = ijPairsReverse[pair<int,int>(a1,a2)];
      //int k2 = this.basis_rlookup(b1,b2);
      int k2 = ijPairsReverse[pair<int,int>(b1,b2)];
      //cout << "k1: " << k1 << " k2: " << k2 << endl;

      int antipairs[4][2];
      antipairs[0][0] = a1-b1; antipairs[0][1] = a2-b2;
      antipairs[1][0] = a1-b1; antipairs[1][1] = a2+b2;
      antipairs[2][0] = a1+b1; antipairs[2][1] = a2-b2;
      antipairs[3][0] = a1+b1; antipairs[3][1] = a2+b2;
      
      for (int c = 0;c < 4;c++) 
      {
        int i = antipairs[c][0];
        int j = antipairs[c][1];

        //int index = this.basis_rlookup(i,j);
        if (ijPairsReverse.find(pair<int,int>(i,j)) == ijPairsReverse.end())
          continue;

        int index = ijPairsReverse[pair<int,int>(i,j)];
          
        double coef = - coefdensity(a1,a2,b1,b2,c,0) * inv_lambda_b;
        //this.Ck[index].set(k1,k2,coef);
        C(k2,k1,index) = coef;
        //this.Ck[index].set(k2,k1,coef * -lambda_b/lambda_a);
        C(k1,k2,index) = coef * -lambda_b / lambda_a;
        //cout << "i: " << i << " j: " << j << endl;
        //cout << "index: " << index << " coef: " << coef << endl;

        /*
        //if (index == 0 && k1 == 1 && k2 == 6)
        if (index == 0 && k1 == 6 && k2 == 1)
        {
          cout << __FILE__ << " " << __FUNCTION__ << " " << __LINE__ << " : " << endl;
          cout << " coef: " << coef << endl;
          cout << " c: " << c << endl;
          cout << " k1 pair: " << a1 << " " << a2 << endl;
          cout << " k2 pair: " << b1 << " " << b2 << endl;
          cout << " antipairs: " << i << " " << j << endl;
          //cout << "index: " << index << " coef: " << coef << endl;
        } 
        //if (index == 0 && k1 == 3 && k2 == 8)
        if (index == 0 && k1 == 7 && k2 == 2)
        */
        if (fabs(coef) > 0.0 && index == 0)
        {
          cout << __FILE__ << " " << __FUNCTION__ << " " << __LINE__ << " : " << endl;
          cout << " coef: " << coef << endl;
          cout << " anti coef: " << coef * -lambda_b / lambda_a << endl;
          cout << " c: " << c << endl;
          cout << " k1: " << k1 << endl;
          cout << " k2: " << k2 << endl;
          cout << " k1 pair: " << a1 << " " << a2 << endl;
          cout << " k2 pair: " << b1 << " " << b2 << endl;
          cout << " antipairs: " << i << " " << j << endl;
          //cout << "index: " << index << " coef: " << coef << endl;
        }
      }
    }
  }
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

  glutSwapBuffers();
}

///////////////////////////////////////////////////////////////////////
// throw down some particles
///////////////////////////////////////////////////////////////////////
void seedParticles()
{
  static MERSENNETWISTER twister(123456);
  const VEC3F lengths = vectorField2D.lengths();

  for (int x = 0; x < 10000; x++)
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
      TIMER::printTimings();
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
    case 'q':
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
        //cout << " k1: " << k1 << " k2: " << k2 << " a1: " << a1 << " a2: " << a2 << " b1: " << b1 << " b2: " << b2 << " coef: " << coef << endl;
      }
    }
    cout << d1 << endl;
  }

  //cout << " newC = " << newC << endl;
  //cout << " numericalC= " << C.slab(0) << endl; 
  //exit(0);
}

///////////////////////////////////////////////////////////////////////
// build a matrix indexing convenience class
///////////////////////////////////////////////////////////////////////
class INDEX {
public:
  INDEX(int x, int y, int component) :
    _x(x), _y(y), _component(component) {};

  bool operator<(const INDEX& rhs) const
  {
    const int lhsIndex = (_x + _y * _xRes) * _totalComponents + _component;
    const int rhsIndex = (rhs._x + rhs._y * _xRes) * _totalComponents + rhs._component;

    return lhsIndex < rhsIndex;
  };

  int _x;
  int _y;
  int _component;
  static int _xRes;
  static int _totalComponents;
};

int INDEX::_xRes;
int INDEX::_totalComponents;

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
void stencilFriendlyDoubleCurl(VECTOR3_FIELD_2D& curlCurl, VECTOR3_FIELD_2D& oneOne)
{
  int xRes = curlCurl.xRes();
  int yRes = curlCurl.yRes();
  VEC3F lengths = curlCurl.lengths();
  Real dx = lengths[0] / (xRes - 1);
  Real dy = lengths[1] / (yRes - 1);

  // the non-verbose way, but more convenient for converting to a stencil
  for (int y = 1; y < yRes - 1; y++)
    for (int x = 1; x < xRes - 1; x++)
    {
      VEC3F middle = oneOne(x,y);
      VEC3F up    = oneOne(x,y + 1);
      VEC3F down  = oneOne(x,y - 1);
      VEC3F left  = oneOne(x - 1,y);
      VEC3F right = oneOne(x + 1,y);

      VEC3F upLeft  = oneOne(x - 1, y + 1);
      VEC3F upRight = oneOne(x + 1, y + 1);
      VEC3F downLeft  = oneOne(x - 1, y - 1);
      VEC3F downRight = oneOne(x + 1, y - 1);

      // x component:
      // -\frac{\partial^2 F_x}{\partial y^2} + 
      //  \frac{\partial^2 F_y}{\partial x \partial y}
      Real xDyDy = (up[0] -2.0 * middle[0] + down[0]) / (dx * dx);
      Real yDxDy = (upRight[1] + downLeft[1] - upLeft[1] - downRight[1]) / (4.0 * dx * dy);
      curlCurl(x,y)[0] = -xDyDy + yDxDy;

      // y component:
      // -\frac{\partial^2 F_y}{\partial x^2} + 
      //  \frac{\partial^2 F_x}{\partial y \partial x} \; \mathbf{y}
      Real yDxDx = (left[1] -2.0 * middle[1] + right[1]) / (dy * dy);
      Real xDxDy = (upRight[0] + downLeft[0] - upLeft[0] - downRight[0]) / (4.0 * dx * dy);
      curlCurl(x,y)[1] = -yDxDx + xDxDy;
    }

  // do the x == 0 strip, except the corners
  for (int y = 1; y < yRes - 1; y++)
  {
    int x = 0;
    VEC3F up    = oneOne(x,y + 1);
    VEC3F down  = oneOne(x,y - 1);
    VEC3F right = oneOne(x + 1,y);
    VEC3F middle = oneOne(x,y);
    VEC3F upRight = oneOne(x + 1, y + 1);
    VEC3F downRight = oneOne(x + 1, y - 1);

    Real xDyDy = (up[0] -2.0 * middle[0] + down[0]) / (dx * dx);
    Real yDxDy = (upRight[1] + down[1] - up[1] - downRight[1]) / (2.0 * dx * dy);
    curlCurl(x,y)[0] = -xDyDy + yDxDy;

    Real yDxDx = (2.0 * middle[1] - 5.0 * right[1] + 4.0 * oneOne(x+2,y)[1] - oneOne(x+3,y)[1]) / (dx * dx);
    Real xDxDy = (upRight[0] + down[0] - up[0] - downRight[0]) / (2.0 * dx * dy);
    curlCurl(x,y)[1] = -yDxDx + xDxDy;
  }
  // do the x = xRes - 1 strip, except the corners
  for (int y = 1; y < yRes - 1; y++)
  {
    int x = xRes - 1;
    VEC3F up    = oneOne(x,y + 1);
    VEC3F down  = oneOne(x,y - 1);
    VEC3F left = oneOne(x - 1,y);
    VEC3F middle = oneOne(x,y);
    VEC3F upLeft = oneOne(x - 1, y + 1);
    VEC3F downLeft = oneOne(x - 1, y - 1);

    Real xDyDy = (up[0] -2.0 * middle[0] + down[0]) / (dx * dx);
    Real yDxDy = (up[1] + downLeft[1] - upLeft[1] - down[1]) / (2.0 * dx * dy);
    curlCurl(x,y)[0] = -xDyDy + yDxDy;

    Real yDxDx = (2.0 * middle[1] - 5.0 * left[1] + 4.0 * oneOne(x-2,y)[1] - oneOne(x-3,y)[1]) / (dx * dx);
    Real xDxDy = (up[0] + downLeft[0] - upLeft[0] - down[0]) / (2.0 * dx * dy);
    curlCurl(x,y)[1] = -yDxDx + xDxDy;
  }
  // do the y == 0 strip, except the corners
  for (int x = 1; x < xRes - 1; x++)
  {
    int y = 0;
    VEC3F middle = oneOne(x,y);
    VEC3F up    = oneOne(x,y + 1);
    VEC3F left  = oneOne(x - 1,y);
    VEC3F right = oneOne(x + 1,y);
    VEC3F upRight = oneOne(x + 1, y + 1);
    VEC3F upLeft  = oneOne(x - 1, y + 1);

    Real xDyDy = (2.0 * middle[0] - 5.0 * up[0] + 4.0 * oneOne(x,y+2)[0] - oneOne(x,y+3)[0]) / (dx * dx);
    Real yDxDy = (upRight[1] + left[1] - upLeft[1] - right[1]) / (2.0 * dx * dy);
    curlCurl(x,y)[0] = -xDyDy + yDxDy;
    
    Real yDxDx = (left[1] -2.0 * middle[1] + right[1]) / (dy * dy);
    Real xDxDy = (upRight[0] + left[0] - upLeft[0] - right[0]) / (2.0 * dx * dy);
    curlCurl(x,y)[1] = -yDxDx + xDxDy;
  }
  // do the y == yRes - 1 strip, except the corners
  for (int x = 1; x < xRes - 1; x++)
  {
    int y = yRes - 1;
    VEC3F middle = oneOne(x,y);
    VEC3F left  = oneOne(x - 1,y);
    VEC3F right = oneOne(x + 1,y);
    VEC3F down = oneOne(x,y - 1);
    VEC3F downLeft  = oneOne(x - 1, y - 1);
    VEC3F downRight = oneOne(x + 1, y - 1);
    Real xDyDy = (2.0 * middle[0] - 5.0 * down[0] + 4.0 * oneOne(x,y-2)[0] - oneOne(x,y-3)[0]) / (dx * dx);
    Real yDxDy = (right[1] + downLeft[1] - left[1] - downRight[1]) / (2.0 * dx * dy);
    curlCurl(x,y)[0] = -xDyDy + yDxDy;

    Real yDxDx = (left[1] -2.0 * middle[1] + right[1]) / (dy * dy);
    Real xDxDy = (right[0] + downLeft[0] - left[0] - downRight[0]) / (2.0 * dx * dy);
    curlCurl(x,y)[1] = -yDxDx + xDxDy;
  }

  // do the corners
  {
    int x = 0;
    int y = 0;
    VEC3F middle = oneOne(x,y);
    VEC3F up    = oneOne(x,y + 1);
    VEC3F right = oneOne(x + 1,y);
    VEC3F upRight = oneOne(x + 1, y + 1);
    Real xDyDy = (2.0 * middle[0] - 5.0 * up[0] + 4.0 * oneOne(x,y+2)[0] - oneOne(x,y+3)[0]) / (dx * dx);
    Real yDxDy = (upRight[1] + middle[1] - up[1] - right[1]) / (dx * dy);
    curlCurl(x,y)[0] = -xDyDy + yDxDy;
    
    Real yDxDx = (2.0 * middle[1] - 5.0 * right[1] + 4.0 * oneOne(x+2,y)[1] - oneOne(x+3,y)[1]) / (dx * dx);
    Real xDxDy = (upRight[0] + middle[0] - up[0] - right[0]) / (dx * dy);
    curlCurl(x,y)[1] = -yDxDx + xDxDy;
  }
  {
    int x = 0;
    int y = yRes - 1;
    VEC3F middle = oneOne(x,y);
    VEC3F down = oneOne(x,y - 1);
    VEC3F right = oneOne(x + 1,y);
    VEC3F downRight = oneOne(x + 1, y - 1);
    Real xDyDy = (2.0 * middle[0] - 5.0 * down[0] + 4.0 * oneOne(x,y-2)[0] - oneOne(x,y-3)[0]) / (dx * dx);
    Real yDxDy = (right[1] + down[1] - middle[1] - downRight[1]) / (dx * dy);
    curlCurl(x,y)[0] = -xDyDy + yDxDy;
    
    Real yDxDx = (2.0 * middle[1] - 5.0 * right[1] + 4.0 * oneOne(x+2,y)[1] - oneOne(x+3,y)[1]) / (dx * dx);
    Real xDxDy = (right[0] + down[0] - middle[0] - downRight[0]) / (dx * dy);
    curlCurl(x,y)[1] = -yDxDx + xDxDy;
  }
  {
    int x = xRes - 1;
    int y = 0;
    VEC3F middle = oneOne(x,y);
    VEC3F up    = oneOne(x,y + 1);
    VEC3F left = oneOne(x - 1,y);
    VEC3F upLeft = oneOne(x - 1,y + 1);
    Real xDyDy = (2.0 * middle[0] - 5.0 * up[0] + 4.0 * oneOne(x,y+2)[0] - oneOne(x,y+3)[0]) / (dx * dx);
    Real yDxDy = (up[1] + left[1] - upLeft[1] - middle[1]) / (dx * dy);
    curlCurl(x,y)[0] = -xDyDy + yDxDy;
    
    Real yDxDx = (2.0 * middle[1] - 5.0 * left[1] + 4.0 * oneOne(x-2,y)[1] - oneOne(x-3,y)[1]) / (dx * dx);
    Real xDxDy = (up[0] + left[0] - upLeft[0] - middle[0]) / (dx * dy);
    curlCurl(x,y)[1] = -yDxDx + xDxDy;
  }
  {
    int x = xRes - 1;
    int y = yRes - 1;
    VEC3F middle = oneOne(x,y);
    VEC3F down = oneOne(x,y - 1);
    VEC3F left = oneOne(x - 1,y);
    VEC3F downLeft = oneOne(x - 1,y - 1);
    Real xDyDy = (2.0 * middle[0] - 5.0 * down[0] + 4.0 * oneOne(x,y-2)[0] - oneOne(x,y-3)[0]) / (dx * dx);
    Real yDxDy = (middle[1] + downLeft[1] - left[1] - down[1]) / (dx * dy);
    curlCurl(x,y)[0] = -xDyDy + yDxDy;

    Real yDxDx = (2.0 * middle[1] - 5.0 * left[1] + 4.0 * oneOne(x-2,y)[1] - oneOne(x-3,y)[1]) / (dx * dx);
    Real xDxDy = (middle[0] + downLeft[0] - left[0] - down[0]) / (dx * dy);
    curlCurl(x,y)[1] = -yDxDx + xDxDy;
  }
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
void verboseDoubleCurl(VECTOR3_FIELD_2D& curlCurl, VECTOR3_FIELD_2D& oneOne)
{
  int xRes = curlCurl.xRes();
  int yRes = curlCurl.yRes();
  VEC3F lengths = curlCurl.lengths();
  Real dx = lengths[0] / (xRes - 1);
  Real dy = lengths[1] / (yRes - 1);

  for (int y = 0; y < yRes; y++)
    for (int x = 0; x < xRes; x++)
    {
      VEC3F middle = oneOne(x,y);
      const VEC3F zeros;
      VEC3F up    = (y != yRes - 1) ? oneOne(x,y + 1) : zeros;
      VEC3F down  = (y != 0)        ? oneOne(x,y - 1) : zeros;
      VEC3F left  = (x != 0)        ? oneOne(x - 1,y) : zeros;
      VEC3F right = (x != xRes - 1) ? oneOne(x + 1,y) : zeros;

      VEC3F upLeft  = (x != 0 && y != yRes - 1)        ? oneOne(x - 1, y + 1) : zeros;
      VEC3F upRight = (x != xRes - 1 && y != yRes - 1) ? oneOne(x + 1, y + 1) : zeros;
      VEC3F downLeft  = (x != 0 && y != 0)        ? oneOne(x - 1, y - 1) : zeros;
      VEC3F downRight = (x != xRes - 1 && y != 0) ? oneOne(x + 1, y - 1) : zeros;

      // x component:
      // -\frac{\partial^2 F_x}{\partial y^2} + 
      //  \frac{\partial^2 F_y}{\partial x \partial y}
      Real xDyDy = (up[0] -2.0 * middle[0] + down[0]) / (dx * dx);
      Real yDxDy = (upRight[1] + downLeft[1] - upLeft[1] - downRight[1]) / (4.0 * dx * dy);
      curlCurl(x,y)[0] = -xDyDy + yDxDy;

      if (x == 0 || x == xRes - 1)
      {
        if (x == 0)
          yDxDy = (upRight[1] + down[1] - up[1] - downRight[1]) / (2.0 * dx * dy);
        if (x == xRes - 1)
          yDxDy = (up[1] + downLeft[1] - upLeft[1] - down[1]) / (2.0 * dx * dy);
        curlCurl(x,y)[0] = -xDyDy + yDxDy;
      }

      if (y == 0 || y == yRes - 1)
      {
        curlCurl(x,y)[0] = 0;
        if (y == 0)
        {
          xDyDy = (2.0 * middle[0] - 5.0 * up[0] + 4.0 * oneOne(x,y+2)[0] - oneOne(x,y+3)[0]) / (dx * dx);

          if (x != 0 && x != xRes - 1) 
            yDxDy = (upRight[1] + left[1] - upLeft[1] - right[1]) / (2.0 * dx * dy);
          if (x == 0)
            yDxDy = (upRight[1] + middle[1] - up[1] - right[1]) / (dx * dy);
          if (x == xRes - 1)
            yDxDy = (up[1] + left[1] - upLeft[1] - middle[1]) / (dx * dy);

          curlCurl(x,y)[0] = -xDyDy + yDxDy;
        }
        if (y == yRes - 1)
        {
          xDyDy = (2.0 * middle[0] - 5.0 * down[0] + 4.0 * oneOne(x,y-2)[0] - oneOne(x,y-3)[0]) / (dx * dx);
         
          if (x != 0 && x != xRes - 1) 
            yDxDy = (right[1] + downLeft[1] - left[1] - downRight[1]) / (2.0 * dx * dy);

          if (x == 0)
            yDxDy = (right[1] + down[1] - middle[1] - downRight[1]) / (dx * dy);
          
          if (x == xRes - 1)
            yDxDy = (middle[1] + downLeft[1] - left[1] - down[1]) / (dx * dy);
          curlCurl(x,y)[0] = -xDyDy + yDxDy;
        }
      }

      // y component:
      // -\frac{\partial^2 F_y}{\partial x^2} + 
      //  \frac{\partial^2 F_x}{\partial y \partial x} \; \mathbf{y}
      Real yDxDx = (left[1] -2.0 * middle[1] + right[1]) / (dy * dy);
      Real xDxDy = (upRight[0] + downLeft[0] - upLeft[0] - downRight[0]) / (4.0 * dx * dy);
      curlCurl(x,y)[1] = -yDxDx + xDxDy;
      
      if (y == 0 || y == yRes - 1)
      {
        if (y == 0)
          xDxDy = (upRight[0] + left[0] - upLeft[0] - right[0]) / (2.0 * dx * dy);
        if (y == yRes - 1)
          xDxDy = (right[0] + downLeft[0] - left[0] - downRight[0]) / (2.0 * dx * dy);
        curlCurl(x,y)[1] = -yDxDx + xDxDy;
      }

      if (x == 0 || x == xRes - 1)
      {
        curlCurl(x,y)[1] = 0;
        if (x == 0)
        {
          yDxDx = (2.0 * middle[1] - 5.0 * right[1] + 4.0 * oneOne(x+2,y)[1] - oneOne(x+3,y)[1]) / (dx * dx);
          
          if (y != 0 && y != yRes - 1)
            xDxDy = (upRight[0] + down[0] - up[0] - downRight[0]) / (2.0 * dx * dy);
          if (y == 0)
            xDxDy = (upRight[0] + middle[0] - up[0] - right[0]) / (dx * dy);
          if (y == yRes - 1)
            xDxDy = (right[0] + down[0] - middle[0] - downRight[0]) / (dx * dy);

          curlCurl(x,y)[1] = -yDxDx + xDxDy;
        }
        if (x == xRes - 1)
        {
          yDxDx = (2.0 * middle[1] - 5.0 * left[1] + 4.0 * oneOne(x-2,y)[1] - oneOne(x-3,y)[1]) / (dx * dx);

          if (y != 0 && y != yRes - 1)
            xDxDy = (up[0] + downLeft[0] - upLeft[0] - down[0]) / (2.0 * dx * dy);
          if (y == 0)
            xDxDy = (up[0] + left[0] - upLeft[0] - middle[0]) / (dx * dy);
          if (y == yRes - 1)
            xDxDy = (middle[0] + downLeft[0] - left[0] - down[0]) / (dx * dy);
          
          curlCurl(x,y)[1] = -yDxDx + xDxDy;
        }
      }
    }
}

// Lookup tabel of matrix indices
map<INDEX, int> lookup;

///////////////////////////////////////////////////////////////////////
// Add an entry to the sparse matrix
///////////////////////////////////////////////////////////////////////
void addEntry(SPARSE_MATRIX& A, INDEX& row, INDEX& col, Real entry)
{
  if (lookup.find(row) == lookup.end())
  {
    //cout << __FILE__ << " " << __FUNCTION__ << " " << __LINE__ << " : " << endl;
    //cout << " Bad row lookup! " << endl;
    //exit(0);
    return;
  }
  if (lookup.find(col) == lookup.end())
  {
    //cout << __FILE__ << " " << __FUNCTION__ << " " << __LINE__ << " : " << endl;
    //cout << " Bad column lookup! " << endl;
    //exit(0);
    return;
  }

  A(lookup[row], lookup[col]) += entry;
}

#define FIRST_ORDER 0

///////////////////////////////////////////////////////////////////////
// build the sparse matrix corresponding to the double curl
///////////////////////////////////////////////////////////////////////
void buildA(int xRes, int yRes, VEC3F lengths, SPARSE_MATRIX& A)
{
  int totalCells = xRes * yRes;
  A = SPARSE_MATRIX(2 * totalCells, 2 * totalCells);

  // build a table of the indices
  INDEX::_xRes = xRes;
  INDEX::_totalComponents = 3;
  lookup.clear();
  int i = 0;
  for (int y = 0; y < yRes; y++)
    for (int x = 0; x < xRes; x++, i += 2)
    {
      INDEX index0(x,y,0);
      lookup[index0] = i;
      INDEX index1(x,y,1);
      lookup[index1] = i + 1;
    }

  Real dx = lengths[0] / (xRes - 1);
  Real dy = lengths[1] / (yRes - 1);

  // the non-verbose way, but more convenient for converting to a stencil
  for (int y = 1; y < yRes - 1; y++)
    for (int x = 1; x < xRes - 1; x++)
    {
      INDEX middle0(x,y,0);
      INDEX middle1(x,y,1);
      INDEX up0       (x,     y + 1,0);
      INDEX down0     (x,     y - 1,0);
      INDEX upRight1  (x + 1, y + 1,1);
      INDEX downLeft1 (x - 1, y - 1,1);
      INDEX upLeft1   (x - 1, y + 1,1);
      INDEX downRight1(x + 1, y - 1,1);
      INDEX upRight0  (x + 1, y + 1,0);
      INDEX downLeft0 (x - 1, y - 1,0);
      INDEX upLeft0   (x - 1, y + 1,0);
      INDEX downRight0(x + 1, y - 1,0);
      
      INDEX left1     (x - 1, y,    1);
      INDEX right1    (x + 1, y,    1);
      /*
      VEC3F middle = oneOne(x,y);
      VEC3F up    = oneOne(x,y + 1);
      VEC3F down  = oneOne(x,y - 1);
      VEC3F left  = oneOne(x - 1,y);
      VEC3F right = oneOne(x + 1,y);

      VEC3F upLeft  = oneOne(x - 1, y + 1);
      VEC3F upRight = oneOne(x + 1, y + 1);
      VEC3F downLeft  = oneOne(x - 1, y - 1);
      VEC3F downRight = oneOne(x + 1, y - 1);
      */
      //curlCurl(x,y)[0] = xDyDy + yDxDy;
      //Real xDyDy = (-up[0] +2.0 * middle[0] - down[0]) / (dx * dx);
      addEntry(A, middle0, up0,     -1.0 / (dx * dx));
      addEntry(A, middle0, middle0,  2.0 / (dx * dx));
      addEntry(A, middle0, down0,   -1.0 / (dx * dx));
      
      //Real yDxDy = (upRight[1] + downLeft[1] - upLeft[1] - downRight[1]) / (4.0 * dx * dy);
      addEntry(A, middle0, upRight1,    1.0 / (4.0 * dx * dx));
      addEntry(A, middle0, downLeft1,   1.0 / (4.0 * dx * dx));
      addEntry(A, middle0, upLeft1,    -1.0 / (4.0 * dx * dx));
      addEntry(A, middle0, downRight1, -1.0 / (4.0 * dx * dx));

      //curlCurl(x,y)[1] = yDxDx + xDxDy;
      //Real yDxDx = (-left[1] + 2.0 * middle[1] - right[1]) / (dy * dy);
      addEntry(A, middle1, left1,   -1.0 / (dy * dy));
      addEntry(A, middle1, middle1,  2.0 / (dy * dy));
      addEntry(A, middle1, right1,  -1.0 / (dy * dy));
      
      //Real xDxDy = (upRight[0] + downLeft[0] - upLeft[0] - downRight[0]) / (4.0 * dx * dy);
      addEntry(A, middle1, upRight0,    1.0 / (4.0 * dx * dy));
      addEntry(A, middle1, downLeft0,   1.0 / (4.0 * dx * dy));
      addEntry(A, middle1, upLeft0,    -1.0 / (4.0 * dx * dy));
      addEntry(A, middle1, downRight0, -1.0 / (4.0 * dx * dy));
    }

  // do the x == 0 strip, except the corners
  for (int y = 1; y < yRes - 1; y++)
  {
    int x = 0;
    /*
    VEC3F up    = oneOne(x,y + 1);
    VEC3F down  = oneOne(x,y - 1);
    VEC3F right = oneOne(x + 1,y);
    VEC3F middle = oneOne(x,y);
    VEC3F upRight = oneOne(x + 1, y + 1);
    VEC3F downRight = oneOne(x + 1, y - 1);
    */
    INDEX middle0(x,y,0);
    INDEX middle1(x,y,1);
    INDEX up0  (x,y + 1,0);
    INDEX up1  (x,y + 1,1);
    INDEX down0(x,y - 1,0);
    INDEX down1(x,y - 1,1);
    INDEX upRight1  (x + 1, y + 1,1);
    INDEX downRight1(x + 1, y - 1,1);
    INDEX upRight0  (x + 1, y + 1,0);
    INDEX downRight0(x + 1, y - 1,0);
    
    INDEX right1          (x + 1,y,1);
    INDEX rightRight1     (x + 2,y,1);
    INDEX rightRightRight1(x + 3,y,1);

    //curlCurl(x,y)[0] = xDyDy + yDxDy;
    //Real xDyDy = (-up[0] +2.0 * middle[0] - down[0]) / (dx * dx);
    addEntry(A, middle0, up0,     -1.0 / (dx * dx));
    addEntry(A, middle0, middle0,  2.0 / (dx * dx));
    addEntry(A, middle0, down0,   -1.0 / (dx * dx));
    //Real yDxDy = (upRight[1] + down[1] - up[1] - downRight[1]) / (2.0 * dx * dy);
    addEntry(A, middle0, upRight1,    1.0 / (2.0 * dx * dx));
    addEntry(A, middle0, down1,       1.0 / (2.0 * dx * dx));
    addEntry(A, middle0, up1,        -1.0 / (2.0 * dx * dx));
    addEntry(A, middle0, downRight1, -1.0 / (2.0 * dx * dx));

    //curlCurl(x,y)[1] = yDxDx + xDxDy;
    //Real yDxDx = (-2.0 * middle[1] + 5.0 * right[1] - 4.0 * oneOne(x+2,y)[1] + oneOne(x+3,y)[1]) / (dx * dx);
#if FIRST_ORDER
    addEntry(A, middle1, middle1,          1.0 / (dx * dx));
    addEntry(A, middle1, right1,          -2.0 / (dx * dx));
    addEntry(A, middle1, rightRight1,      1.0 / (dx * dx));
#else
    addEntry(A, middle1, middle1,         -2.0 / (dx * dx));
    addEntry(A, middle1, right1,           5.0 / (dx * dx));
    addEntry(A, middle1, rightRight1,     -4.0 / (dx * dx));
    addEntry(A, middle1, rightRightRight1, 1.0 / (dx * dx));
#endif
    
    //Real xDxDy = (upRight[0] + down[0] - up[0] - downRight[0]) / (2.0 * dx * dy);
    addEntry(A, middle1, upRight0,    1.0 / (2.0 * dx * dy));
    addEntry(A, middle1, down0,       1.0 / (2.0 * dx * dy));
    addEntry(A, middle1, up0,        -1.0 / (2.0 * dx * dy));
    addEntry(A, middle1, downRight0, -1.0 / (2.0 * dx * dy));
  }
  // do the x = xRes - 1 strip, except the corners
  for (int y = 1; y < yRes - 1; y++)
  {
    int x = xRes - 1;
    /*
    VEC3F up    = oneOne(x,y + 1);
    VEC3F down  = oneOne(x,y - 1);
    VEC3F left = oneOne(x - 1,y);
    VEC3F middle = oneOne(x,y);
    VEC3F upLeft = oneOne(x - 1, y + 1);
    VEC3F downLeft = oneOne(x - 1, y - 1);
    */
    INDEX middle0(x,y,0);
    INDEX middle1(x,y,1);
    INDEX up0  (x,y + 1,0);
    INDEX down0(x,y - 1,0);
    INDEX up1  (x,y + 1,1);
    INDEX down1(x,y - 1,1);
    INDEX upLeft1  (x - 1, y + 1,1);
    INDEX downLeft1(x - 1, y - 1,1);
    INDEX upLeft0  (x - 1, y + 1,0);
    INDEX downLeft0(x - 1, y - 1,0);

    INDEX left1        (x - 1,y,1);
    INDEX leftLeft1    (x - 2,y,1);
    INDEX leftLeftLeft1(x - 3,y,1);

    //curlCurl(x,y)[0] = xDyDy + yDxDy;
    //Real xDyDy = (-up[0] +2.0 * middle[0] - down[0]) / (dx * dx);
    addEntry(A, middle0, up0,     -1.0 / (dx * dx));
    addEntry(A, middle0, middle0,  2.0 / (dx * dx));
    addEntry(A, middle0, down0,   -1.0 / (dx * dx));

    //Real yDxDy = (up[1] + downLeft[1] - upLeft[1] - down[1]) / (2.0 * dx * dy);
    addEntry(A, middle0, up1,       1.0 / (2.0 * dx * dx));
    addEntry(A, middle0, downLeft1, 1.0 / (2.0 * dx * dx));
    addEntry(A, middle0, upLeft1,  -1.0 / (2.0 * dx * dx));
    addEntry(A, middle0, down1,    -1.0 / (2.0 * dx * dx));

    //curlCurl(x,y)[1] = yDxDx + xDxDy;
    //Real yDxDx = (-2.0 * middle[1] + 5.0 * left[1] - 4.0 * oneOne(x-2,y)[1] + oneOne(x-3,y)[1]) / (dx * dx);
#if FIRST_ORDER
    addEntry(A, middle1, middle1,       1.0 / (dx * dx));
    addEntry(A, middle1, left1,        -2.0 / (dx * dx));
    addEntry(A, middle1, leftLeft1,     1.0 / (dx * dx));
#else
    addEntry(A, middle1, middle1,      -2.0 / (dx * dx));
    addEntry(A, middle1, left1,         5.0 / (dx * dx));
    addEntry(A, middle1, leftLeft1,    -4.0 / (dx * dx));
    addEntry(A, middle1, leftLeftLeft1, 1.0 / (dx * dx));
#endif

    //Real xDxDy = (up[0] + downLeft[0] - upLeft[0] - down[0]) / (2.0 * dx * dy);
    addEntry(A, middle1, up0,       1.0 / (2.0 * dx * dy));
    addEntry(A, middle1, downLeft0, 1.0 / (2.0 * dx * dy));
    addEntry(A, middle1, upLeft0,  -1.0 / (2.0 * dx * dy));
    addEntry(A, middle1, down0,    -1.0 / (2.0 * dx * dy));
  }
  // do the y == 0 strip, except the corners
  for (int x = 1; x < xRes - 1; x++)
  {
    int y = 0;
    /*
    VEC3F middle = oneOne(x,y);
    VEC3F up    = oneOne(x,y + 1);
    VEC3F left  = oneOne(x - 1,y);
    VEC3F right = oneOne(x + 1,y);
    VEC3F upRight = oneOne(x + 1, y + 1);
    VEC3F upLeft  = oneOne(x - 1, y + 1);
    */
    INDEX middle0(x,y,0);
    INDEX middle1(x,y,1);
    INDEX up0    (x,y + 1,0);
    INDEX upUp0  (x,y + 2,0);
    INDEX upUpUp0(x,y + 3,0);
    INDEX left1  (x - 1,y,1);
    INDEX right1 (x + 1,y,1);
    INDEX upRight1(x + 1, y + 1,1);
    INDEX upLeft1 (x - 1, y + 1,1);
    INDEX left0  (x - 1,y,0);
    INDEX right0 (x + 1,y,0);
    INDEX upRight0(x + 1, y + 1,0);
    INDEX upLeft0 (x - 1, y + 1,0);

    //curlCurl(x,y)[0] = xDyDy + yDxDy;
    //Real xDyDy = (-2.0 * middle[0] + 5.0 * up[0] - 4.0 * oneOne(x,y+2)[0] + oneOne(x,y+3)[0]) / (dx * dx);
#if FIRST_ORDER
    addEntry(A, middle0, middle0, 1.0 / (dx * dx));
    addEntry(A, middle0, up0,    -2.0 / (dx * dx));
    addEntry(A, middle0, upUp0,   1.0 / (dx * dx));
#else
    addEntry(A, middle0, middle0,-2.0 / (dx * dx));
    addEntry(A, middle0, up0,     5.0 / (dx * dx));
    addEntry(A, middle0, upUp0,  -4.0 / (dx * dx));
    addEntry(A, middle0, upUpUp0, 1.0 / (dx * dx));
#endif
    
    //Real yDxDy = (upRight[1] + left[1] - upLeft[1] - right[1]) / (2.0 * dx * dy);
    addEntry(A, middle0, upRight1, 1.0 / (2.0 * dx * dx));
    addEntry(A, middle0, left1,    1.0 / (2.0 * dx * dx));
    addEntry(A, middle0, upLeft1, -1.0 / (2.0 * dx * dx));
    addEntry(A, middle0, right1,  -1.0 / (2.0 * dx * dx));
    
    //curlCurl(x,y)[1] = yDxDx + xDxDy;
    //Real yDxDx = (-left[1] +2.0 * middle[1] - right[1]) / (dy * dy);
    addEntry(A, middle1, left1,   -1.0 / (dy * dy));
    addEntry(A, middle1, middle1,  2.0 / (dy * dy));
    addEntry(A, middle1, right1,  -1.0 / (dy * dy));
    
    //Real xDxDy = (upRight[0] + left[0] - upLeft[0] - right[0]) / (2.0 * dx * dy);
    addEntry(A, middle1, upRight0, 1.0 / (2.0 * dx * dy));
    addEntry(A, middle1, left0,    1.0 / (2.0 * dx * dy));
    addEntry(A, middle1, upLeft0, -1.0 / (2.0 * dx * dy));
    addEntry(A, middle1, right0,  -1.0 / (2.0 * dx * dy));
  }
  // do the y == yRes - 1 strip, except the corners
  for (int x = 1; x < xRes - 1; x++)
  {
    int y = yRes - 1;
    /*
    VEC3F middle = oneOne(x,y);
    VEC3F left  = oneOne(x - 1,y);
    VEC3F right = oneOne(x + 1,y);
    VEC3F down = oneOne(x,y - 1);
    VEC3F downLeft  = oneOne(x - 1, y - 1);
    VEC3F downRight = oneOne(x + 1, y - 1);
    */
    INDEX middle0(x,y,0);
    INDEX middle1(x,y,1);
    INDEX down0        (x,y - 1,0);
    INDEX downDown0    (x,y - 2,0);
    INDEX downDownDown0(x,y - 3,0);
    INDEX downLeft0 (x - 1, y - 1,0);
    INDEX downRight0(x + 1, y - 1,0);
    INDEX downLeft1 (x - 1, y - 1,1);
    INDEX downRight1(x + 1, y - 1,1);
    INDEX left1     (x - 1, y,    1);
    INDEX right1    (x + 1, y,    1);
    INDEX left0     (x - 1, y,    0);
    INDEX right0    (x + 1, y,    0);

    //curlCurl(x,y)[0] = xDyDy + yDxDy;
    //Real xDyDy = (-2.0 * middle[0] + 5.0 * down[0] - 4.0 * oneOne(x,y-2)[0] + oneOne(x,y-3)[0]) / (dx * dx);
#if FIRST_ORDER
    addEntry(A, middle0, middle0,       1.0 / (dx * dx));
    addEntry(A, middle0, down0,        -2.0 / (dx * dx));
    addEntry(A, middle0, downDown0,     1.0 / (dx * dx));
#else
    addEntry(A, middle0, middle0,      -2.0 / (dx * dx));
    addEntry(A, middle0, down0,         5.0 / (dx * dx));
    addEntry(A, middle0, downDown0,    -4.0 / (dx * dx));
    addEntry(A, middle0, downDownDown0, 1.0 / (dx * dx));
#endif
    
    //Real yDxDy = (right[1] + downLeft[1] - left[1] - downRight[1]) / (2.0 * dx * dy);
    addEntry(A, middle0, right1,      1.0 / (2.0 * dx * dx));
    addEntry(A, middle0, downLeft1,   1.0 / (2.0 * dx * dx));
    addEntry(A, middle0, left1,      -1.0 / (2.0 * dx * dx));
    addEntry(A, middle0, downRight1, -1.0 / (2.0 * dx * dx));

    //curlCurl(x,y)[1] = yDxDx + xDxDy;
    //Real yDxDx = (-left[1] +2.0 * middle[1] - right[1]) / (dy * dy);
    addEntry(A, middle1, left1,   -1.0 / (dy * dy));
    addEntry(A, middle1, middle1,  2.0 / (dy * dy));
    addEntry(A, middle1, right1,  -1.0 / (dy * dy));

    //Real xDxDy = (right[0] + downLeft[0] - left[0] - downRight[0]) / (2.0 * dx * dy);
    addEntry(A, middle1, right0,      1.0 / (2.0 * dx * dy));
    addEntry(A, middle1, downLeft0,   1.0 / (2.0 * dx * dy));
    addEntry(A, middle1, left0,      -1.0 / (2.0 * dx * dy));
    addEntry(A, middle1, downRight0, -1.0 / (2.0 * dx * dy));
  }

  // do the corners
  {
    int x = 0;
    int y = 0;
    //VEC3F middle = oneOne(x,y);
    //VEC3F up    = oneOne(x,y + 1);
    //VEC3F right = oneOne(x + 1,y);
    //VEC3F upRight = oneOne(x + 1, y + 1);
    INDEX middle0(x,y,0);
    INDEX middle1(x,y,1);
    INDEX up1    (x,y + 1,1);
    INDEX right0 (x + 1,y,0);
    INDEX upRight0(x + 1, y + 1,0);
    INDEX upRight1(x + 1, y + 1,1);
    
    INDEX up0    (x,y + 1,0);
    INDEX upUp0  (x,y + 2,0);
    INDEX upUpUp0(x,y + 3,0);
    
    INDEX right1          (x + 1,y,1);
    INDEX rightRight1     (x + 2,y,1);
    INDEX rightRightRight1(x + 3,y,1);

    //curlCurl(x,y)[0] = xDyDy + yDxDy;
    //Real xDyDy = (-2.0 * middle[0] + 5.0 * up[0] - 4.0 * oneOne(x,y+2)[0] + oneOne(x,y+3)[0]) / (dx * dx);
#if FIRST_ORDER
    addEntry(A, middle0, middle0, 1.0 / (dx * dx));
    addEntry(A, middle0, up0,    -2.0 / (dx * dx));
    addEntry(A, middle0, upUp0,   1.0 / (dx * dx));
#else
    addEntry(A, middle0, middle0,-2.0 / (dx * dx));
    addEntry(A, middle0, up0,     5.0 / (dx * dx));
    addEntry(A, middle0, upUp0,  -4.0 / (dx * dx));
    addEntry(A, middle0, upUpUp0, 1.0 / (dx * dx));
#endif

    //Real yDxDy = (upRight[1] + middle[1] - up[1] - right[1]) / (dx * dy);
    addEntry(A, middle0, upRight1, 1.0 / (dx * dx));
    addEntry(A, middle0, middle1,  1.0 / (dx * dx));
    addEntry(A, middle0, up1,     -1.0 / (dx * dx));
    addEntry(A, middle0, right1,  -1.0 / (dx * dx));
    
    //curlCurl(x,y)[1] = yDxDx + xDxDy;
    //Real yDxDx = (-2.0 * middle[1] + 5.0 * right[1] - 4.0 * oneOne(x+2,y)[1] + oneOne(x+3,y)[1]) / (dx * dx);
#if FIRST_ORDER
    addEntry(A, middle1, middle1,          1.0 / (dx * dx));
    addEntry(A, middle1, right1,          -2.0 / (dx * dx));
    addEntry(A, middle1, rightRight1,      1.0 / (dx * dx));
#else
    addEntry(A, middle1, middle1,         -2.0 / (dx * dx));
    addEntry(A, middle1, right1,           5.0 / (dx * dx));
    addEntry(A, middle1, rightRight1,     -4.0 / (dx * dx));
    addEntry(A, middle1, rightRightRight1, 1.0 / (dx * dx));
#endif

    //Real xDxDy = (upRight[0] + middle[0] - up[0] - right[0]) / (dx * dy);
    addEntry(A, middle1, upRight0, 1.0 / (dx * dy));
    addEntry(A, middle1, middle0,  1.0 / (dx * dy));
    addEntry(A, middle1, up0,     -1.0 / (dx * dy));
    addEntry(A, middle1, right0,  -1.0 / (dx * dy));
  }
  {
    int x = 0;
    int y = yRes - 1;
    INDEX middle0(x,y,0);
    INDEX middle1(x,y,1);
    INDEX down1        (x,y - 1,1);
    INDEX down0        (x,y - 1,0);
    INDEX downDown0    (x,y - 2,0);
    INDEX downDownDown0(x,y - 3,0);
    INDEX downRight0(x + 1, y - 1,0);
    INDEX downRight1(x + 1, y - 1,1);
    INDEX right0          (x + 1,y,0);
    INDEX right1          (x + 1,y,1);
    INDEX rightRight1     (x + 2,y,1);
    INDEX rightRightRight1(x + 3,y,1);
    //VEC3F middle = oneOne(x,y);
    //VEC3F down = oneOne(x,y - 1);
    //VEC3F right = oneOne(x + 1,y);
    //VEC3F downRight = oneOne(x + 1, y - 1);
    
    //curlCurl(x,y)[0] = xDyDy + yDxDy;
    //Real xDyDy = (-2.0 * middle[0] + 5.0 * down[0] - 4.0 * oneOne(x,y-2)[0] + oneOne(x,y-3)[0]) / (dx * dx);
#if FIRST_ORDER
    addEntry(A, middle0, middle0,       1.0 / (dx * dx));
    addEntry(A, middle0, down0,        -2.0 / (dx * dx));
    addEntry(A, middle0, downDown0,     1.0 / (dx * dx));
#else
    addEntry(A, middle0, middle0,      -2.0 / (dx * dx));
    addEntry(A, middle0, down0,         5.0 / (dx * dx));
    addEntry(A, middle0, downDown0,    -4.0 / (dx * dx));
    addEntry(A, middle0, downDownDown0, 1.0 / (dx * dx));
#endif

    //Real yDxDy = (right[1] + down[1] - middle[1] - downRight[1]) / (dx * dy);
    addEntry(A, middle0, right1,      1.0 / (dx * dx));
    addEntry(A, middle0, down1,       1.0 / (dx * dx));
    addEntry(A, middle0, middle1,    -1.0 / (dx * dx));
    addEntry(A, middle0, downRight1, -1.0 / (dx * dx));
    
    //curlCurl(x,y)[1] = yDxDx + xDxDy;
    //Real yDxDx = (-2.0 * middle[1] + 5.0 * right[1] - 4.0 * oneOne(x+2,y)[1] + oneOne(x+3,y)[1]) / (dx * dx);
#if FIRST_ORDER
    addEntry(A, middle1, middle1,          1.0 / (dx * dx));
    addEntry(A, middle1, right1,          -2.0 / (dx * dx));
    addEntry(A, middle1, rightRight1,      1.0 / (dx * dx));
#else
    addEntry(A, middle1, middle1,         -2.0 / (dx * dx));
    addEntry(A, middle1, right1,           5.0 / (dx * dx));
    addEntry(A, middle1, rightRight1,     -4.0 / (dx * dx));
    addEntry(A, middle1, rightRightRight1, 1.0 / (dx * dx));
#endif

    //Real xDxDy = (right[0] + down[0] - middle[0] - downRight[0]) / (dx * dy);
    addEntry(A, middle1, right0,      1.0 / (dx * dx));
    addEntry(A, middle1, down0,       1.0 / (dx * dx));
    addEntry(A, middle1, middle0,    -1.0 / (dx * dx));
    addEntry(A, middle1, downRight0, -1.0 / (dx * dx));
  }
  {
    int x = xRes - 1;
    int y = 0;
    //VEC3F middle = oneOne(x,y);
    //VEC3F up    = oneOne(x,y + 1);
    //VEC3F left = oneOne(x - 1,y);
    //VEC3F upLeft = oneOne(x - 1,y + 1);
    INDEX middle0(x,y,0);
    INDEX middle1(x,y,1);
    
    INDEX up1  (x,y + 1,1);
    INDEX left0(x - 1,y,0);

    INDEX up0    (x,y + 1,0);
    INDEX upUp0  (x,y + 2,0);
    INDEX upUpUp0(x,y + 3,0);

    INDEX left1        (x - 1,y,1);
    INDEX leftLeft1    (x - 2,y,1);
    INDEX leftLeftLeft1(x - 3,y,1);

    INDEX upLeft0(x - 1, y + 1,0);
    INDEX upLeft1(x - 1, y + 1,1);

    //curlCurl(x,y)[0] = xDyDy + yDxDy;
    //Real xDyDy = (-2.0 * middle[0] + 5.0 * up[0] - 4.0 * oneOne(x,y+2)[0] + oneOne(x,y+3)[0]) / (dx * dx);
#if FIRST_ORDER
    addEntry(A, middle0, middle0, 1.0 / (dx * dx));
    addEntry(A, middle0, up0,    -2.0 / (dx * dx));
    addEntry(A, middle0, upUp0,   1.0 / (dx * dx));
#else
    addEntry(A, middle0, middle0,-2.0 / (dx * dx));
    addEntry(A, middle0, up0,     5.0 / (dx * dx));
    addEntry(A, middle0, upUp0,  -4.0 / (dx * dx));
    addEntry(A, middle0, upUpUp0, 1.0 / (dx * dx));
#endif

    //Real yDxDy = (up[1] + left[1] - upLeft[1] - middle[1]) / (dx * dy);
    addEntry(A, middle0, up1,      1.0 / (dx * dx));
    addEntry(A, middle0, left1,    1.0 / (dx * dx));
    addEntry(A, middle0, upLeft1, -1.0 / (dx * dx));
    addEntry(A, middle0, middle1, -1.0 / (dx * dx));
    
    //curlCurl(x,y)[1] = yDxDx + xDxDy;
    //Real yDxDx = (-2.0 * middle[1] + 5.0 * left[1] - 4.0 * oneOne(x-2,y)[1] + oneOne(x-3,y)[1]) / (dx * dx);
#if FIRST_ORDER
    addEntry(A, middle1, middle1,       1.0 / (dx * dx));
    addEntry(A, middle1, left1,        -2.0 / (dx * dx));
    addEntry(A, middle1, leftLeft1,     1.0 / (dx * dx));
#else
    addEntry(A, middle1, middle1,      -2.0 / (dx * dx));
    addEntry(A, middle1, left1,         5.0 / (dx * dx));
    addEntry(A, middle1, leftLeft1,    -4.0 / (dx * dx));
    addEntry(A, middle1, leftLeftLeft1, 1.0 / (dx * dx));
#endif
    
    //Real xDxDy = (up[0] + left[0] - upLeft[0] - middle[0]) / (dx * dy);
    addEntry(A, middle1, up0,      1.0 / (dx * dx));
    addEntry(A, middle1, left0,    1.0 / (dx * dx));
    addEntry(A, middle1, upLeft0, -1.0 / (dx * dx));
    addEntry(A, middle1, middle0, -1.0 / (dx * dx));
  }
  {
    int x = xRes - 1;
    int y = yRes - 1;
    //VEC3F middle = oneOne(x,y);
    //VEC3F down = oneOne(x,y - 1);
    //VEC3F left = oneOne(x - 1,y);
    //VEC3F downLeft = oneOne(x - 1,y - 1);
    INDEX middle0(x,y,0);
    INDEX middle1(x,y,1);

    INDEX down0        (x,y - 1,0);
    INDEX downDown0    (x,y - 2,0);
    INDEX downDownDown0(x,y - 3,0);
    
    INDEX down1        (x,y - 1,1);
    INDEX downLeft0    (x - 1, y - 1,0);
    INDEX downLeft1    (x - 1, y - 1,1);

    INDEX left0        (x - 1,y,0);
    INDEX left1        (x - 1,y,1);
    INDEX leftLeft1    (x - 2,y,1);
    INDEX leftLeftLeft1(x - 3,y,1);

    //curlCurl(x,y)[0] = xDyDy + yDxDy;
    //Real xDyDy = (-2.0 * middle[0] + 5.0 * down[0] - 4.0 * oneOne(x,y-2)[0] + oneOne(x,y-3)[0]) / (dx * dx);
#if FIRST_ORDER
    addEntry(A, middle0, middle0,       1.0 / (dx * dx));
    addEntry(A, middle0, down0,        -2.0 / (dx * dx));
    addEntry(A, middle0, downDown0,     1.0 / (dx * dx));
#else
    addEntry(A, middle0, middle0,      -2.0 / (dx * dx));
    addEntry(A, middle0, down0,         5.0 / (dx * dx));
    addEntry(A, middle0, downDown0,    -4.0 / (dx * dx));
    addEntry(A, middle0, downDownDown0, 1.0 / (dx * dx));
#endif

    //Real yDxDy = (middle[1] + downLeft[1] - left[1] - down[1]) / (dx * dy);
    addEntry(A, middle0, middle1,    1.0 / (dx * dx));
    addEntry(A, middle0, downLeft1,  1.0 / (dx * dx));
    addEntry(A, middle0, left1,     -1.0 / (dx * dx));
    addEntry(A, middle0, down1,     -1.0 / (dx * dx));

    //curlCurl(x,y)[1] = yDxDx + xDxDy;
    //Real yDxDx = (-2.0 * middle[1] + 5.0 * left[1] - 4.0 * oneOne(x-2,y)[1] + oneOne(x-3,y)[1]) / (dx * dx);
#if FIRST_ORDER
    addEntry(A, middle1, middle1,       1.0 / (dx * dx));
    addEntry(A, middle1, left1,        -2.0 / (dx * dx));
    addEntry(A, middle1, leftLeft1,     1.0 / (dx * dx));
#else
    addEntry(A, middle1, middle1,      -2.0 / (dx * dx));
    addEntry(A, middle1, left1,         5.0 / (dx * dx));
    addEntry(A, middle1, leftLeft1,    -4.0 / (dx * dx));
    addEntry(A, middle1, leftLeftLeft1, 1.0 / (dx * dx));
#endif

    //Real xDxDy = (middle[0] + downLeft[0] - left[0] - down[0]) / (dx * dy);
    addEntry(A, middle1, middle0,    1.0 / (dx * dx));
    addEntry(A, middle1, downLeft0,  1.0 / (dx * dx));
    addEntry(A, middle1, left0,     -1.0 / (dx * dx));
    addEntry(A, middle1, down0,     -1.0 / (dx * dx));
  }
}

///////////////////////////////////////////////////////////////////////
// build the sparse matrix corresponding to the double curl
///////////////////////////////////////////////////////////////////////
void buildTruncatedA(int xRes, int yRes, VEC3F lengths, SPARSE_MATRIX& A)
{
  int totalCells = xRes * yRes;
  A = SPARSE_MATRIX(2 * totalCells, 2 * totalCells);

  // build a table of the indices
  INDEX::_xRes = xRes;
  INDEX::_totalComponents = 3;
  lookup.clear();
  int i = 0;
  for (int y = 0; y < yRes; y++)
    for (int x = 0; x < xRes; x++, i += 2)
    {
      INDEX index0(x,y,0);
      lookup[index0] = i;
      INDEX index1(x,y,1);
      lookup[index1] = i + 1;
    }

  Real dx = lengths[0] / (xRes - 1);
  Real dy = lengths[1] / (yRes - 1);

  // the non-verbose way, but more convenient for converting to a stencil
  for (int y = 0; y < yRes; y++)
    for (int x = 0; x < xRes; x++)
    {
      INDEX middle0(x,y,0);
      INDEX middle1(x,y,1);
      INDEX up0       (x,     y + 1,0);
      INDEX down0     (x,     y - 1,0);
      INDEX upRight1  (x + 1, y + 1,1);
      INDEX downLeft1 (x - 1, y - 1,1);
      INDEX upLeft1   (x - 1, y + 1,1);
      INDEX downRight1(x + 1, y - 1,1);
      INDEX upRight0  (x + 1, y + 1,0);
      INDEX downLeft0 (x - 1, y - 1,0);
      INDEX upLeft0   (x - 1, y + 1,0);
      INDEX downRight0(x + 1, y - 1,0);
      
      INDEX left1     (x - 1, y,    1);
      INDEX right1    (x + 1, y,    1);
      
      //curlCurl(x,y)[0] = xDyDy + yDxDy;
      //Real xDyDy = (-up[0] +2.0 * middle[0] - down[0]) / (dx * dx);
      addEntry(A, middle0, up0,     -1.0 / (dx * dx));
      addEntry(A, middle0, middle0,  2.0 / (dx * dx));
      addEntry(A, middle0, down0,   -1.0 / (dx * dx));
      
      //Real yDxDy = (upRight[1] + downLeft[1] - upLeft[1] - downRight[1]) / (4.0 * dx * dy);
      addEntry(A, middle0, upRight1,    1.0 / (4.0 * dx * dx));
      addEntry(A, middle0, downLeft1,   1.0 / (4.0 * dx * dx));
      addEntry(A, middle0, upLeft1,    -1.0 / (4.0 * dx * dx));
      addEntry(A, middle0, downRight1, -1.0 / (4.0 * dx * dx));

      //curlCurl(x,y)[1] = yDxDx + xDxDy;
      //Real yDxDx = (-left[1] + 2.0 * middle[1] - right[1]) / (dy * dy);
      addEntry(A, middle1, left1,   -1.0 / (dy * dy));
      addEntry(A, middle1, middle1,  2.0 / (dy * dy));
      addEntry(A, middle1, right1,  -1.0 / (dy * dy));
      
      //Real xDxDy = (upRight[0] + downLeft[0] - upLeft[0] - downRight[0]) / (4.0 * dx * dy);
      addEntry(A, middle1, upRight0,    1.0 / (4.0 * dx * dy));
      addEntry(A, middle1, downLeft0,   1.0 / (4.0 * dx * dy));
      addEntry(A, middle1, upLeft0,    -1.0 / (4.0 * dx * dy));
      addEntry(A, middle1, downRight0, -1.0 / (4.0 * dx * dy));
    }
}

///////////////////////////////////////////////////////////////////////
// build and test the double curl matrix
///////////////////////////////////////////////////////////////////////
void doubleCurl()
{
  int xRes = 100;
  int yRes = 100;
  //int xRes = 20;
  //int yRes = 20;
  VEC3F center(M_PI / 2.0, M_PI / 2.0, 0);
  VEC3F lengths(M_PI, M_PI, 0);
  Real dx = lengths[0] / (xRes - 1);
  Real dy = lengths[1] / (yRes - 1);
  cout << " dx: " << dx << " dy: " << dy << endl;

  // compute the double curl of a field directly
  VECTOR3_FIELD_2D oneOne(xRes, yRes, center, lengths);
  MERSENNETWISTER twister(123456);
  oneOne.eigenfunction(1,1);
  /*
  for (int y = 0; y < yRes; y++)
    for (int x = 0; x < xRes; x++)
    {
      oneOne(x,y)[0] *= 5;
      oneOne(x,y)[1] *= 0.5;
    }
  */
  //oneOne.stompBorder();
  VECTOR3_FIELD_2D curlCurl(xRes, yRes, center, lengths);

  // the verbose way
  //verboseDoubleCurl(curlCurl, oneOne);

  //VECTOR3_FIELD_2D verboseCurl = curlCurl;

  // the stencil-friendly way  
  stencilFriendlyDoubleCurl(curlCurl, oneOne);

  //VECTOR3_FIELD_2D diff3 = verboseCurl;
  //diff3 -= curlCurl;

  //cout << " diff: " << diff3.sumSq() << endl;

  oneOne.writeLIC("cc_before.jpg");
  curlCurl.writeLIC("cc_after.jpg");

  //FIELDVIEW2D(oneOne.scalarField(0));
  //FIELDVIEW2D(curlCurl.scalarField(0));
  //FIELDVIEW2D(oneOne.scalarField(1));
  //FIELDVIEW2D(curlCurl.scalarField(1));

  //FIELD_2D ground = oneOne.scalarField(1);
  //FIELD_2D test = curlCurl.scalarField(1);
  FIELD_2D ground = oneOne.scalarField(0);
  FIELD_2D test = curlCurl.scalarField(0);
  test /= ground;
  //FIELDVIEW2D(test);
  
  ground = oneOne.scalarField(1);
  test = curlCurl.scalarField(1);
  //FIELDVIEW2D(ground);
  test /= ground;
  //FIELDVIEW2D(ground);
  //FIELDVIEW2D(test);

  // build the sparse matrix
  SPARSE_MATRIX A;
  buildA(xRes, yRes, lengths, A);
  //buildTruncatedA(xRes, yRes, lengths, A);
  
  // should do a multiply here to verify that the matrix was build correctly
  VECTOR oneOneFlattened = oneOne.flattenXY();

  VECTOR product = A * oneOneFlattened;

  VECTOR3_FIELD_2D productField(oneOne);
  productField.unflattenXY(product);

  //FIELDVIEW2D(productField.scalarField(0));
  //FIELDVIEW2D(productField.scalarField(1));
  //FIELDVIEW2D(productField.scalarField(0) - curlCurl.scalarField(0));
  //FIELDVIEW2D(productField.scalarField(1) - curlCurl.scalarField(1));
  //exit(0);

  //FIELDVIEW2D(oneOne.scalarField(0));
  //FIELDVIEW2D(curlCurl.scalarField(0));
  //FIELDVIEW2D(curlCurlProduct.scalarField(0));
  //FIELDVIEW2D(diffX);
  //FIELDVIEW2D(diffY);
  A.matlabEigs(6, "./data/eigenvectors.matrix", "./data/eigenvalues.vector");

  MATRIX E("./data/eigenvectors.matrix");

  for (int c = 0; c < 2; c++)
  {
    VECTOR eigenvector = E.getColumn(c);
    VECTOR3_FIELD_2D finalEigenvector(oneOne);
    finalEigenvector.unflattenXY(eigenvector);
    FIELDVIEW2D(finalEigenvector.scalarField(0));
    FIELDVIEW2D(finalEigenvector.scalarField(1));
  }

  exit(0);
  /*
  int i = 0;

  VEC3F mean;
  VEC3F maxs;
  //for (int y = 0; y < yRes; y++)
  //  for (int x = 0; x < xRes; x++, i += 2)
  for (int y = 1; y < yRes - 1; y++)
    for (int x = 1; x < xRes - 1; x++, i += 2)
    {
      oneOne(x,y)[0] = eigenvector[i];
      oneOne(x,y)[1] = eigenvector[i + 1];

      if (fabs(eigenvector[0]) > maxs[0])
        maxs[0] = fabs(eigenvector[0]);
      if (fabs(eigenvector[1]) > maxs[1])
        maxs[1] = fabs(eigenvector[1]);
      oneOne(x,y)[2] = 0;
      mean[0] += eigenvector[i];
      mean[1] += eigenvector[i + 1];
    }
  oneOne.writeLIC("eigenvector.jpg");
  */

  /*  
  for (int y = 1; y < yRes - 1; y++)
    for (int x = 1; x < xRes - 1; x++)
    {
      oneOne(x,y)[0] *= 1.0 / maxs[0];
      oneOne(x,y)[1] *= 1.0 / maxs[1];
    }
  */

  /*
  mean *= 2.0 / i;

  oneOne.stompBorder();
  for (int y = 0; y < yRes; y++)
    for (int x = 0; x < xRes; x++)
    {
      if (x == 0 || y == 0 || y == yRes - 1 || x == xRes - 1)
      {
        oneOne(x,y)[0] = mean[0];
        oneOne(x,y)[1] = mean[1];
      }
    }

  FIELDVIEW2D(oneOne.scalarField(0));
  FIELDVIEW2D(oneOne.scalarField(1));
  */

  exit(0);
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
  //doubleCurl();

  //tests();
  //test2();

  //int xRes = 65;
  //int yRes = 65;
  int xRes = 64;
  int yRes = 64;
  VEC3F center(M_PI / 2.0, M_PI / 2.0, 0);
  VEC3F lengths(M_PI, M_PI, 0);
  vectorField2D = VECTOR3_FIELD_2D(xRes, yRes, center, lengths);
  vectorField2D.eigenfunction(1,1);
  vectorField2D.writeLIC("1_1.jpg");

  bool success = velocityU.read("./data/velocityU.matrix");

  /*
  if (success)
  {
    success = vorticityU.read("./data/vorticityU.matrix");
    assert(success);
    success = C.read("./data/C.tensor");
    assert(success);
  }
  else
  */

  /*
  {
    buildIJPairsPerfectSquare(basisRank);
    buildVelocityBasisPerfectSquare(basisRank);
    buildVorticityBasisPerfectSquare(basisRank);
    //buildPerfectSquareC();
    buildC();
  }
  */
  {
    buildIJPairs(basisRank);
    buildVelocityBasis(basisRank);
    buildVorticityBasis(basisRank);
    //C.read("./data/C.tensor");
    buildC();
    //C.write("./data/C.tensor");
  }

  //VECTOR::printVertical = true;
  //cout << " velocity1 = " << velocityU.getColumn(0) << ";" << endl;
  //cout << " velocity7 = " << velocityU.getColumn(7) << ";" << endl;

  // dump out the first C
  //cout << "C = " << C.slab(0) << endl; 

  // initialize vorticity vectors
  w = VECTOR(basisRank);
  wDot = VECTOR(basisRank);

  // insert a vertical impulse right in the middle
  VECTOR3_FIELD_2D impulse(xRes, yRes, center, lengths);
  //impulse(xRes / 2, yRes / 2)[1] = 1000;
  impulse(xRes / 2, yRes / 2)[0] = 1000;

  VECTOR force = velocityU ^ impulse.flatten();
  w += force;

  eyeCenter = center;
  zoom = lengths[0] * 1.1;
  distanceField = vectorField2D.LIC();

  seedParticles();

  glutInit(&argc, argv);

  glvuWindow();
  return 1;
}
