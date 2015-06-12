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
//VEC3F center(0.5, 0.5, 0.5);
VEC3F lengths(1,1,1);
FIELD_3D field(10,10,10, center, lengths);

// these determine the size of the basis that will be used
//int res = 20;
//int dim = 2;
int res = 100;
int dim = 3;

//int res = 100;
//VECTOR3_FIELD_3D velocityField(50,50,50, center, lengths);
//VECTOR3_FIELD_3D velocityField(100,100,100, center, lengths);
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

// use for reverse lookups later
int slabSize;
int xRes;

VECTOR w;
VECTOR wDot;
Real dt = 0.001;

///////////////////////////////////////////////////////////////////////
// step the system forward in time
///////////////////////////////////////////////////////////////////////
void stepEigenfunctions()
{
  TIMER functionTimer(__FUNCTION__);
  int basisRank = ixyz.size();
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

  for (int k = 0; k < basisRank; k++)
  {
    Real lambda = -(eigenvalues[k]);

    //const Real viscosity = 0.5;
    //const Real viscosity = 1.0;
    //const Real viscosity = 5;
    const Real viscosity = 10;

    // diffuse
    w[k] *= exp(lambda * dt * viscosity);
  }

  {
  TIMER reconstructionTimer("Velocity Reconstruction");
  VECTOR final = velocityU * w;
  velocityField.unflatten(final);
  }
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

  // always push in the zero modes
  int columnsMade = 0;
  int zeros[3][4] = {{0,1,1,0}, {0,1,0,1}, {1,0,1,1}};
  for (int x = 0; x < 3; x++)
  {
    vector<int> zeroMode;
    for (int y = 0; y < 4; y++)
      zeroMode.push_back(zeros[x][y]);

    ixyz.push_back(zeroMode);
    Real eigenvalue = zeros[x][1] * zeros[x][1] + zeros[x][2] * zeros[x][2] + zeros[x][3] * zeros[x][3];
    eigenvalues.push_back(eigenvalue);

    int reverse = zeros[x][3] * slabSize + zeros[x][2] * xRes + zeros[x][1] * 3 + zeros[x][0];
    ixyzReverse[reverse] = ixyz.size() - 1;
  }

  // add the cube of modes
  for (int z = 0; z < dim; z++)
    for (int y = 0; y < dim; y++)
      for (int x = 0; x < dim; x++)
        for (int i = 0; i < 3; i++)
        {
          vector<int> mode;
          mode.push_back(i);
          mode.push_back(x + 1);
          mode.push_back(y + 1);
          mode.push_back(z + 1);
          ixyz.push_back(mode);
    
          Real eigenvalue = (x + 1) * (x + 1) + (y + 1) * (y + 1) + (z + 1) * (z + 1);
          eigenvalues.push_back(eigenvalue);

          int reverse = mode[3] * slabSize + mode[2] * xRes + mode[1] * 3 + mode[0];
          ixyzReverse[reverse] = ixyz.size() - 1;
        }
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
  for (int x = 0; x < ixyz.size(); x++)
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
  vector<VECTOR> columns(ixyz.size());
#pragma omp parallel
#pragma omp for  schedule(dynamic)
  for (int x = 0; x < ixyz.size(); x++)
  {
    int i  = ixyz[x][0];
    int k1 = ixyz[x][1];
    int k2 = ixyz[x][2];
    int k3 = ixyz[x][3];
    cout << " Making eigenfunction (" << i << ", " << k1 << "," << k2 << "," << k3 << ")" << endl;
    velocityField.eigenfunctionUnscaled(i,k1,k2,k3);
    VECTOR column = velocityField.flattened();
    columns[x] = column;
  }
  cout << " Built " << ixyz.size() << " eigenfunctions " << endl;

  velocityU = MATRIX(columns);
  velocityU.write(filename.c_str());
}

///////////////////////////////////////////////////////////////////////
// build the structure coefficient tensor
///////////////////////////////////////////////////////////////////////
void buildC(const string& filename)
{
  TIMER functionTimer(__FUNCTION__);
  int basisRank = ixyz.size();

  int xRes = velocityField.xRes();
  int yRes = velocityField.yRes();
  int zRes = velocityField.zRes();

  C = TENSOR3(basisRank, basisRank, basisRank);
  for (int d1 = 0; d1 < basisRank; d1++)
  {
    vector<int> a123 = ixyz[d1];
#pragma omp parallel
#pragma omp for  schedule(dynamic)
    for (int d2 = 0; d2 < basisRank; d2++) 
    {
      vector<int> b123 = ixyz[d2];

      int a = reverseLookup(a123);
      int b = reverseLookup(b123);

      for (int d3 = 0; d3 < basisRank; d3++)
      {
        vector<int> k123 = ixyz[d3];
        int k = reverseLookup(k123);
        
        //Real coef = 0;
        Real coef = VECTOR3_FIELD_3D::structureCoefficient(a123, b123, k123, xRes, yRes, zRes);

        //newC(k1, k2) = coef;
        C(b, a, k) = coef;
        //cout << " k1: " << k1 << " k2: " << k2 << " a1: " << a1 << " a2: " << a2 << " b1: " << b1 << " b2: " << b2 << " coef: " << coef << endl;
      }
    }
    cout << d1 << endl;
  }
  C.write(filename);

  //cout << " newC = " << newC << endl;
  //cout << " numericalC= " << C.slab(0) << endl; 
  //exit(0);
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
      //glTranslatef(0.5, 0.5, 0.5);
      //fluid->density().draw();
      //fluid->density().drawBoundingBox();
      //field.drawBoundingBox();
      velocityField.drawBoundingBox();
    glPopMatrix();

    glColor4f(3,0,0,1);
    /*
    glPointSize(3);
    glBegin(GL_POINTS);
      for (int x = 0; x < particles.size(); x++)
      {
        VEC3F velocity = velocityField(particles[x]);
        glNormal3f(velocity[0], velocity[1], velocity[2]);
        glVertex3f(particles[x][0], particles[x][1], particles[x][2]);
      }
    glEnd();
    */

    for (int x = 0; x < ribbons.size(); x++)
    {
      VEC3F velocity = velocityField(particles[x]);
      //glNormal3f(velocity[0], velocity[1], velocity[2]);
      glLineWidth(2);
      glBegin(GL_LINE_STRIP);
        list<VEC3F>::iterator iter;
        list<VEC3F>::iterator iterVelocity = velocityRibbons[x].begin();
        for (iter = ribbons[x].begin(); iter != ribbons[x].end(); iter++)
        {
          glNormal3f((*iterVelocity)[0], (*iterVelocity)[1], (*iterVelocity)[2]);
          glVertex3f((*iter)[0], (*iter)[1], (*iter)[2]);
        }
      glEnd();
    }

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
    case 't':
      TIMER::printTimings();
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
 
 /* 
  GLfloat lightOnePosition[] = {-10.0, -4.0, -10.0, 1.0};
  GLfloat lightOneColor[] = {1.0, 1.0, 1.0, 1.0};
  glLightfv(GL_LIGHT1, GL_POSITION, lightOnePosition);
  glLightfv(GL_LIGHT1, GL_DIFFUSE, lightOneColor);
  
  GLfloat lightTwoPosition[] = {10.0, 0, 0, 1.0};
  GLfloat lightTwoColor[] = {1.0, 1.0, 1.0, 1.0};
  glLightfv(GL_LIGHT1, GL_POSITION, lightTwoPosition);
  glLightfv(GL_LIGHT1, GL_DIFFUSE, lightTwoColor);
  
  GLfloat lightThreePosition[] = {0, 10.0, 0, 1.0};
  GLfloat lightThreeColor[] = {1.0, 1.0, 1.0, 1.0};
  glLightfv(GL_LIGHT1, GL_POSITION, lightThreePosition);
  glLightfv(GL_LIGHT1, GL_DIFFUSE, lightThreeColor);
  */
  GLfloat lightColor[] = {1.0, 1.0, 1.0, 1.0};
  for (int x = 0; x < 6; x++)
  {
    GLfloat lightPosition[] = {0,0,0,1.0};
    lightPosition[x] = 10;

    if (x > 2)
      lightPosition[x] *= -1;
    glLightfv(GL_LIGHT1, GL_POSITION, lightPosition);
    glLightfv(GL_LIGHT1, GL_DIFFUSE, lightColor);
  }

  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); 
  glEnable(GL_LIGHTING);
  glEnable(GL_LIGHT0);
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

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
  buildTableIXYZ(dim);

  char buffer[256];
  bool success = true;

  sprintf(buffer, "./data/C.res.%i.dim.%i.tensor", res, dim);
  success = C.read(buffer);
  if (!success)
    buildC(buffer);
  else
    cout << buffer << " found! " << endl;

  // this is not actually needed at runtime
  /*
  sprintf(buffer, "./data/vorticity.res.%i.dim.%i.matrix", res, dim);
  success = vorticityU.read(buffer);
  if (!success)
    buildVorticityBasis(buffer);
  else
    cout << buffer << " found! " << endl;
    */

  sprintf(buffer, "./data/velocity.res.%i.dim.%i.matrix", res, dim);
  success = velocityU.read(buffer);
  if (!success)
    buildVelocityBasis(buffer);
  else
    cout << buffer << " found! " << endl;

  TIMER::printTimings();

  //exit(0);
  w = VECTOR(ixyz.size());
  wDot = VECTOR(ixyz.size());

  VECTOR3_FIELD_3D impulse(velocityField);
  impulse = 0;
  //impulse(impulse.xRes() / 2, impulse.yRes() / 2, impulse.zRes() / 2)[1] = 1;
  int half = impulse.xRes() / 2;
  impulse(half, half, half)[1] = 1;
  //impulse(half, half, half)[1] = 1;
  VECTOR force = velocityU ^ impulse.flattened();
  w += force;

  //velocityField.eigenfunction(2,1,1,1);
  //velocityField.eigenfunction(0,1,1,0);
  //velocityField.eigenfunction(0,2,3,4);
  //velocityField.eigenfunctionUnscaled(2,1,1,1);
  //velocityField.eigenfunctionUnscaled(0,2,2,2);

  //FIELDVIEW3D(velocityField.scalarField(0));
  //FIELDVIEW3D(velocityField.scalarField(1));

  // fire down some random particles
  MERSENNETWISTER twister(123456);
  int totalParticles = 10000;
  //int totalParticles = 1000;
  for (int x = 0; x < totalParticles; x++)
  {
    VEC3F particle;
    //particle[0] = twister.rand(lengths[0]);
    //particle[1] = twister.rand(lengths[1]);
    //particle[2] = twister.rand(lengths[2]);
    particle[0] = twister.rand();
    particle[1] = twister.rand();
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

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
void runEverytime()
{
  stepEigenfunctions();
  
  TIMER advectionTimer("Particle advection");
#pragma omp parallel
#pragma omp for  schedule(dynamic)
  for (int x = 0; x < particles.size(); x++)
  {
    particles[x] += dt * velocityField(particles[x]);

    if (ribbons[x].size() > 20)
      ribbons[x].pop_front();
    ribbons[x].push_back(particles[x]);

    if (velocityRibbons[x].size() > 20)
      velocityRibbons[x].pop_front();
    velocityRibbons[x].push_back(velocityField(particles[x]));
  }
}
/*
{
  const Real dt = 0.01;
  for (int x = 0; x < particles.size(); x++)
  {
    particles[x] += dt * velocityField(particles[x]);

    if (ribbons[x].size() > 20)
      ribbons[x].pop_front();
    ribbons[x].push_back(particles[x]);

    if (velocityRibbons[x].size() > 20)
      velocityRibbons[x].pop_front();
    velocityRibbons[x].push_back(velocityField(particles[x]));
  }
}
*/
