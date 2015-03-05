#include "EIGEN.h"

#include <cmath>

#include "FIELD_2D.h"
#include "FIELD_3D.h"
#include "VECTOR3_FIELD_3D.h"
#include "VEC3.h"

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/gl.h> // OpenGL itself.
#include <GL/glu.h> // GLU support library.
#include <GL/glut.h> // GLUT support library.
#endif

#include <iostream>
#include <QUICKTIME_MOVIE.h>
using namespace std;

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{

  VECTOR data("data.vector");
  int xRes = 48;
  int yRes = 64;
  int zRes = 48;
  VECTOR3_FIELD_3D Test(xRes, yRes, zRes);
  for (int x = 0; x < xRes; x++) {
    for (int y = 0; y < yRes; y++) {
      for (int z = 0; z < zRes; z++) {
        Test(x, y, z) = VEC3F(sin(y), sin(z), sin(x));
      }
    }
  }
  Test.write("test.vectorfield");
  FIELD_3D Distance(xRes, yRes, zRes);
  for (int x = 0; x < xRes; x++) {
    for (int y = 0; y < yRes; y++) {
      for (int z = 0; z < zRes; z++) {
        Distance(x, y, z) = log(1 + x*x + y*y + z*z);
      }
    }
  }
  Distance.write("distance.scalarfield");
 return 0;
}
