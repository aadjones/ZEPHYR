#ifndef FIELD_2D_H
#define FIELD_2D_H

#include <cmath>
#include <string>
#include <iostream>
//#include "VEC3F.h"
#include "VEC3.h"

#define USING_FFTW 1
#if USING_FFTW
#include <fftw3.h>
#endif

#ifndef VARNAME
#define VARNAME(x) #x
#endif
#ifndef FIELDVIEW2D
#define FIELDVIEW2D(x) FIELD_2D::fieldViewer(x, VARNAME(x)); sleep(1);
#endif

using namespace std;

class FIELD_2D {
public:
  FIELD_2D();
  FIELD_2D(const int& rows, const int& cols, const VEC3F& lengths = VEC3F(1,1,0));
  FIELD_2D(const FIELD_2D& m);
  ~FIELD_2D();

  // accessors
  inline Real& operator()(int x, int y) { return _data[y * _xRes + x]; };
  const Real operator()(int x, int y) const { return _data[y * _xRes + x]; };
  inline Real& operator[](int x) { return _data[x]; };
  const Real operator[](int x) const { return _data[x]; };
  Real* data() { return _data; };
  const int xRes() const { return _xRes; };
  const int yRes() const { return _yRes; };
  const int totalCells() const { return _totalCells; };
  const VEC3F& lengths() const { return _lengths; };

  // common field operations
  void clear();
  void normalize();
  FIELD_2D& abs();

  Real min();
  Real max();

  // field maximum cell index
  VEC3F maxIndex();

  // field minimum cell index
  VEC3F minIndex();

  // max absolute index
  int maxAbsIndex();

  // take the log
  void log(Real base = 2.0);
 
  // generic IO functions
  void writeMatlab(string filename, string variableName) const;
  void write(string filename) const;
  void read(string filename);

  // some image file support
  void writePPM(string filename);
  void writeJPG(string filename);
  void writePNG(string filename);
  void readPNG(string filename);

  void resizeAndWipe(int xRes, int yRes);

  // overloaded operators
  FIELD_2D& operator=(const Real& alpha);
  FIELD_2D& operator=(const FIELD_2D& A);
  FIELD_2D& operator*=(const Real& alpha);
  FIELD_2D& operator/=(const Real& alpha);
  FIELD_2D& operator+=(const Real& alpha);
  FIELD_2D& operator-=(const FIELD_2D& input);
  FIELD_2D& operator+=(const FIELD_2D& input);
  FIELD_2D& operator*=(const FIELD_2D& input);
  FIELD_2D& operator/=(const FIELD_2D& input);

  // sum of all entries
  Real sum();
  
  // set to a checkboard for debugging
  void setToCheckerboard(int xChecks = 10, int yChecks = 10);
  
  // pass a field to fieldViewer2D
  static void fieldViewer(const FIELD_2D& field, string name);

  // take the spatial derivative
  Real Dx(const int x, const int y) const;
  Real Dy(const int x, const int y) const;
  FIELD_2D Dx() const;
  FIELD_2D Dy() const;
  FIELD_2D Dz() const;

  // flip the X and Y of the current field
  FIELD_2D flipXY();

  // NOTE: these are all in-place transforms
  void yDCT();
  void xDCT();

  void xDST();
  void yDST();
  
  void xIDCT();
  void yIDCT();

  void xIDST();
  void yIDST();

  void xIDSTyIDCT();
  void xIDCTyIDST();

#if USING_FFTW
  void xIDCT(const fftw_plan& plan);
  void yIDCT(const fftw_plan& plan);
  void xPlanIDCT(fftw_plan& plan);
  void yPlanIDCT(fftw_plan& plan);

  void xIDST(const fftw_plan& plan);
  void yIDST(const fftw_plan& plan);
  void xPlanIDST(fftw_plan& plan);
  void yPlanIDST(fftw_plan& plan);
  
  void xIDSTyIDCT(const fftw_plan& plan);
  void xPlanIDSTyIDCT(fftw_plan& plan);
  
  void xIDCTyIDST(const fftw_plan& plan);
  void xPlanIDCTyIDST(fftw_plan& plan);
#endif

private:
  int _xRes;
  int _yRes;

  VEC3F _lengths;
  Real _dx;
  Real _dy;

  int _totalCells;
  Real* _data;
};

FIELD_2D operator*(const FIELD_2D& A, const Real alpha);
FIELD_2D operator/(const FIELD_2D& A, const Real alpha);
FIELD_2D operator+(const FIELD_2D& A, const Real alpha);
FIELD_2D operator*(const Real alpha, const FIELD_2D& A);
FIELD_2D operator+(const Real alpha, const FIELD_2D& A);
FIELD_2D operator-(const FIELD_2D& A, const FIELD_2D& B);
FIELD_2D operator+(const FIELD_2D& A, const FIELD_2D& B);

#endif
