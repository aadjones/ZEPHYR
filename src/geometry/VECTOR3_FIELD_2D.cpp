#include "VECTOR3_FIELD_2D.h"
//#include <omp.h>

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/gl.h> // OpenGL itself.
//#include <GL/glu.h> // GLU support library.
#include <GL/glut.h> // GLUT support library.
#endif

#include "MERSENNETWISTER.h"
#include "MATRIX3.h"

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
VECTOR3_FIELD_2D::VECTOR3_FIELD_2D(const int& xRes, const int& yRes, const VEC3F& center, const VEC3F& lengths) :
  _xRes(xRes), _yRes(yRes), _center(center), _lengths(lengths), _initialized(true)
{
  _totalCells = _xRes * _yRes;
  _data = new VEC3F[_totalCells];

  _dx = _lengths[0] / _xRes;
  _dy = _lengths[1] / _yRes;
  _invDx = 1.0 / _dx;
  _invDy = 1.0 / _dy;
}

VECTOR3_FIELD_2D::VECTOR3_FIELD_2D(double* data, const int& xRes, const int& yRes, const VEC3F& center, const VEC3F& lengths) :
  _xRes(xRes), _yRes(yRes), _center(center), _lengths(lengths), _initialized(true)
{
  _totalCells = _xRes * _yRes;
  _data = new VEC3F[_totalCells];

  _dx = _lengths[0] / _xRes;
  _dy = _lengths[1] / _yRes;
  _invDx = 1.0 / _dx;
  _invDy = 1.0 / _dy;

  for (int x = 0; x < _totalCells; x++)
  {
    _data[x][0] = data[3 * x];
    _data[x][1] = data[3 * x + 1];
  }
}

VECTOR3_FIELD_2D::VECTOR3_FIELD_2D(float* xData, float* yData, const int& xRes, const int& yRes, 
    const VEC3F& center, const VEC3F& lengths) :
  _xRes(xRes), _yRes(yRes), _center(center), _lengths(lengths), _initialized(true)
{
  _totalCells = _xRes * _yRes;
  _data = new VEC3F[_totalCells];

  _dx = _lengths[0] / _xRes;
  _dy = _lengths[1] / _yRes;
  _invDx = 1.0 / _dx;
  _invDy = 1.0 / _dy;

  for (int x = 0; x < _totalCells; x++)
  {
    _data[x][0] = xData[x];
    _data[x][1] = yData[x];
  }
}

VECTOR3_FIELD_2D::VECTOR3_FIELD_2D(const VECTOR3_FIELD_2D& m) :
  _xRes(m.xRes()), _yRes(m.yRes()), _center(m.center()), _lengths(m.lengths()), _initialized(true)
{
  _totalCells = _xRes * _yRes;
  _data = new VEC3F[_totalCells];

  _dx = _lengths[0] / _xRes;
  _dy = _lengths[1] / _yRes;
  _invDx = 1.0 / _dx;
  _invDy = 1.0 / _dy;

  for (int x = 0; x < _totalCells; x++)
    _data[x] = m[x];
}

VECTOR3_FIELD_2D::VECTOR3_FIELD_2D(const FIELD_2D& m) :
  _xRes(m.xRes()), _yRes(m.yRes()), _initialized(true)
{
  _totalCells = _xRes * _yRes;
  _data = new VEC3F[_totalCells];

  _lengths[0] = 1.0;
  _lengths[1] = 1.0;
  _lengths[2] = 1.0;

  _dx = _lengths[0] / _xRes;
  _dy = _lengths[1] / _yRes;
  _invDx = 1.0 / _dx;
  _invDy = 1.0 / _dy;

  for (int x = 0; x < _totalCells; x++)
    _data[x] = 0;

  cout << " cells: " << _totalCells << endl;
  cout << " zero entry: " << _data[0] << endl;

  // do the middle x gradient
  for (int y = 0; y < _yRes; y++)
    for (int x = 1; x < _xRes - 1; x++)
      (*this)(x,y)[0] = (m(x+1, y) - m(x-1,y)) * 0.5;

  // do the left x gradient
  for (int y = 0; y < _yRes; y++)
    (*this)(0,y)[0] = m(1, y) - m(0,y);

  // do the right x gradient
  for (int y = 0; y < _yRes; y++)
    (*this)(_xRes - 1,y)[0] = m(_xRes - 1, y) - m(_xRes - 2,y);

  // do the middle y gradient
  for (int y = 1; y < _yRes - 1; y++)
    for (int x = 0; x < _xRes; x++)
      (*this)(x,y)[1] = (m(x, y+1) - m(x,y-1)) * 0.5;

  // do the bottom y gradient
  for (int x = 0; x < _xRes; x++)
    (*this)(x,0)[1] = m(x,1) - m(x,0);
  
  // do the top y gradient
  for (int x = 0; x < _xRes; x++)
    (*this)(x,_yRes - 1)[1] = m(x,_yRes - 1) - m(x,_yRes - 2);

  for (int y = 0; y < _yRes; y++)
    for (int x = 0; x < _xRes; x++)
    {
      (*this)(x,y) *= 1.0 / _xRes;

      float xComp = (*this)(x,y)[0];

      (*this)(x,y)[0] = (*this)(x,y)[1];
      (*this)(x,y)[1] = -xComp;
    }
}


VECTOR3_FIELD_2D::VECTOR3_FIELD_2D() :
  _xRes(-1), _yRes(-1), _totalCells(-1), _data(NULL), _initialized(false)
{
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
VECTOR3_FIELD_2D::~VECTOR3_FIELD_2D()
{
  delete[] _data;
}
  
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
void VECTOR3_FIELD_2D::clear()
{
  for (int x = 0; x < _totalCells; x++)
    _data[x] = 0.0;
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
VECTOR3_FIELD_2D& VECTOR3_FIELD_2D::operator=(const VECTOR3_FIELD_2D& input)
{
  if (input.xRes() != _xRes || input.yRes() != _yRes) 
  {
    delete[] _data;

    _xRes = input.xRes();
    _yRes = input.yRes();

    _totalCells = _xRes * _yRes;
    _data = new VEC3F[_totalCells];
  }

  _center = input.center();
  _lengths = input.lengths();

  _dx = _lengths[0] / _xRes;
  _dy = _lengths[1] / _yRes;
  _invDx = 1.0 / _dx;
  _invDy = 1.0 / _dy;
  
  for (int x = 0; x < _totalCells; x++)
    _data[x] = input[x];

  _initialized = input._initialized;

  return *this;
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
FIELD_2D VECTOR3_FIELD_2D::scalarField(int component) const
{
  //assert(component >= 0 && component < 2);
  assert(component >= 0 && component <= 2);

  FIELD_2D final(_xRes, _yRes);

  for (int x = 0; x < _totalCells; x++)
    final[x] = _data[x][component];

  return final;
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
FIELD_2D VECTOR3_FIELD_2D::magnitudeField() const
{
  FIELD_2D final(_xRes, _yRes);

  for (int x = 0; x < _totalCells; x++)
    final[x] = norm(_data[x]);

  return final;
}

///////////////////////////////////////////////////////////////////////
// take the field dot product
///////////////////////////////////////////////////////////////////////
FIELD_2D operator*(const VECTOR3_FIELD_2D&u, const VECTOR3_FIELD_2D& v)
{
  assert(u.xRes() == v.xRes());
  assert(u.yRes() == v.yRes());

  FIELD_2D final(u.xRes(), u.yRes());

  int totalCells = u.totalCells();
  for (int x = 0; x < totalCells; x++)
    final[x] = u[x] * v[x];

  return final;
}

///////////////////////////////////////////////////////////////////////
// take the field dot product
///////////////////////////////////////////////////////////////////////
VECTOR3_FIELD_2D operator*(const FIELD_2D&u, const VECTOR3_FIELD_2D& v)
{
  assert(u.xRes() == v.xRes());
  assert(u.yRes() == v.yRes());

  VECTOR3_FIELD_2D final(v);

  int totalCells = u.totalCells();
  for (int x = 0; x < totalCells; x++)
  {
    Real uReal = u[x];
    VEC3F vVec = v[x];
    VEC3F product = uReal * vVec;
    final[x] = product;
  }

  return final;
}

///////////////////////////////////////////////////////////////////////
// lookup value at some real-valued spatial position
///////////////////////////////////////////////////////////////////////
const VEC3F VECTOR3_FIELD_2D::operator()(const VEC3F& position) const
{
  VEC3F positionCopy = position;

  // get the lower corner position
  VEC3F corner = _center - (Real)0.5 * _lengths;
  VEC3F dxs(_dx, _dy, 0);
  corner += (Real)0.5 * dxs;

  // recenter position
  positionCopy -= corner;

  positionCopy[0] *= _invDx;
  positionCopy[1] *= _invDy;
  positionCopy[2] *= 0;

  int x0 = (int)positionCopy[0];
  int x1    = x0 + 1;
  int y0 = (int)positionCopy[1];
  int y1    = y0 + 1;

  // clamp everything
  x0 = (x0 < 0) ? 0 : x0;
  y0 = (y0 < 0) ? 0 : y0;
  
  x1 = (x1 < 0) ? 0 : x1;
  y1 = (y1 < 0) ? 0 : y1;

  x0 = (x0 > _xRes - 1) ? _xRes - 1 : x0;
  y0 = (y0 > _yRes - 1) ? _yRes - 1 : y0;

  x1 = (x1 > _xRes - 1) ? _xRes - 1 : x1;
  y1 = (y1 > _yRes - 1) ? _yRes - 1 : y1;

  // get interpolation weights
  const Real s1 = positionCopy[0]- x0;
  const Real s0 = 1.0f - s1;
  const Real t1 = positionCopy[1]- y0;
  const Real t0 = 1.0f - t1;

  const int i00 = x0 + y0 * _xRes;
  const int i01 = x0 + y1 * _xRes;
  const int i10 = x1 + y0 * _xRes;
  const int i11 = x1 + y1 * _xRes;

  // interpolate
  // (indices could be computed once)
  return s0 * (t0 * _data[i00] + t1 * _data[i01]) +
         s1 * (t0 * _data[i10] + t1 * _data[i11]);
}
/*
{
  int x0 = (int)position[0];
  int x1    = x0 + 1;
  int y0 = (int)position[1];
  int y1    = y0 + 1;

  // clamp everything
  x0 = (x0 < 0) ? 0 : x0;
  y0 = (y0 < 0) ? 0 : y0;
  
  x1 = (x1 < 0) ? 0 : x1;
  y1 = (y1 < 0) ? 0 : y1;

  x0 = (x0 > _xRes - 1) ? _xRes - 1 : x0;
  y0 = (y0 > _yRes - 1) ? _yRes - 1 : y0;

  x1 = (x1 > _xRes - 1) ? _xRes - 1 : x1;
  y1 = (y1 > _yRes - 1) ? _yRes - 1 : y1;

  // get interpolation weights
  const Real s1 = position[0]- x0;
  const Real s0 = 1.0f - s1;
  const Real t1 = position[1]- y0;
  const Real t0 = 1.0f - t1;

  const int i00 = x0 + y0 * _xRes;
  const int i01 = x0 + y1 * _xRes;
  const int i10 = x1 + y0 * _xRes;
  const int i11 = x1 + y1 * _xRes;

  // interpolate
  // (indices could be computed once)
  return s0 * (t0 * _data[i00] + t1 * _data[i01]) +
         s1 * (t0 * _data[i10] + t1 * _data[i11]);
}
*/

///////////////////////////////////////////////////////////////////////
// check if any entry is a nan
///////////////////////////////////////////////////////////////////////
bool VECTOR3_FIELD_2D::isNan()
{
  for (int x = 0; x < _totalCells; x++)
    for (int y = 0; y < 3; y++)
      if (isnan(_data[x][y]))
        return true;

  return false;
}

///////////////////////////////////////////////////////////////////////
// real-valued cell center coordinates
///////////////////////////////////////////////////////////////////////
VEC3F VECTOR3_FIELD_2D::cellCenter(int x, int y) const
{
  VEC3F halfLengths = (Real)0.5 * _lengths;

  // set it to the lower corner
  VEC3F final = _center - halfLengths;

  // displace to the NNN corner
  final[0] += x * _dx;
  final[1] += y * _dy;

  // displace it to the cell center
  final[0] += _dx * 0.5;
  final[1] += _dy * 0.5;

  return final;
}

///////////////////////////////////////////////////////////////////////
// normalize all the vectors in the field
///////////////////////////////////////////////////////////////////////
void VECTOR3_FIELD_2D::normalize()
{
  for (int x = 0; x < _totalCells; x++)
    _data[x].normalize();
}

//////////////////////////////////////////////////////////////////////
// draw to GL
//////////////////////////////////////////////////////////////////////
void VECTOR3_FIELD_2D::draw()
{
  //float radius = 0.001;
  int stride = 1;
  Real scale = _dx;
  
  glBegin(GL_LINES);
    for (int y = 0; y < _yRes; y+=stride)
      for (int x = 0; x < _xRes; x+=stride)
      {
        const VEC3F& origin = cellCenter(x,y);
        //VEC3F endpoint = origin + scale * (*this)(x,y);
        VEC3F endpoint = origin + scale * (*this)[x + y * _xRes];

        glColor4f(1,1,1,1);
        glVertex3f(origin[0], origin[1], origin[2]);
        glColor4f(0,0,0,0);
        glVertex3f(endpoint[0], endpoint[1], endpoint[2]);
      }
  glEnd();
}

//////////////////////////////////////////////////////////////////////
// write a LIC image
// http://www.zhanpingliu.org/research/flowvis/LIC/MiniLIC.c
//////////////////////////////////////////////////////////////////////
void VECTOR3_FIELD_2D::writeLIC(int scaleUp, const char* filename)
{
  // create the filters
  int filterSize = 64;
  float* forwardFilter = new float[filterSize];
  float* backwardFilter = new float[filterSize];
  for(int i = 0; i < filterSize; i++)  
    forwardFilter[i] = backwardFilter[i] = i;

  float kernelLength = 10;
  VECTOR3_FIELD_2D& vectorField = *this;
  int xRes = vectorField.xRes() * scaleUp;
  int yRes = vectorField.yRes() * scaleUp;

  cout << " Writing LIC resolution: " << xRes << " " << yRes << endl;

  // generate a noise field
  FIELD_2D noiseField(xRes, yRes);
  for(int j = 0; j < yRes; j++)
    for(int i = 0; i < xRes; i++)
    { 
      int  r = rand();
      r = (  (r & 0xff) + ( (r & 0xff00) >> 8 )  ) & 0xff;
      noiseField(i,j) = (unsigned char) r;
    }

  // create the final image field
  FIELD_2D finalImage(xRes, yRes);

  int   maxAdvects = kernelLength * 3;
  float len2ID = (filterSize - 1) / kernelLength; ///map a curve LENgth TO an ID in the LUT

  ///for each pixel in the 2D output LIC image///
  for (int j = 0; j < yRes; j++)
    for (int i = 0; i < xRes; i++)
    { 
      ///init the composite texture accumulators and the weight accumulators///
      float textureAccum[] = {0,0};
      float weightAccum[]  = {0,0};
      float textureValue = 0;
    
      ///for either advection direction///
      for(int advectionDirection = 0; advectionDirection < 2; advectionDirection++)
      { 
        ///init the step counter, curve-length measurer, and streamline seed///
        int advects = 0;
        float currentLength = 0.0f;
        float clippedX0 = i + 0.5f;
        float clippedY0 = j + 0.5f;

        ///access the target filter LUT///
        float* weightLUT = (advectionDirection == 0) ? forwardFilter : backwardFilter;

        /// until the streamline is advected long enough or a tightly  spiralling center / focus is encountered///
        while (currentLength < kernelLength && advects < maxAdvects) 
        {
          ///access the vector at the sample
          VEC3F position((float)i / scaleUp, (float)j / scaleUp, 0);
          float vectorX = vectorField(position)[0];
          float vectorY = vectorField(position)[1];

          /// negate the vector for the backward-advection case///
          vectorX = (advectionDirection == 0) ? vectorX : -vectorX;
          vectorY = (advectionDirection == 0) ? vectorY : -vectorY;

          ///clip the segment against the pixel boundaries --- find the shorter from the two clipped segments///
          ///replace  all  if-statements  whenever  possible  as  they  might  affect the computational speed///
          const float lineSquare = 100000;
          const float vectorMin = 0.05;
          float segmentLength = lineSquare;
          segmentLength = (vectorX < -vectorMin) ? ( int(     clippedX0         ) - clippedX0 ) / vectorX : segmentLength;
          segmentLength = (vectorX >  vectorMin) ? ( int( int(clippedX0) + 1.5f ) - clippedX0 ) / vectorX : segmentLength;

          if (vectorY < -vectorMin)
          {
            float tmpLength = (int(clippedY0) - clippedY0) / vectorY;
            
            if (tmpLength < segmentLength) 
              segmentLength = tmpLength;
          }

          if (vectorY > vectorMin)
          {
            float tmpLength = (int(int(clippedY0) + 1.5f) - clippedY0) / vectorY;
            if (tmpLength <  segmentLength)
              segmentLength = tmpLength;
          }
          
          ///update the curve-length measurers///
          float previousLength  = currentLength;
          currentLength += segmentLength;
          segmentLength += 0.0004f;
         
          ///check if the filter has reached either end///
          segmentLength = (currentLength > kernelLength) ? ( (currentLength = kernelLength) - previousLength ) : segmentLength;

          ///obtain the next clip point///
          float clippedX1 = clippedX0 + vectorX * segmentLength;
          float clippedY1 = clippedY0 + vectorY * segmentLength;

          ///obtain the middle point of the segment as the texture-contributing sample///
          float sampleX = (clippedX0 + clippedX1) * 0.5f;
          float sampleY = (clippedY0 + clippedY1) * 0.5f;

          ///obtain the texture value of the sample///
          textureValue = noiseField(sampleX, sampleY);

          ///update the accumulated weight and the accumulated composite texture (texture x weight)
          float currentWeightAccum = weightLUT[ int(currentLength * len2ID) ];
          float sampleWeight = currentWeightAccum - weightAccum[advectionDirection];     
          weightAccum[advectionDirection] = currentWeightAccum;               
          textureAccum[advectionDirection] += textureValue * sampleWeight;
        
          ///update the step counter and the "current" clip point
          advects++;
          clippedX0 = clippedX1;
          clippedY0 = clippedY1;

          ///check if the streamline has gone beyond the flow field
          if (clippedX0 < 0.0f || clippedX0 >= xRes ||
              clippedY0 < 0.0f || clippedY0 >= yRes)  break;
        } 
      }

      ///normalize the accumulated composite texture
      textureValue = (textureAccum[0] + textureAccum[1]) / (weightAccum[0] + weightAccum[1]);

      ///clamp the texture value against the displayable intensity range [0, 255]
      textureValue = (textureValue <   0.0f) ?   0.0f : textureValue;
      textureValue = (textureValue > 255.0f) ? 255.0f : textureValue; 
      finalImage(i,j) = textureValue / 255.0;
    }

  finalImage.writeJPG(filename);
  delete[] forwardFilter;
  delete[] backwardFilter;
}

//////////////////////////////////////////////////////////////////////
// write a LIC image
// http://www.zhanpingliu.org/research/flowvis/LIC/MiniLIC.c
//////////////////////////////////////////////////////////////////////
void VECTOR3_FIELD_2D::writeLIC(const char* filename)
{
  // create the filters
  int filterSize = 64;
  float* forwardFilter = new float[filterSize];
  float* backwardFilter = new float[filterSize];
  for(int i = 0; i < filterSize; i++)  
    forwardFilter[i] = backwardFilter[i] = i;

  float kernelLength = 10;
  VECTOR3_FIELD_2D& vectorField = *this;
  int xRes = vectorField.xRes();
  int yRes = vectorField.yRes();

  // generate a noise field
  FIELD_2D noiseField(xRes, yRes);
  for(int j = 0; j < yRes; j++)
    for(int i = 0; i < xRes; i++)
    { 
      int  r = rand();
      r = (  (r & 0xff) + ( (r & 0xff00) >> 8 )  ) & 0xff;
      noiseField(i,j) = (unsigned char) r;
    }

  // create the final image field
  FIELD_2D finalImage(xRes, yRes);

  int   maxAdvects = kernelLength * 3;
  float len2ID = (filterSize - 1) / kernelLength; ///map a curve LENgth TO an ID in the LUT

  ///for each pixel in the 2D output LIC image///
  for (int j = 0; j < yRes; j++)
    for (int i = 0; i < xRes; i++)
    { 
      ///init the composite texture accumulators and the weight accumulators///
      float textureAccum[] = {0,0};
      float weightAccum[]  = {0,0};
      float textureValue = 0;
    
      ///for either advection direction///
      for(int advectionDirection = 0; advectionDirection < 2; advectionDirection++)
      { 
        ///init the step counter, curve-length measurer, and streamline seed///
        int advects = 0;
        float currentLength = 0.0f;
        float clippedX0 = i + 0.5f;
        float clippedY0 = j + 0.5f;

        ///access the target filter LUT///
        float* weightLUT = (advectionDirection == 0) ? forwardFilter : backwardFilter;

        /// until the streamline is advected long enough or a tightly  spiralling center / focus is encountered///
        while (currentLength < kernelLength && advects < maxAdvects) 
        {
          ///access the vector at the sample
          float vectorX = vectorField(i,j)[0];
          float vectorY = vectorField(i,j)[1];

          /// negate the vector for the backward-advection case///
          vectorX = (advectionDirection == 0) ? vectorX : -vectorX;
          vectorY = (advectionDirection == 0) ? vectorY : -vectorY;

          ///clip the segment against the pixel boundaries --- find the shorter from the two clipped segments///
          ///replace  all  if-statements  whenever  possible  as  they  might  affect the computational speed///
          const float lineSquare = 100000;
          const float vectorMin = 0.05;
          float segmentLength = lineSquare;
          segmentLength = (vectorX < -vectorMin) ? ( int(     clippedX0         ) - clippedX0 ) / vectorX : segmentLength;
          segmentLength = (vectorX >  vectorMin) ? ( int( int(clippedX0) + 1.5f ) - clippedX0 ) / vectorX : segmentLength;

          if (vectorY < -vectorMin)
          {
            float tmpLength = (int(clippedY0) - clippedY0) / vectorY;
            
            if (tmpLength < segmentLength) 
              segmentLength = tmpLength;
          }

          if (vectorY > vectorMin)
          {
            float tmpLength = (int(int(clippedY0) + 1.5f) - clippedY0) / vectorY;
            if (tmpLength <  segmentLength)
              segmentLength = tmpLength;
          }
          
          ///update the curve-length measurers///
          float previousLength  = currentLength;
          currentLength += segmentLength;
          segmentLength += 0.0004f;
         
          ///check if the filter has reached either end///
          segmentLength = (currentLength > kernelLength) ? ( (currentLength = kernelLength) - previousLength ) : segmentLength;

          ///obtain the next clip point///
          float clippedX1 = clippedX0 + vectorX * segmentLength;
          float clippedY1 = clippedY0 + vectorY * segmentLength;

          ///obtain the middle point of the segment as the texture-contributing sample///
          float sampleX = (clippedX0 + clippedX1) * 0.5f;
          float sampleY = (clippedY0 + clippedY1) * 0.5f;

          ///obtain the texture value of the sample///
          textureValue = noiseField(sampleX, sampleY);

          ///update the accumulated weight and the accumulated composite texture (texture x weight)
          float currentWeightAccum = weightLUT[ int(currentLength * len2ID) ];
          float sampleWeight = currentWeightAccum - weightAccum[advectionDirection];     
          weightAccum[advectionDirection] = currentWeightAccum;               
          textureAccum[advectionDirection] += textureValue * sampleWeight;
        
          ///update the step counter and the "current" clip point
          advects++;
          clippedX0 = clippedX1;
          clippedY0 = clippedY1;

          ///check if the streamline has gone beyond the flow field
          if (clippedX0 < 0.0f || clippedX0 >= xRes ||
              clippedY0 < 0.0f || clippedY0 >= yRes)  break;
        } 
      }

      ///normalize the accumulated composite texture
      textureValue = (textureAccum[0] + textureAccum[1]) / (weightAccum[0] + weightAccum[1]);

      ///clamp the texture value against the displayable intensity range [0, 255]
      textureValue = (textureValue <   0.0f) ?   0.0f : textureValue;
      textureValue = (textureValue > 255.0f) ? 255.0f : textureValue; 
      finalImage(i,j) = textureValue / 255.0;
    }

  finalImage.writeJPG(filename);
  delete[] forwardFilter;
  delete[] backwardFilter;
}

//////////////////////////////////////////////////////////////////////
// write a LIC image
// http://www.zhanpingliu.org/research/flowvis/LIC/MiniLIC.c
//////////////////////////////////////////////////////////////////////
FIELD_2D VECTOR3_FIELD_2D::LIC()
{
  // create the filters
  int filterSize = 64;
  float* forwardFilter = new float[filterSize];
  float* backwardFilter = new float[filterSize];
  for(int i = 0; i < filterSize; i++)  
    forwardFilter[i] = backwardFilter[i] = i;

  float kernelLength = 10;
  VECTOR3_FIELD_2D& vectorField = *this;
  int xRes = vectorField.xRes();
  int yRes = vectorField.yRes();

  // generate a noise field
  FIELD_2D noiseField(xRes, yRes);
  MERSENNETWISTER twister(123456);
  for(int j = 0; j < yRes; j++)
    for(int i = 0; i < xRes; i++)
    { 
      Real r = 255 * twister.rand();
      //r = (  (r & 0xff) + ( (r & 0xff00) >> 8 )  ) & 0xff;
      noiseField(i,j) = r;
    }

  // create the final image field
  FIELD_2D finalImage(xRes, yRes);

  int   maxAdvects = kernelLength * 3;
  float len2ID = (filterSize - 1) / kernelLength; ///map a curve LENgth TO an ID in the LUT

  ///for each pixel in the 2D output LIC image///
  for (int j = 0; j < yRes; j++)
    for (int i = 0; i < xRes; i++)
    { 
      ///init the composite texture accumulators and the weight accumulators///
      float textureAccum[] = {0,0};
      float weightAccum[]  = {0,0};
      float textureValue = 0;
    
      ///for either advection direction///
      for(int advectionDirection = 0; advectionDirection < 2; advectionDirection++)
      { 
        ///init the step counter, curve-length measurer, and streamline seed///
        int advects = 0;
        float currentLength = 0.0f;
        float clippedX0 = i + 0.5f;
        float clippedY0 = j + 0.5f;

        ///access the target filter LUT///
        float* weightLUT = (advectionDirection == 0) ? forwardFilter : backwardFilter;

        /// until the streamline is advected long enough or a tightly  spiralling center / focus is encountered///
        while (currentLength < kernelLength && advects < maxAdvects) 
        {
          ///access the vector at the sample
          float vectorX = vectorField(i,j)[0];
          float vectorY = vectorField(i,j)[1];

          /// negate the vector for the backward-advection case///
          vectorX = (advectionDirection == 0) ? vectorX : -vectorX;
          vectorY = (advectionDirection == 0) ? vectorY : -vectorY;

          ///clip the segment against the pixel boundaries --- find the shorter from the two clipped segments///
          ///replace  all  if-statements  whenever  possible  as  they  might  affect the computational speed///
          const float lineSquare = 100000;
          const float vectorMin = 0.05;
          float segmentLength = lineSquare;
          segmentLength = (vectorX < -vectorMin) ? ( int(     clippedX0         ) - clippedX0 ) / vectorX : segmentLength;
          segmentLength = (vectorX >  vectorMin) ? ( int( int(clippedX0) + 1.5f ) - clippedX0 ) / vectorX : segmentLength;

          if (vectorY < -vectorMin)
          {
            float tmpLength = (int(clippedY0) - clippedY0) / vectorY;
            
            if (tmpLength < segmentLength) 
              segmentLength = tmpLength;
          }

          if (vectorY > vectorMin)
          {
            float tmpLength = (int(int(clippedY0) + 1.5f) - clippedY0) / vectorY;
            if (tmpLength <  segmentLength)
              segmentLength = tmpLength;
          }
          
          ///update the curve-length measurers///
          float previousLength  = currentLength;
          currentLength += segmentLength;
          segmentLength += 0.0004f;
         
          ///check if the filter has reached either end///
          segmentLength = (currentLength > kernelLength) ? ( (currentLength = kernelLength) - previousLength ) : segmentLength;

          ///obtain the next clip point///
          float clippedX1 = clippedX0 + vectorX * segmentLength;
          float clippedY1 = clippedY0 + vectorY * segmentLength;

          ///obtain the middle point of the segment as the texture-contributing sample///
          float sampleX = (clippedX0 + clippedX1) * 0.5f;
          float sampleY = (clippedY0 + clippedY1) * 0.5f;

          ///obtain the texture value of the sample///
          textureValue = noiseField(sampleX, sampleY);

          ///update the accumulated weight and the accumulated composite texture (texture x weight)
          float currentWeightAccum = weightLUT[ int(currentLength * len2ID) ];
          float sampleWeight = currentWeightAccum - weightAccum[advectionDirection];     
          weightAccum[advectionDirection] = currentWeightAccum;               
          textureAccum[advectionDirection] += textureValue * sampleWeight;
        
          ///update the step counter and the "current" clip point
          advects++;
          clippedX0 = clippedX1;
          clippedY0 = clippedY1;

          ///check if the streamline has gone beyond the flow field
          if (clippedX0 < 0.0f || clippedX0 >= xRes ||
              clippedY0 < 0.0f || clippedY0 >= yRes)  break;
        } 
      }

      ///normalize the accumulated composite texture
      textureValue = (textureAccum[0] + textureAccum[1]) / (weightAccum[0] + weightAccum[1]);

      ///clamp the texture value against the displayable intensity range [0, 255]
      textureValue = (textureValue <   0.0f) ?   0.0f : textureValue;
      textureValue = (textureValue > 255.0f) ? 255.0f : textureValue; 
      finalImage(i,j) = textureValue / 255.0;
    }

  delete[] forwardFilter;
  delete[] backwardFilter;

  return finalImage;
}

//////////////////////////////////////////////////////////////////////
// compute the curl of two crossed vorticity functions
//////////////////////////////////////////////////////////////////////
void VECTOR3_FIELD_2D::curlCrossedVorticity(int a1, int a2, int b1, int b2)
{
  Real dx = 3.14f / (_xRes - 1);
  Real dy = 3.14f / (_yRes - 1);

  for (int j = 0; j < _yRes; j++)
    for (int i = 0; i < _xRes; i++)
    {
      Real x = i * dx;
      Real y = j * dy;

      // a_1 b_2 \cos(a_1 x) \cos(b_2 y) \sin(a_2 x) \sin(b_1 y) -
      // a_2 b_1 \cos(a_2 x) \cos(b_1 y) \sin(a_1 x) \sin(b_2 y).
      //Real cross = a1 * b2 * cos(a1 * x) * cos(b2 * y) * sin(a2 * x) * sin(b1 * y) -
      //             a2 * b1 * cos(a2 * x) * cos(b1 * y) * sin(a1 * x) * sin(b2 * y);
      Real cross = a1 * b2 * cos(a1 * x) * cos(b2 * y) * sin(a2 * y) * sin(b1 * x) -
                   a2 * b1 * cos(a2 * y) * cos(b1 * x) * sin(a1 * x) * sin(b2 * y);
                   
      (*this)(i,j)[0] = 0;
      (*this)(i,j)[1] = 0;
      (*this)(i,j)[2] = cross; 
    }
  return;
}

//////////////////////////////////////////////////////////////////////
// compute the curl of two crossed vorticity functions, and project
// it onto a third
//////////////////////////////////////////////////////////////////////
void VECTOR3_FIELD_2D::dotCurledCrossed(int a1, int a2, int b1, int b2, int k1, int k2)
{
  Real dx = 3.14f / (_xRes - 1);
  Real dy = 3.14f / (_yRes - 1);

  for (int j = 0; j < _yRes; j++)
    for (int i = 0; i < _xRes; i++)
    {
      Real x = i * dx;
      Real y = j * dy;

      // a_1 b_2 \cos(a_1 x) \cos(b_2 y) \sin(a_2 x) \sin(b_1 y) -
      // a_2 b_1 \cos(a_2 x) \cos(b_1 y) \sin(a_1 x) \sin(b_2 y).
      Real cross = a1 * b2 * cos(a1 * x) * cos(b2 * y) * sin(a2 * y) * sin(b1 * x) -
                   a2 * b1 * cos(a2 * y) * cos(b1 * x) * sin(a1 * x) * sin(b2 * y);
                   
      (*this)(i,j)[2] = cross * sin(k1 * x) * sin(k2 * y); 
    }
}

//////////////////////////////////////////////////////////////////////
// compute the vorticity function according to the 
// DeWitt et al. paper
//////////////////////////////////////////////////////////////////////
void VECTOR3_FIELD_2D::vorticity(int k1, int k2)
{
  int index = 0;
  Real dx = 3.14f / (_xRes - 1);
  Real dy = 3.14f / (_yRes - 1);
  for (int y = 0; y < _yRes; y++)
    for (int x = 0; x < _xRes; x++, index++)
    {
      //VEC3F center = cellCenter(x,y);

      //Real xReal = center[0];
      //Real yReal = center[1];
      Real xReal = x * dx;
      Real yReal = y * dy;

      _data[index][0] = 0;
      _data[index][1] = 0;
      _data[index][2] = sin(k1 * xReal) * sin(k2 * yReal);
    }
}

//////////////////////////////////////////////////////////////////////
// convert to and from a big vector
//////////////////////////////////////////////////////////////////////
VECTOR VECTOR3_FIELD_2D::flatten()
{
  VECTOR final(_xRes * _yRes * 3);

  int index = 0;
  int index3 = 0;
  for (int y = 0; y < _yRes; y++)
    for (int x = 0; x < _xRes; x++, index++, index3 += 3)
    {
      final[index3] = _data[index][0];
      final[index3 + 1] = _data[index][1];
      final[index3 + 2] = _data[index][2];
    }
  return final;
}

//////////////////////////////////////////////////////////////////////
// convert to and from a big vector
//////////////////////////////////////////////////////////////////////
VECTOR VECTOR3_FIELD_2D::flattenXY()
{
  VECTOR final(_xRes * _yRes * 2);

  int index = 0;
  int index2 = 0;
  for (int y = 0; y < _yRes; y++)
    for (int x = 0; x < _xRes; x++, index++, index2 += 2)
    {
      final[index2] = _data[index][0];
      final[index2 + 1] = _data[index][1];
    }
  return final;
}

//////////////////////////////////////////////////////////////////////
// convert to and from a big vector
//////////////////////////////////////////////////////////////////////
void VECTOR3_FIELD_2D::unflatten(const VECTOR& v)
{
  assert(v.size() == _xRes * _yRes * 3);

  int index = 0;
  int index3 = 0;
  for (int y = 0; y < _yRes; y++)
    for (int x = 0; x < _xRes; x++, index++, index3 += 3)
    {
      _data[index][0] = v[index3];
      _data[index][1] = v[index3 + 1];
      _data[index][2] = v[index3 + 2];
    }
}

//////////////////////////////////////////////////////////////////////
// convert to and from a big vector
//////////////////////////////////////////////////////////////////////
void VECTOR3_FIELD_2D::unflattenXY(const VECTOR& v)
{
  assert(v.size() == _xRes * _yRes * 2);

  int index = 0;
  int index2 = 0;
  for (int y = 0; y < _yRes; y++)
    for (int x = 0; x < _xRes; x++, index++, index2 += 2)
    {
      _data[index][0] = v[index2];
      _data[index][1] = v[index2 + 1];
      _data[index][2] = 0;
    }
}

//////////////////////////////////////////////////////////////////////
// get the curl of the scalar field
//////////////////////////////////////////////////////////////////////
VECTOR3_FIELD_2D VECTOR3_FIELD_2D::curl(const FIELD_2D& scalar)
{
  const int xRes = scalar.xRes();
  const int yRes = scalar.yRes();
  VECTOR3_FIELD_2D final(xRes, yRes, VEC3F(0,0,0), scalar.lengths());

  const FIELD_2D Dx = scalar.Dx();
  const FIELD_2D Dy = scalar.Dy();

  for (int y = 0; y < yRes; y++)
    for (int x = 0; x < xRes; x++)
    {
      final(x,y)[0] = Dy(x,y);
      final(x,y)[1] = -Dx(x,y);
      final(x,y)[2] = 0;
    }

  return final;
}

//////////////////////////////////////////////////////////////////////
// get a single structure coefficient
//////////////////////////////////////////////////////////////////////
Real VECTOR3_FIELD_2D::structureCoefficient(int a1, int a2, int b1, int b2, int k1, int k2, int xRes, int yRes)
{
  VEC3F center(M_PI / 2, M_PI / 2, 0);
  VEC3F lengths(M_PI, M_PI, 0);

  // i advects j
  VECTOR3_FIELD_2D iVelocity(xRes, yRes, center, lengths);
  VECTOR3_FIELD_2D jVelocity(xRes, yRes, center, lengths);

  // get the analytic eigenfunctions
  iVelocity.eigenfunctionUnscaled(a1,a2);
  jVelocity.eigenfunctionUnscaled(b1,b2);

  // the vorticity field
  VECTOR3_FIELD_2D kVorticity(xRes, yRes, center, lengths);
  kVorticity.vorticity(k1, k2);

  // advect field j using field i
  VECTOR3_FIELD_2D crossField(xRes, yRes, center, lengths);
  for (int y = 0; y < yRes; y++)
    for (int x = 0; x < xRes; x++)
    {
      MATRIX3 cross = MATRIX3::cross(iVelocity(x,y));
      crossField(x,y) = cross * jVelocity(x,y);
    }

  // dot product of vorticity and advected velocity
  FIELD_2D crossFieldZ = crossField.scalarField(2);
  crossFieldZ *= kVorticity.scalarField(2);
  Real dot = crossFieldZ.sum();

  // do some picky final scalings that arise from the domain dimensions
  const Real dxSq = (lengths[0] / xRes) * (lengths[0] / xRes);
  Real final = dot * dxSq;

  // the integral is actuall defined over [-pi, pi], not [0, pi], but it's just 4 repetitions
  // of the same function
  final *= 4.0;

  // get rid of the double pi that appears because of the domain dimensions
  final *= 1.0 / (M_PI * M_PI);

  // scale according to the eigenvalue
  final *= 1.0 / (b1 * b1 + b2 * b2);

  return final;
}

//////////////////////////////////////////////////////////////////////
// get a single structure coefficient
//////////////////////////////////////////////////////////////////////
Real VECTOR3_FIELD_2D::structureCoefficientAnalytic(int a1, int a2, int b1, int b2, int k1, int k2)
{
  int leftSign1 = 0;
  int rightSign1 = 0;
  if (a1 == b1 + k1)
  {
    leftSign1 = -1;
    rightSign1 = 1;
  }
  if (a1 + b1 == k1)
  {
    leftSign1 = 1;
    rightSign1 = 1;
  }
  if (a1 + k1 == b1)
  {
    leftSign1 = 1;
    rightSign1 = -1;
  }

  int leftSign2 = 0;
  int rightSign2 = 0;
  if (a2 == b2 + k2)
  {
    leftSign2 = 1;
    rightSign2 = -1;
  }
  if (a2 + b2 == k2)
  {
    leftSign2 = 1;
    rightSign2 = 1;
  }
  if (a2 + k2 == b2)
  {
    leftSign2 = -1;
    rightSign2 = 1;
  }

  if (leftSign1 * rightSign1 * leftSign2 * rightSign2 == 0)
    return 0;

  /*
  cout << __FILE__ << " " << __FUNCTION__ << " " << __LINE__ << " : " << endl;
  cout << " b2 * a1: " << b2 * a1 << " b1 * a2: " << b1 * a2 << endl;
  cout << " signs: " << leftSign1 << " " << leftSign2 << " " << rightSign1 << " " << rightSign2 << endl;
  cout << " final: " << (b2 * a1 - b1 * a2) << endl;
  cout << " final: " << (Real)(b2 * a1 - b1 * a2) / (b1 * b1 + b2 * b2) << endl;
  cout << " final: " << (Real)(b2 * a1 * leftSign1 * leftSign2 - b1 * a2 * rightSign1 * rightSign2) / (b1 * b1 + b2 * b2) << endl;
  */
  return 0.25 * (Real)(b2 * a1 * leftSign1 * leftSign2 - b1 * a2 * rightSign1 * rightSign2) / (b1 * b1 + b2 * b2);
}

//////////////////////////////////////////////////////////////////////
// get the summed square of all the entries
//////////////////////////////////////////////////////////////////////
Real VECTOR3_FIELD_2D::sumSq()
{
  Real final = 0;

  for (int x = 0; x < _totalCells; x++)
    final += norm(_data[x]);

  return final;
}

//////////////////////////////////////////////////////////////////////
// compute the Laplacian Eigenfunction according to the 
// DeWitt et al. paper
//////////////////////////////////////////////////////////////////////
void VECTOR3_FIELD_2D::eigenfunction(int k1, int k2)
{
  Real invKs = 1.0 / (k1 * k1 + k2 * k2);
  int index = 0;

  //Real dx = 3.14f / (_xRes - 1);
  //Real dy = 3.14f / (_yRes - 1);
  Real dx = M_PI / (_xRes - 1);
  Real dy = M_PI / (_yRes - 1);

  //for (int x = 0; x < _xRes; x++)
  //  for (int y = 0; y < _yRes; y++, index++)
  for (int y = 0; y < _yRes; y++)
    for (int x = 0; x < _xRes; x++, index++)
    {
      Real xReal = x * dx;
      Real yReal = y * dy;

      //_data[index][0] =  k2 * sin(k1 * xReal) * cos(k2 * (yReal + 0.5 * dy));
      //_data[index][1] = -k1 * cos(k1 * (xReal + 0.5 * dx)) * sin(k2 * yReal);
      
      // staggered grid
      //_data[index][0] =  invKs * k2 * sin(k1 * xReal) * cos(k2 * (yReal + 0.5 * dy));
      //_data[index][1] = -invKs * k1 * cos(k1 * (xReal + 0.5 * dx)) * sin(k2 * yReal);

      // centered grid
      _data[index][0] =  invKs * k2 * sin(k1 * xReal) * cos(k2 * yReal);
      _data[index][1] = -invKs * k1 * cos(k1 * xReal) * sin(k2 * yReal);
      
      // dummy with just a single trig per component
      //_data[index][0] =  invKs * k2 * cos(k2 * yReal);
      //_data[index][1] = -invKs * k1 * sin(k2 * yReal);
      _data[index][2] = 0;
    }
}

//////////////////////////////////////////////////////////////////////
// compute the Laplacian Eigenfunction according to the 
// DeWitt et al. paper
//
// Space out the points in x and y in a way that is friendly with
// FFTW, i.e. upon transformation will hit certain frequencies
// dead on
//////////////////////////////////////////////////////////////////////
void VECTOR3_FIELD_2D::eigenfunctionFFTW(int k1, int k2)
{
  Real invKs = 1.0 / (k1 * k1 + k2 * k2);
  int index = 0;

  for (int y = 0; y < _yRes; y++)
    for (int x = 0; x < _xRes; x++, index++)
    {
      Real xReal, xStart, dx;
      Real yReal, yStart, dy;

      //_data[index][0] =  k2 * sin(k1 * xReal) * cos(k2 * (yReal + 0.5 * dy));
      //_data[index][1] = -k1 * cos(k1 * (xReal + 0.5 * dx)) * sin(k2 * yReal);
      
      // staggered grid
      //_data[index][0] =  invKs * k2 * sin(k1 * xReal) * cos(k2 * (yReal + 0.5 * dy));
      //_data[index][1] = -invKs * k1 * cos(k1 * (xReal + 0.5 * dx)) * sin(k2 * yReal);

      // centered grid
      //  tried optimizing this, but didn't seem to matter. Most of the time is
      //  spent in the trig call
      xStart = M_PI / (_xRes + 1);
      dx = (M_PI - 2 * xStart) / (_xRes - 1);
      xReal = xStart + x * dx;

      yStart = M_PI / (2.0 * _yRes);
      dy = (M_PI - 2 * yStart) / (_yRes - 1);
      yReal = yStart + y * dy;

      _data[index][0] =  invKs * k2 * sin(k1 * xReal) * cos(k2 * yReal);

      xStart = M_PI / (2.0 * _xRes);
      dx = (M_PI - 2 * xStart) / (_xRes - 1);
      xReal = xStart + x * dx;

      yStart = M_PI / (_yRes + 1);
      dy = (M_PI - 2 * yStart) / (_yRes - 1);
      yReal = yStart + y * dy;

      _data[index][1] = -invKs * k1 * cos(k1 * xReal) * sin(k2 * yReal);
      
      // dummy with just a single trig per component
      //_data[index][0] =  invKs * k2 * cos(k2 * yReal);
      //_data[index][1] = -invKs * k1 * sin(k2 * yReal);
      _data[index][2] = 0;
    }
}

//////////////////////////////////////////////////////////////////////
// compute the Laplacian Eigenfunction according to the 
// DeWitt et al. paper
//////////////////////////////////////////////////////////////////////
void VECTOR3_FIELD_2D::eigenfunctionUnscaled(int k1, int k2)
{
  int index = 0;

  Real dx = 3.14f / (_xRes - 1);
  Real dy = 3.14f / (_yRes - 1);

  //for (int x = 0; x < _xRes; x++)
  //  for (int y = 0; y < _yRes; y++, index++)
  for (int y = 0; y < _yRes; y++)
    for (int x = 0; x < _xRes; x++, index++)
    {
      Real xReal = x * dx;
      Real yReal = y * dy;

      // centered grid
      _data[index][0] =  k2 * sin(k1 * xReal) * cos(k2 * yReal);
      _data[index][1] = -k1 * cos(k1 * xReal) * sin(k2 * yReal);

      // staggered grid
      //_data[index][0] =  k2 * sin(k1 * xReal) * cos(k2 * (yReal + 0.5 * dy));
      //_data[index][1] = -k1 * cos(k1 * (xReal + 0.5 * dx)) * sin(k2 * yReal);
      
      //_data[index][0] =  invKs * k2 * sin(k1 * xReal) * cos(k2 * yReal);
      //_data[index][1] = -invKs * k1 * cos(k1 * xReal) * sin(k2 * yReal);
      _data[index][2] = 0;
    }
}

//////////////////////////////////////////////////////////////////////
// stomp the border to zero
//////////////////////////////////////////////////////////////////////
void VECTOR3_FIELD_2D::stompBorder()
{
  for (int x = 0; x < _xRes; x++)
  {
    (*this)(x, 0) = 0;
    (*this)(x, _yRes - 1) = 0;
  }
  for (int y = 0; y < _yRes; y++)
  {
    (*this)(0, y) = 0;
    (*this)(_xRes - 1, y) = 0;
  }
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
void VECTOR3_FIELD_2D::setX(const FIELD_2D& xField)
{
  assert(xField.xRes() == _xRes && xField.yRes() == _yRes);
  for (int x = 0; x < _totalCells; x++)
    _data[x][0] = xField[x];
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
void VECTOR3_FIELD_2D::setY(const FIELD_2D& yField)
{
  assert(yField.xRes() == _xRes && yField.yRes() == _yRes);
  for (int x = 0; x < _totalCells; x++)
    _data[x][1] = yField[x];
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
VECTOR3_FIELD_2D& VECTOR3_FIELD_2D::operator*=(const Real& alpha)
{
  for (int x = 0; x < _totalCells; x++)
    _data[x] *= alpha;
  return *this;
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
VECTOR3_FIELD_2D& VECTOR3_FIELD_2D::operator+=(const VECTOR3_FIELD_2D& input)
{
  assert(input._totalCells == _totalCells);
  for (int x = 0; x < _totalCells; x++)
    _data[x] += input._data[x];
  return *this;
}
