#include "FIELD_2D.h"
#include <jpeglib.h>
#include <png.h>
#include <cassert>

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
FIELD_2D::FIELD_2D(const int& rows, const int& cols, const VEC3F& lengths) :
  _xRes(rows), _yRes(cols), _lengths(lengths)
{
  _totalCells = _xRes * _yRes;
  _data = new Real[_totalCells];

  _dx = _lengths[0] / _xRes;
  _dy = _lengths[1] / _yRes;

  for (int x = 0; x < _totalCells; x++)
    _data[x] = 0.0;
}

FIELD_2D::FIELD_2D(const FIELD_2D& m) :
  _xRes(m.xRes()), _yRes(m.yRes())
{
  _totalCells = _xRes * _yRes;
  _data = new Real[_totalCells];
  _lengths = m._lengths;
  _dx = m._dx;
  _dy = m._dy;

  for (int x = 0; x < _totalCells; x++)
    _data[x] = m[x];
}

FIELD_2D::FIELD_2D() :
  _xRes(0), _yRes(0), _lengths(1,1,0), _totalCells(0), _data(NULL)
{
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
FIELD_2D::~FIELD_2D()
{
  delete[] _data;
}
  
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
void FIELD_2D::clear()
{
  for (int x = 0; x < _totalCells; x++)
    _data[x] = 0.0;
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
void FIELD_2D::write(string filename) const
{
  FILE* file;
  file = fopen(filename.c_str(), "wb");
  if (file == NULL)
  {
    cout << __FILE__ << " " << __FUNCTION__ << " " << __LINE__ << " : " << endl;
    cout << " FIELD_2D write failed! " << endl;
    cout << " Could not open file " << filename.c_str() << endl;
    exit(0);
  }

  // write dimensions
  fwrite((void*)&_xRes, sizeof(int), 1, file);
  fwrite((void*)&_yRes, sizeof(int), 1, file);

  // always write out as a double
  if (sizeof(Real) != sizeof(double))
  {
    double* dataDouble = new double[_totalCells];
    for (int x = 0; x < _totalCells; x++)
      dataDouble[x] = _data[x];

    fwrite((void*)dataDouble, sizeof(double), _totalCells, file);
    delete[] dataDouble;
    fclose(file);
  }
  else
    fwrite((void*)_data, sizeof(Real), _totalCells, file);
  fclose(file);
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
void FIELD_2D::read(string filename)
{
  FILE* file;
  file = fopen(filename.c_str(), "rb");
  if (file == NULL)
  {
    cout << __FILE__ << " " << __FUNCTION__ << " " << __LINE__ << " : " << endl;
    cout << " FIELD_2D read failed! " << endl;
    cout << " Could not open file " << filename.c_str() << endl;
    exit(0);
  }

  // read dimensions
  fread((void*)&_xRes, sizeof(int), 1, file);
  fread((void*)&_yRes, sizeof(int), 1, file);
  _totalCells = _xRes * _yRes;
  if (_data) delete[] _data;
  _data = new Real[_totalCells];

  // always read in as a double
  if (sizeof(Real) != sizeof(double))
  {
    double* dataDouble = new double[_totalCells];
    fread((void*)dataDouble, sizeof(double), _totalCells, file);

    for (int x = 0; x < _totalCells; x++)
      _data[x] = dataDouble[x];

    delete[] dataDouble;
  }
  else
    fread((void*)_data, sizeof(Real), _totalCells, file);
  fclose(file);
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
void FIELD_2D::writePPM(string filename)
{
  FILE *fp;
  unsigned char* pixels = new unsigned char[3 * _totalCells];

  for (int x = 0; x < _totalCells; x++)
  {
    pixels[3 * x] = 255 * _data[x];
    pixels[3 * x + 1] = 255 * _data[x];
    pixels[3 * x + 2] = 255 * _data[x];
  }

  fp = fopen(filename.c_str(), "wb");
  fprintf(fp, "P6\n%d %d\n255\n", _xRes, _yRes);
  fwrite(pixels, 1, _totalCells * 3, fp);
  fclose(fp);
  delete[] pixels;
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
void FIELD_2D::writeMatlab(string filename, string variableName) const
{
  FILE* file;
  file = fopen(filename.c_str(), "w");
  fprintf(file, "%s = [", variableName.c_str());
  for (int y = 0; y < _yRes; y++)
  {
    for (int x = 0; x < _xRes; x++)
      fprintf(file, "%f ", (*this)(x,y));
    fprintf(file, "; ");
  }
  fprintf(file, "];\n");

  fclose(file);
}

///////////////////////////////////////////////////////////////////////
// code based on example code from
// http://zarb.org/~gc/html/libpng.html  
///////////////////////////////////////////////////////////////////////
void FIELD_2D::readPNG(string filename)
{
  cout << " Reading in PNG file " << filename.c_str() << endl;

  int width, height;
  png_structp png_ptr;
  png_infop info_ptr;
  png_byte color_type;
  png_byte bit_depth;

  int number_of_passes;
  png_bytep* row_pointers;
  png_byte header[8];    // 8 is the maximum size that can be checked

  // open file and test for it being a png 
  FILE *fp = fopen(filename.c_str(), "rb");
  if (fp == NULL)
  {
    printf("[read_png_file] File %s could not be opened for reading\n", filename.c_str());
    exit(0);
  }
  fread(header, 1, 8, fp);
  if (png_sig_cmp(header, 0, 8))
    printf("[read_png_file] File %s is not recognized as a PNG file\n", filename.c_str());

  // initialize stuff
  png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

  if (!png_ptr)
    printf("[read_png_file] png_create_read_struct failed\n");

  info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr)
    printf("[read_png_file] png_create_info_struct failed\n");

  if (setjmp(png_jmpbuf(png_ptr)))
    printf("[read_png_file] Error during init_io\n");

  png_init_io(png_ptr, fp);
  png_set_sig_bytes(png_ptr, 8);
  png_read_info(png_ptr, info_ptr);

  width = png_get_image_width(png_ptr, info_ptr);
  height = png_get_image_height(png_ptr, info_ptr);
  color_type = png_get_color_type(png_ptr, info_ptr);
  bit_depth = png_get_bit_depth(png_ptr, info_ptr);

  number_of_passes = png_set_interlace_handling(png_ptr);
  png_read_update_info(png_ptr, info_ptr);

  // read file
  if (setjmp(png_jmpbuf(png_ptr)))
    printf("[read_png_file] Error during read_image\n");

  row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * height);
  for (int y = 0; y < height; y++)
    row_pointers[y] = (png_byte*) malloc(png_get_rowbytes(png_ptr,info_ptr));

  png_read_image(png_ptr, row_pointers);
  fclose(fp);

  // push the data into the member variables
  _xRes = width;
  _yRes = height;
  _totalCells = _xRes * _yRes;

  if (_data) delete[] _data;
  _data = new Real[_totalCells];

  if (color_type == PNG_COLOR_TYPE_GRAY)
  {
    cout << " PNG color type is gray" << endl;
    for (int y = 0; y < _yRes; y++)
      for (int x = 0; x < _xRes; x++)
        (*this)(x,y) = row_pointers[height - 1 - y][x] / 255.0;
  }
  else if (color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
  {
    cout << " PNG color type is gray with alpha" << endl;
    for (int y = 0; y < _yRes; y++)
      for (int x = 0; x < _xRes; x++)
        (*this)(x,y) = row_pointers[height - 1 - y][2 * x] / 255.0;
  }
  else if (color_type == PNG_COLOR_TYPE_RGB)
  {
    cout << " PNG color type is RGB" << endl;
    for (int y = 0; y < _yRes; y++)
      for (int x = 0; x < _xRes; x++)
      {
        float r = (float)row_pointers[height - 1 - y][3 * x] / 255.0;
        float g = (float)row_pointers[height - 1 - y][3 * x + 1] / 255.0;
        float b = (float)row_pointers[height - 1 - y][3 * x + 2] / 255.0;
        (*this)(x,y) = (r + g + b) / 3.0;
      }
  }
  else if (color_type == PNG_COLOR_TYPE_RGB_ALPHA)
  {
    cout << " PNG color type is RGB with alpha" << endl;
    for (int y = 0; y < _yRes; y++)
      for (int x = 0; x < _xRes; x++)
      {
        float r = (float)row_pointers[height - 1 - y][4 * x] / 255.0;
        float g = (float)row_pointers[height - 1 - y][4 * x + 1] / 255.0;
        float b = (float)row_pointers[height - 1 - y][4 * x + 2] / 255.0;
        (*this)(x,y) = (r + g + b) / 3.0;
      }
  }
  else
  {
    cout << " PNG color type " << (int)color_type << " is unsupported! " << endl;
    exit(0);
  }

  for (int y = 0; y < height; y++)
    free(row_pointers[y]);
  free(row_pointers);
}

///////////////////////////////////////////////////////////////////////
// code based on example code from
// http://zarb.org/~gc/html/libpng.html  
///////////////////////////////////////////////////////////////////////
void FIELD_2D::writePNG(string filename)
{
  cout << " Writing out PNG file " << filename.c_str() << endl;

  int width = _xRes; 
  int height = _yRes;

  // copy image data into pointers
  png_bytep* row_pointers;
  row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * height);
  for (int y = 0; y < height; y++)
    row_pointers[y] = (png_byte*) malloc(sizeof(png_byte) * width);

  for (int y = 0; y < height; y++)
    for (int x = 0; x < width; x++)
    {
      float value = (*this)(x,y) * 255;
      value = (value > 255)  ? 255 : value;
      value = (value < 0)  ? 0 : value;
      row_pointers[height - 1 - y][x] = (unsigned char)value;
    }

  png_structp png_ptr;
  png_infop info_ptr;
  png_byte color_type = PNG_COLOR_TYPE_GRAY;
  png_byte bit_depth = 8;

  // create file
  FILE *fp = fopen(filename.c_str(), "wb");
  if (fp == NULL)
    printf("[write_png_file] File %s could not be opened for writing\n", filename.c_str());

  // initialize stuff
  png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

  if (!png_ptr)
    printf("[write_png_file] png_create_write_struct failed\n");

  info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr)
    printf("[write_png_file] png_create_info_struct failed\n");

  if (setjmp(png_jmpbuf(png_ptr)))
    printf("[write_png_file] Error during init_io\n");

  png_init_io(png_ptr, fp);

  // write header
  if (setjmp(png_jmpbuf(png_ptr)))
    printf("[write_png_file] Error during writing header\n");

  png_set_IHDR(png_ptr, info_ptr, width, height,
       bit_depth, color_type, PNG_INTERLACE_NONE,
       PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

  png_write_info(png_ptr, info_ptr);

  // write bytes
  if (setjmp(png_jmpbuf(png_ptr)))
    printf("[write_png_file] Error during writing bytes\n");

  png_write_image(png_ptr, row_pointers);
  
  // end write
  if (setjmp(png_jmpbuf(png_ptr)))
    printf("[write_png_file] Error during end of write\n");

  png_write_end(png_ptr, NULL);

  // cleanup heap allocation
  for (int y=0; y<height; y++)
    free(row_pointers[y]);
  free(row_pointers);

  fclose(fp);
}

///////////////////////////////////////////////////////////////////////
// jpeglib code based on:
// http://andrewewhite.net/wordpress/2008/09/02/very-simple-jpeg-writer-in-c-c
///////////////////////////////////////////////////////////////////////
void FIELD_2D::writeJPG(string filename)
{
  cout << " Writing out JPG file " << filename.c_str() << endl;

  FILE* file = fopen(filename.c_str(), "wb");
 
  if (file == NULL)
  {
    cout << __FILE__ << " " << __FUNCTION__ << " " << __LINE__ << " : " << endl;
    cout << " Could not open file: " << filename.c_str() << endl;
    exit(0);
  }

  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr       jerr;
   
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);
  jpeg_stdio_dest(&cinfo, file);
   
  cinfo.image_width      = _xRes;
  cinfo.image_height     = _yRes;
  cinfo.input_components = 3;
  cinfo.in_color_space   = JCS_RGB;

  jpeg_set_defaults(&cinfo);
  // set the quality [0..100]
  jpeg_set_quality (&cinfo, 100, true);
  jpeg_start_compress(&cinfo, true);

  // copy data to a char buffer
  unsigned char* buffer = new unsigned char[3 * _totalCells];
  int index = 0;
  for (int y = 0; y < _yRes; y++)
    for (int x = 0; x < _xRes; x++, index++)
    {
      float entry = (*this)(x, _yRes - 1 - y);
      entry = (entry < 0.0) ? 0.0 : entry;
      entry = (entry > 1.0) ? 1.0 : entry;

      buffer[3 * index] = (unsigned char) (255 * entry);
      buffer[3 * index + 1] = (unsigned char) (255 * entry);
      buffer[3 * index + 2] = (unsigned char) (255 * entry);
    }

  JSAMPROW row_pointer;
 
  while (cinfo.next_scanline < cinfo.image_height) {
    int index = cinfo.next_scanline * 3 * _xRes;
    row_pointer = (JSAMPROW)&buffer[index];
    jpeg_write_scanlines(&cinfo, &row_pointer, 1);
  }
  jpeg_finish_compress(&cinfo);

  delete[] buffer;

  fclose(file);
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
void FIELD_2D::normalize()
{
  Real maxFound = 0.0;
  Real minFound = _data[0];
  for (int x = 0; x < _totalCells; x++)
  {
    maxFound = (_data[x] > maxFound) ? _data[x] : maxFound;
    minFound = (_data[x] < minFound) ? _data[x] : minFound;
  }

  Real range = 1.0 / (maxFound - minFound);
  for (int x = 0; x < _totalCells; x++)
    _data[x] = (_data[x] - minFound) * range;
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
FIELD_2D& FIELD_2D::abs()
{
  for (int x = 0; x < _totalCells; x++)
    _data[x] = fabs(_data[x]);

  return *this;
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
void FIELD_2D::resizeAndWipe(int xRes, int yRes)
{
  if (_xRes == xRes && _yRes == yRes)
  {
    clear();
    return;
  }

  if (_data)
    delete[] _data;

  _xRes = xRes;
  _yRes = yRes;
  _totalCells = _xRes * _yRes;

  _data = new Real[_xRes * _yRes];

  for (int x = 0; x < _totalCells; x++)
    _data[x] = 0;
  
  _dx = _lengths[0] / _xRes;
  _dy = _lengths[1] / _yRes;
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
FIELD_2D& FIELD_2D::operator=(const Real& alpha)
{
  for (int x = 0; x < _totalCells; x++)
    _data[x] = alpha;

  return *this;
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
FIELD_2D& FIELD_2D::operator*=(const Real& alpha)
{
  for (int x = 0; x < _totalCells; x++)
    _data[x] *= alpha;

  return *this;
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
FIELD_2D& FIELD_2D::operator/=(const Real& alpha)
{
  for (int x = 0; x < _totalCells; x++)
    _data[x] /= alpha;

  return *this;
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
FIELD_2D& FIELD_2D::operator+=(const Real& alpha)
{
  for (int x = 0; x < _totalCells; x++)
    _data[x] += alpha;

  return *this;
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
FIELD_2D& FIELD_2D::operator-=(const FIELD_2D& input)
{
  assert(input.xRes() == _xRes);
  assert(input.yRes() == _yRes);
  for (int x = 0; x < _totalCells; x++)
    _data[x] -= input[x];

  return *this;
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
FIELD_2D& FIELD_2D::operator+=(const FIELD_2D& input)
{
  assert(input.xRes() == _xRes);
  assert(input.yRes() == _yRes);
  for (int x = 0; x < _totalCells; x++)
    _data[x] += input[x];

  return *this;
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
FIELD_2D& FIELD_2D::operator*=(const FIELD_2D& input)
{
  assert(input.xRes() == _xRes);
  assert(input.yRes() == _yRes);

  for (int x = 0; x < _totalCells; x++)
    _data[x] *= input[x];

  return *this;
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
FIELD_2D& FIELD_2D::operator/=(const FIELD_2D& input)
{
  assert(input.xRes() == _xRes);
  assert(input.yRes() == _yRes);

  for (int x = 0; x < _totalCells; x++)
    if (fabs(input[x]) > 1e-6)
      _data[x] /= input[x];
    else
      _data[x] = 0;

  return *this;
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
FIELD_2D operator*(const FIELD_2D& A, const Real alpha)
{
  FIELD_2D final(A);
  final *= alpha;
  return final;
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
FIELD_2D operator/(const FIELD_2D& A, const Real alpha)
{
  FIELD_2D final(A);
  final /= alpha;
  return final;
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
FIELD_2D operator+(const FIELD_2D& A, const FIELD_2D& B)
{
  FIELD_2D final(A);
  final += B;
  return final;
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
FIELD_2D operator-(const FIELD_2D& A, const FIELD_2D& B)
{
  FIELD_2D final(A);
  final -= B;
  return final;
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
FIELD_2D operator+(const FIELD_2D& A, const Real alpha)
{
  FIELD_2D final(A);
  final += alpha;
  return final;
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
FIELD_2D operator*(const Real alpha, const FIELD_2D& A)
{
  return A * alpha;
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
FIELD_2D operator+(const Real alpha, const FIELD_2D& A)
{
  return A + alpha;
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
FIELD_2D& FIELD_2D::operator=(const FIELD_2D& A)
{
  resizeAndWipe(A.xRes(), A.yRes());

  for (int x = 0; x < _totalCells; x++)
    _data[x] = A[x];

  return *this;
}

///////////////////////////////////////////////////////////////////////
// sum of all entries
///////////////////////////////////////////////////////////////////////
Real FIELD_2D::sum()
{
  Real total = 0;
  for (int x = 0; x < _totalCells; x++)
    total += _data[x];

  return total;
}

///////////////////////////////////////////////////////////////////////
// take the log
///////////////////////////////////////////////////////////////////////
void FIELD_2D::log(Real base)
{
  Real scale = 1.0 / std::log(base);
  for (int x = 0; x < _totalCells; x++)
    _data[x] = std::log(_data[x]) * scale;
}

///////////////////////////////////////////////////////////////////////
// get the min of the field
///////////////////////////////////////////////////////////////////////
Real FIELD_2D::min()
{
  assert(_xRes > 0);
  assert(_yRes > 0);
  Real final = _data[0];

  for (int i = 0; i < _xRes * _yRes; i++)
    final = (_data[i] < final) ? _data[i] : final;

  return final;
}

///////////////////////////////////////////////////////////////////////
// get the max of the field
///////////////////////////////////////////////////////////////////////
Real FIELD_2D::max()
{
  assert(_xRes > 0);
  assert(_yRes > 0);
  Real final = _data[0];

  for (int i = 0; i < _xRes * _yRes; i++)
    final = (_data[i] > final) ? _data[i] : final;

  return final;
}

///////////////////////////////////////////////////////////////////////
// set to a checkboard for debugging
///////////////////////////////////////////////////////////////////////
void FIELD_2D::setToCheckerboard(int xChecks, int yChecks)
{
  for (int x = 0; x < _xRes; x++)
    for (int y = 0; y < _yRes; y++)
    {
      int xMod = (x / (_xRes / xChecks)) % 2;
      int yMod = (y / (_yRes / yChecks)) % 2;

      if ((xMod && yMod) || (!xMod && !yMod))
        (*this)(x,y) = 1;
    }
}

///////////////////////////////////////////////////////////////////////
// pass a field to fieldViewer2D
///////////////////////////////////////////////////////////////////////
void FIELD_2D::fieldViewer(const FIELD_2D& field, string name)
{
  field.write("temp.field");
  string execute("./bin/fieldViewer temp.field \"");
  execute = execute + name + string("\" &");
  cout << " Executing " << execute.c_str() << endl;
  system(execute.c_str());
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
Real FIELD_2D::Dx(const int x, const int y) const
{
  assert(x >= 0);
  assert(x < _xRes);
  assert(y >= 0);
  assert(y < _yRes);
  int index = x + y * _xRes;
  const Real right = (x < _xRes - 1) ? _data[index + 1] : _data[index];
  const Real left  = (x > 0)         ? _data[index - 1] : _data[index];
  const Real denom = (x > 0 && x < _xRes -1) ? 1.0 / (2.0 * _dx) : 1.0 / _dx;
  return (right - left) * denom;
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
Real FIELD_2D::Dy(const int x, const int y) const
{
  assert(x >= 0);
  assert(x < _xRes);
  assert(y >= 0);
  assert(y < _yRes);

  int index = x + y * _xRes;

  const Real up   = (y < _yRes - 1) ? _data[index + _xRes] : _data[index];
  const Real down = (y > 0)         ? _data[index - _xRes] : _data[index];
  const Real denom = (y > 0 && y < _yRes -1) ? 1.0 / (2.0 * _dy) : 1.0 / _dy;
  return (up - down) * denom;
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
FIELD_2D FIELD_2D::Dx() const
{
  FIELD_2D final(_xRes, _yRes, _lengths);

  for (int y = 0; y < _yRes; y++)
    for (int x = 0; x < _xRes; x++)
      final(x,y) = Dx(x,y);

  return final;
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
FIELD_2D FIELD_2D::Dy() const
{
  FIELD_2D final(_xRes, _yRes, _lengths);

  for (int y = 0; y < _yRes; y++)
    for (int x = 0; x < _xRes; x++)
      final(x,y) = Dy(x,y);

  return final;
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
FIELD_2D FIELD_2D::Dz() const
{
  FIELD_2D final(_xRes, _yRes, _lengths);

  // for a 2D field, the z derivative is always zero

  return final;
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
VEC3F FIELD_2D::maxIndex()
{
  Real maxFound = _data[0];

  VEC3F maxFoundIndex;
  int index = 0;
  for (int y = 0; y < _yRes; y++)
    for (int x = 0; x < _xRes; x++, index++)
      if (_data[index] > maxFound)
      {
        maxFound = _data[index];

        maxFoundIndex[0] = x;
        maxFoundIndex[1] = y;
      }

  return maxFoundIndex;
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
VEC3F FIELD_2D::minIndex()
{
  Real minFound = _data[0];

  VEC3F minFoundIndex;
  int index = 0;
  for (int y = 0; y < _yRes; y++)
    for (int x = 0; x < _xRes; x++, index++)
      if (_data[index] < minFound)
      {
        minFound = _data[index];

        minFoundIndex[0] = x;
        minFoundIndex[1] = y;
      }

  return minFoundIndex;
}

///////////////////////////////////////////////////////////////////////
// flip the X and Y of the current field
///////////////////////////////////////////////////////////////////////
FIELD_2D FIELD_2D::flipXY()
{
  FIELD_2D final(*this);

  for (int y = 0; y < _yRes; y++)
    for (int x = 0; x < _xRes; x++)
      final(x,y) = (*this)(y,x);

  return final;
}

///////////////////////////////////////////////////////////////////////
// take the DCT in the X direction
///////////////////////////////////////////////////////////////////////
void FIELD_2D::xDCT()
{
#if USING_FFTW
  assert(sizeof(Real) == sizeof(double));
  int n[] = {_xRes};
  int howmany = _yRes;
  int flags = FFTW_ESTIMATE;
  fftw_r2r_kind kind =  FFTW_REDFT10;

  fftw_plan plan;
  plan = fftw_plan_many_r2r(
      1, n, howmany,
      _data, NULL,
      1, _xRes, 
      _data, NULL,
      1, _xRes,
      &kind, flags);
  fftw_execute(plan);
  (*this) *= 1.0 / (2.0 * sqrt(_xRes / 2.0));
  //int rank, const int *n, int howmany,
  //double *in, const int *inembed,
  //int istride, int idist,
  //double *out, const int *onembed,
  //int ostride, int odist,
  //const fftw_r2r_kind *kind, unsigned flags
#endif
}

///////////////////////////////////////////////////////////////////////
// take the DCT in the Y direction
///////////////////////////////////////////////////////////////////////
void FIELD_2D::yDCT()
{
#if USING_FFTW
  assert(sizeof(Real) == sizeof(double));
  int n[] = {_xRes};
  int howmany = _yRes;
  int flags = FFTW_ESTIMATE;
  fftw_r2r_kind kind =  FFTW_REDFT10;

  fftw_plan plan;
  plan = fftw_plan_many_r2r(
      1, n, howmany,
      _data, NULL,
      _xRes, 1,
      _data, NULL,
      _xRes, 1,
      &kind, flags);
  fftw_execute(plan);
  (*this) *= 1.0 / (2.0 * sqrt(_xRes / 2.0));
  //int rank, const int *n, int howmany,
  //double *in, const int *inembed,
  //int istride, int idist,
  //double *out, const int *onembed,
  //int ostride, int odist,
  //const fftw_r2r_kind *kind, unsigned flags
#endif
}

#if USING_FFTW
///////////////////////////////////////////////////////////////////////
// take the IDCT in the X direction
///////////////////////////////////////////////////////////////////////
void FIELD_2D::xPlanIDCT(fftw_plan& plan)
{
  assert(sizeof(Real) == sizeof(double));
  int n[] = {_xRes};
  int howmany = _yRes;
  int flags = FFTW_EXHAUSTIVE;
  fftw_r2r_kind kind =  FFTW_REDFT01;

  plan = fftw_plan_many_r2r(
      1, n, howmany,
      _data, NULL,
      1, _xRes,
      _data, NULL,
      1, _xRes,
      &kind, flags);
}

///////////////////////////////////////////////////////////////////////
// take the IDCT in the X direction
///////////////////////////////////////////////////////////////////////
void FIELD_2D::xIDCT(const fftw_plan& plan)
{
  fftw_execute(plan);
  (*this) *= 1.0 / (2.0 * sqrt(_xRes / 2.0));
}
#endif

///////////////////////////////////////////////////////////////////////
// take the IDCT in the X direction
///////////////////////////////////////////////////////////////////////
void FIELD_2D::xIDCT()
{
#if USING_FFTW
  assert(sizeof(Real) == sizeof(double));
  int n[] = {_xRes};
  int howmany = _yRes;
  int flags = FFTW_ESTIMATE;
  fftw_r2r_kind kind =  FFTW_REDFT01;

  fftw_plan plan;
  plan = fftw_plan_many_r2r(
      1, n, howmany,
      _data, NULL,
      1, _xRes,
      _data, NULL,
      1, _xRes,
      &kind, flags);
  fftw_execute(plan);
  (*this) *= 1.0 / (2.0 * sqrt(_xRes / 2.0));
  //int rank, const int *n, int howmany,
  //double *in, const int *inembed,
  //int istride, int idist,
  //double *out, const int *onembed,
  //int ostride, int odist,
  //const fftw_r2r_kind *kind, unsigned flags
#endif
}

#if USING_FFTW
///////////////////////////////////////////////////////////////////////
// take the IDCT in the Y direction
///////////////////////////////////////////////////////////////////////
void FIELD_2D::yIDCT(const fftw_plan& plan)
{
  fftw_execute(plan);
  (*this) *= 1.0 / (2.0 * sqrt(_xRes / 2.0));
}

///////////////////////////////////////////////////////////////////////
// take the IDCT in the Y direction
///////////////////////////////////////////////////////////////////////
void FIELD_2D::yPlanIDCT(fftw_plan& plan)
{
  assert(sizeof(Real) == sizeof(double));
  int n[] = {_xRes};
  int howmany = _yRes;
  int flags = FFTW_EXHAUSTIVE;
  fftw_r2r_kind kind =  FFTW_REDFT01;

  plan = fftw_plan_many_r2r(
      1, n, howmany,
      _data, NULL,
      _xRes, 1,
      _data, NULL,
      _xRes, 1,
      &kind, flags);
}
#endif

///////////////////////////////////////////////////////////////////////
// take the IDCT in the Y direction
///////////////////////////////////////////////////////////////////////
void FIELD_2D::yIDCT()
{
#if USING_FFTW
  assert(sizeof(Real) == sizeof(double));
  int n[] = {_xRes};
  int howmany = _yRes;
  int flags = FFTW_ESTIMATE;
  fftw_r2r_kind kind =  FFTW_REDFT01;

  fftw_plan plan;
  plan = fftw_plan_many_r2r(
      1, n, howmany,
      _data, NULL,
      _xRes, 1,
      _data, NULL,
      _xRes, 1,
      &kind, flags);
  fftw_execute(plan);
  (*this) *= 1.0 / (2.0 * sqrt(_xRes / 2.0));
  //int rank, const int *n, int howmany,
  //double *in, const int *inembed,
  //int istride, int idist,
  //double *out, const int *onembed,
  //int ostride, int odist,
  //const fftw_r2r_kind *kind, unsigned flags
#endif
}

///////////////////////////////////////////////////////////////////////
// take the DST in the Y direction
///////////////////////////////////////////////////////////////////////
void FIELD_2D::yDST()
{
#if USING_FFTW
  assert(sizeof(Real) == sizeof(double));
  int n[] = {_xRes};
  int howmany = _yRes;
  int flags = FFTW_ESTIMATE;
  fftw_r2r_kind kind =  FFTW_RODFT00;

  fftw_plan plan;
  plan = fftw_plan_many_r2r(
      1, n, howmany,
      _data, NULL,
      _xRes, 1,
      _data, NULL,
      _xRes, 1,
      &kind, flags);
  fftw_execute(plan);
  (*this) *= 0.5;


  //int rank, const int *n, int howmany,
  //double *in, const int *inembed,
  //int istride, int idist,
  //double *out, const int *onembed,
  //int ostride, int odist,
  //const fftw_r2r_kind *kind, unsigned flags
#endif
}

///////////////////////////////////////////////////////////////////////
// take the DST in the X direction
///////////////////////////////////////////////////////////////////////
void FIELD_2D::xDST()
{
#if USING_FFTW
  assert(sizeof(Real) == sizeof(double));
  int n[] = {_xRes};
  int howmany = _yRes;
  int flags = FFTW_ESTIMATE;
  fftw_r2r_kind kind =  FFTW_RODFT00;

  fftw_plan plan;
  plan = fftw_plan_many_r2r(
      1, n, howmany,
      _data, NULL,
      1, _xRes,
      _data, NULL,
      1, _xRes,
      &kind, flags);
  fftw_execute(plan);
  (*this) *= 0.5;


  //int rank, const int *n, int howmany,
  //double *in, const int *inembed,
  //int istride, int idist,
  //double *out, const int *onembed,
  //int ostride, int odist,
  //const fftw_r2r_kind *kind, unsigned flags
#endif
}

#if USING_FFTW
///////////////////////////////////////////////////////////////////////
// take the IDST in the Y direction
///////////////////////////////////////////////////////////////////////
void FIELD_2D::yIDST(const fftw_plan& plan)
{
  fftw_execute(plan);
  (*this) *= 1.0 / (_xRes + 1);
}

///////////////////////////////////////////////////////////////////////
// take the IDST in the Y direction
///////////////////////////////////////////////////////////////////////
void FIELD_2D::yPlanIDST(fftw_plan& plan)
{
  assert(sizeof(Real) == sizeof(double));
  int n[] = {_xRes};
  int howmany = _yRes;
  int flags = FFTW_EXHAUSTIVE;
  fftw_r2r_kind kind =  FFTW_RODFT00;

  plan = fftw_plan_many_r2r(
      1, n, howmany,
      _data, NULL,
      _xRes, 1,
      _data, NULL,
      _xRes, 1,
      &kind, flags);
}
#endif

///////////////////////////////////////////////////////////////////////
// take the IDST in the Y direction
///////////////////////////////////////////////////////////////////////
void FIELD_2D::yIDST()
{
#if USING_FFTW
  assert(sizeof(Real) == sizeof(double));
  int n[] = {_xRes};
  int howmany = _yRes;
  int flags = FFTW_ESTIMATE;
  fftw_r2r_kind kind =  FFTW_RODFT00;

  fftw_plan plan;
  plan = fftw_plan_many_r2r(
      1, n, howmany,
      _data, NULL,
      _xRes, 1,
      _data, NULL,
      _xRes, 1,
      &kind, flags);
  fftw_execute(plan);
  (*this) *= 1.0 / (_xRes + 1);

  //int rank, const int *n, int howmany,
  //double *in, const int *inembed,
  //int istride, int idist,
  //double *out, const int *onembed,
  //int ostride, int odist,
  //const fftw_r2r_kind *kind, unsigned flags
#endif
}

#if USING_FFTW
///////////////////////////////////////////////////////////////////////
// take the IDST in the X direction
///////////////////////////////////////////////////////////////////////
void FIELD_2D::xIDST(const fftw_plan& plan)
{
  fftw_execute(plan);
  (*this) *= 1.0 / (_xRes + 1);
}

///////////////////////////////////////////////////////////////////////
// take the IDST in the X direction
///////////////////////////////////////////////////////////////////////
void FIELD_2D::xPlanIDST(fftw_plan& plan)
{
  assert(sizeof(Real) == sizeof(double));
  int n[] = {_xRes};
  int howmany = _yRes;
  int flags = FFTW_EXHAUSTIVE;
  fftw_r2r_kind kind =  FFTW_RODFT00;

  plan = fftw_plan_many_r2r(
      1, n, howmany,
      _data, NULL,
      1, _xRes,
      _data, NULL,
      1, _xRes,
      &kind, flags);
}
#endif

///////////////////////////////////////////////////////////////////////
// take the IDST in the X direction
///////////////////////////////////////////////////////////////////////
void FIELD_2D::xIDST()
{
#if USING_FFTW
  assert(sizeof(Real) == sizeof(double));
  int n[] = {_xRes};
  int howmany = _yRes;
  int flags = FFTW_ESTIMATE;
  fftw_r2r_kind kind =  FFTW_RODFT00;

  fftw_plan plan;
  plan = fftw_plan_many_r2r(
      1, n, howmany,
      _data, NULL,
      1, _xRes,
      _data, NULL,
      1, _xRes,
      &kind, flags);
  fftw_execute(plan);
  (*this) *= 1.0 / (_xRes + 1);

  //int rank, const int *n, int howmany,
  //double *in, const int *inembed,
  //int istride, int idist,
  //double *out, const int *onembed,
  //int ostride, int odist,
  //const fftw_r2r_kind *kind, unsigned flags
#endif
}

#if USING_FFTW
///////////////////////////////////////////////////////////////////////
// take the IDST in the X direction
///////////////////////////////////////////////////////////////////////
void FIELD_2D::xPlanIDSTyIDCT(fftw_plan& plan)
{
  assert(sizeof(Real) == sizeof(double));
  int flags = FFTW_EXHAUSTIVE;

  plan = fftw_plan_r2r_2d(_xRes, _yRes, _data, _data,
                          FFTW_REDFT01, FFTW_RODFT00,
                          flags);
}

///////////////////////////////////////////////////////////////////////
// take the IDST in the X direction
///////////////////////////////////////////////////////////////////////
void FIELD_2D::xIDSTyIDCT(const fftw_plan& plan)
{
  fftw_execute(plan);
  (*this) *= 1.0 / ((_xRes + 1) * (2.0 * sqrt(_xRes / 2.0)));
}
#endif

///////////////////////////////////////////////////////////////////////
// take the IDST in the X direction
///////////////////////////////////////////////////////////////////////
void FIELD_2D::xIDSTyIDCT()
{
#if USING_FFTW
  assert(sizeof(Real) == sizeof(double));
  int flags = FFTW_ESTIMATE;

  fftw_plan plan;
  plan = fftw_plan_r2r_2d(_xRes, _yRes, _data, _data,
                          FFTW_REDFT01, FFTW_RODFT00,
                          flags);

  fftw_execute(plan);
  (*this) *= 1.0 / ((_xRes + 1) * (2.0 * sqrt(_xRes / 2.0)));
#endif
}


#if USING_FFTW
///////////////////////////////////////////////////////////////////////
// take the IDST in the X direction
///////////////////////////////////////////////////////////////////////
void FIELD_2D::xPlanIDCTyIDST(fftw_plan& plan)
{
  assert(sizeof(Real) == sizeof(double));
  int flags = FFTW_EXHAUSTIVE;

  plan = fftw_plan_r2r_2d(_xRes, _yRes, _data, _data,
                          FFTW_RODFT00, FFTW_REDFT01, 
                          flags);
}

///////////////////////////////////////////////////////////////////////
// take the IDST in the X direction
///////////////////////////////////////////////////////////////////////
void FIELD_2D::xIDCTyIDST(const fftw_plan& plan)
{
  fftw_execute(plan);
  (*this) *= 1.0 / ((_xRes + 1) * (2.0 * sqrt(_xRes / 2.0)));
}
#endif

///////////////////////////////////////////////////////////////////////
// take the IDST in the X direction
///////////////////////////////////////////////////////////////////////
void FIELD_2D::xIDCTyIDST()
{
#if USING_FFTW
  assert(sizeof(Real) == sizeof(double));
  int flags = FFTW_ESTIMATE;

  fftw_plan plan;
  plan = fftw_plan_r2r_2d(_xRes, _yRes, _data, _data,
                          FFTW_RODFT00, FFTW_REDFT01, 
                          flags);

  fftw_execute(plan);
  (*this) *= 1.0 / ((_xRes + 1) * (2.0 * sqrt(_xRes / 2.0)));
#endif
}

///////////////////////////////////////////////////////////////////////
// max absolute index
///////////////////////////////////////////////////////////////////////
int FIELD_2D::maxAbsIndex()
{
  int final = 0;
  Real bestFound = fabs(_data[0]);
  for (int x = 1; x < _totalCells; x++)
  {
    if (fabs(_data[x]) > bestFound)
    {
      final = x;
      bestFound = fabs(_data[x]);
    }
  }

  return final;
}
