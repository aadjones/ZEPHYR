#include "EIGEN.h"
#include "COMPRESSION_UTILITIES.h"
#include <iostream>
#include <vector>
#include <string>
#include <sys/stat.h>  // for using stat to check whether a file exists
#include "VECTOR.h"
#include "FIELD_3D.h"
#include "INTEGER_FIELD_3D.h"
#include "MATRIX.h"


VECTOR CastToVector(double* data, int size) {
  VECTOR x(size);
  for (int i = 0; i < size; i++) {
    x[i] = data[i];
  }
  return x;
}

VECTOR CastToVector(int* data, int size) {
  VECTOR x(size);
  for (int i = 0; i < size; i++) {
    x[i] = data[i];
  }
  return x;
}

VECTOR CastToVector(short* data, int size) {
  VECTOR x(size);
  for (int i = 0; i < size; i++) {
    x[i] = data[i];
  }
  return x;
}

double* CastToDouble(const VECTOR& x, double* array) {
  for (int i = 0; i < x.size(); i++) {
    array[i] = x[i];
  }
  return array;
}

int* CastToInt(const VECTOR& x, int* array) {
  for (int i = 0; i < x.size(); i++) {
    int data_i = (int) x[i];
    array[i] = data_i;
  }
  return array;
}

short* CastToShort(const VECTOR& x, short* array) {
  for (int i = 0; i < x.size(); i++) {
    short data_i = (short) x[i];
    array[i] = data_i;
  }
  return array;
}

long* CastToLong(const VECTOR& x, long* array) {
  for (int i = 0; i < x.size(); i++) {
    long data_i = (long) x[i];
    array[i] = data_i;
  }
  return array;
}

VECTOR CastIntToVector(const vector<int>& V) {
  int length = V.size();
  VECTOR result(length);

  for (int i = 0; i < length; i++) {
    result[i] = V[i];
  }
  return result;
}

vector<int> CastVectorToInt(const VECTOR& V) {
  int length = V.size();
  vector<int> result(length, 0);
  for (int i = 0; i < length; i++) {
    result[i] = V[i];
  }
  return result;
}

vector<double> CastVectorToDouble(const VECTOR& V) {
  int length = V.size();
  vector<double> result(length, 0.0);
  for (int i = 0; i < length; i++) {
    result[i] = V[i];
  } 
  return result;
}

VECTOR CastIntToVector(const vector<short>& V) {
  int length = V.size();
  VECTOR result(length);

  for (int i = 0; i < length; i++) {
    result[i] = V[i];
  }
  return result;
}

VECTOR CastDoubleToVector(const vector<double>& V) {
  int length = V.size();
  VECTOR result(length);

  for (int i = 0; i < length; i++) {
    result[i] = V[i];
  }
  return result;
}

int RoundToInt(const double& x) {
  int result = 0;
  if (x > 0.0) {
    result = (int) (x + 0.5);
  }
  else {
    result = (int) (x - 0.5);
  }
  return result;
}

INTEGER_FIELD_3D RoundFieldToInt(const FIELD_3D& F) {
  int xRes = F.xRes();
  int yRes = F.yRes();
  int zRes = F.zRes();

  INTEGER_FIELD_3D result(xRes, yRes, zRes);

  for (int x = 0; x < xRes; x++) {
    for (int y = 0; y < yRes; y++) {
      for (int z = 0; z < zRes; z++) {
        int rounded = RoundToInt(F(x, y, z));
        result(x, y, z) = rounded;
      }
    }
  }
  return result;
}

FIELD_3D CastIntFieldToDouble(const INTEGER_FIELD_3D& F) {
  int xRes = F.xRes();
  int yRes = F.yRes();
  int zRes = F.zRes();

  FIELD_3D result(xRes, yRes, zRes);

  for (int x = 0; x < xRes; x++) {
    for (int y = 0; y < yRes; y++) {
      for (int z = 0; z < zRes; z++) {
        double casted = (double) F(x, y, z);
        result(x, y, z) = casted;
      }
    }
  }
  return result;
}

vector<int> ModifiedCumSum(vector<int>& V) {
  // athough V is passed by reference, it will not be modified.
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

VECTOR ModifiedCumSum(const VECTOR& V) {
  vector<int> V_int = CastVectorToInt(V);
  vector<int> result_int = ModifiedCumSum(V_int);
  VECTOR result = CastIntToVector(result_int);
  return result;
}

MATRIX ModifiedCumSum(const MATRIX& M) {
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
void DeleteIfExists(const char* filename) {
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
      cout << filename << " does not yet exist." << endl;
    }
    return;
  }
void PrefixBinary(string prefix, string filename, string newFile) {
  string command = "cat " + prefix + ' ' + filename + "> " + newFile;
  const char* command_c = command.c_str();
  system(command_c);
}

void CleanUpPrefix(const char* prefix, const char* filename) {
  string prefix_string(prefix);
  string filename_string(filename);
  string command1 = "rm " + prefix_string;
  string command2 = "rm " + filename_string;
  const char* command1_c = command1.c_str();
  const char* command2_c = command2.c_str();

  system(command1_c);
  system(command2_c);
}
