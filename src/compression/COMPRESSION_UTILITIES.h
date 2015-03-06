//
//  COMPRESSION_UTILITIES.h
//  
//
//  Created by Aaron Demby Jones on 3/5/15.
//
//

#ifndef _COMPRESSION_UTILITIES_h
#define _COMPRESSION_UTILITIES_h

#include "EIGEN.h"
#include <iostream>
#include <vector>
#include <string>
#include <sys/stat.h>  // for using stat to check whether a file exists
#include "VECTOR.h"
#include "FIELD_3D.h"
#include "INTEGER_FIELD_3D.h"
#include "MATRIX.h"


using std::vector;
using std::string;

VECTOR CastToVector(double* data, int size);
   
VECTOR CastToVector(int* data, int size);
  

VECTOR CastToVector(short* data, int size);


double* CastToDouble(const VECTOR& x, double* array);


int* CastToInt(const VECTOR& x, int* array);
  
short* CastToShort(const VECTOR& x, short* array);

long* CastToLong(const VECTOR& x, long* array);
  

VECTOR CastIntToVector(const vector<int>& V);
   

vector<int> CastVectorToInt(const VECTOR& V);
    
vector<double> CastVectorToDouble(const VECTOR& V);
 
VECTOR CastIntToVector(const vector<short>& V);


VECTOR CastDoubleToVector(const vector<double>& V);


int RoundToInt(const double& x);
    


INTEGER_FIELD_3D RoundFieldToInt(const FIELD_3D& F);
  

FIELD_3D CastIntFieldToDouble(const INTEGER_FIELD_3D& F);


vector<int> ModifiedCumSum(vector<int>& V);
   
VECTOR ModifiedCumSum(const VECTOR& V);

MATRIX ModifiedCumSum(const MATRIX& M);

void DeleteIfExists(const char* filename);

void PrefixBinary(string prefix, string filename, string newFile);


void CleanUpPrefix(const char* prefix, const char* filename);
   


#endif
