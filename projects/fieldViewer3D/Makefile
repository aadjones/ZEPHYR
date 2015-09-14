include ../include.mk

# calls:
CC         = g++
#CFLAGS     = ${CFLAGS_COMMON} -O3
#LDFLAGS    = ${LDFLAGS_COMMON}
CFLAGS     = -Wno-deprecated-declarations ${CFLAGS_COMMON} -flax-vector-conversions
LDFLAGS    = -L/opt/local/lib/gcc5/ -L/usr/local/lib/x86_64/ -fopenmp ${LDFLAGS_COMMON} 
EXECUTABLE = ../../bin/fieldViewer3D

SOURCES    = fieldViewer3D.cpp \
						../../src/linearalgebra/VECTOR.cpp \
						../../src/linearalgebra/MATRIX3.cpp \
						../../src/geometry/FIELD_2D.cpp \
						../../src/geometry/VECTOR3_FIELD_3D.cpp \
						../../src/geometry/VECTOR3_FIELD_2D.cpp \
						 ../../src/util/TIMER.cpp \
						 ../../src/util/MIN_HEAP.cpp \
						../../src/geometry/FIELD_3D.cpp
#SOURCES    = example2.cpp ../../src/geometry/FIELD_2D.cpp
OBJECTS    = $(SOURCES:.cpp=.o)

all: $(SOURCES) $(EXECUTABLE)
	
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f *.o
