include ../include.mk

# calls:
CC         = g++
CFLAGS     = ${CFLAGS_COMMON} -O3
LDFLAGS    = ${LDFLAGS_COMMON}
EXECUTABLE = ../../bin/vectorFieldWriter

SOURCES    = vectorFieldWriter.cpp \
						../../src/linearalgebra/VECTOR.cpp \
						../../src/linearalgebra/MATRIX3.cpp \
						../../src/geometry/FIELD_2D.cpp \
						../../src/geometry/VECTOR3_FIELD_3D.cpp \
						../../src/geometry/VECTOR3_FIELD_2D.cpp \
						 ../../src/util/TIMER.cpp \
						 ../../src/util/MIN_HEAP.cpp \
						../../src/geometry/FIELD_3D.cpp
OBJECTS    = $(SOURCES:.cpp=.o)

all: $(SOURCES) $(EXECUTABLE)
	
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f *.o
