CXX = g++
CXXFLAGS = -std=c++11 -Wall -O2 -I.
LDFLAGS = -lglfw -lGL -lassimp -ldl

SRC = main.cpp glad.c
OUT = water_waves

all: $(OUT)

$(OUT): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(OUT) $(SRC) $(LDFLAGS)

clean:
	rm -f $(OUT)

.PHONY: all clean