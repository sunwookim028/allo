CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -pedantic -O2
SRC := kernel.cpp tb.cpp
HDR := kernel.h
BIN := tb

.PHONY: all run clean

all: $(BIN)

$(BIN): $(SRC) $(HDR)
	$(CXX) $(CXXFLAGS) $(SRC) -o $@

run: $(BIN)
	./$(BIN)

clean:
	rm -f $(BIN)




