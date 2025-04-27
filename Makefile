# Compiler and flags
CXX = g++
CXXFLAGS = -Wall -g -O0 --std=c++17 -fno-common

# Directories
VENDOR_DIR = vendor
INCLUDE_DIRS = -I$(VENDOR_DIR)/cadmium_v2/include \
               -I$(VENDOR_DIR)/googletest/googletest/include \
               -I$(VENDOR_DIR)/googletest/googlemock/include \
               -I$(VENDOR_DIR)/json/include \
			   -I$(VENDOR_DIR)/matplotplusplus/source \
			   -I$(VENDOR_DIR)/matplotplusplus/build/source/matplot \
               -I$(VENDOR_DIR)/eigen

LIB_DIRS = -L$(VENDOR_DIR)/googletest/lib 
GTEST_LIBS = $(VENDOR_DIR)/googletest/lib/libgtest.a \
             $(VENDOR_DIR)/googletest/lib/libgtest_main.a

MATPLOT_LIBS = $(VENDOR_DIR)/matplotplusplus/build/source/matplot/libmatplot.a $(VENDOR_DIR)/matplotplusplus/build/source/3rd_party/libnodesoup.a

BIN_DIR = bin
SRC_DIR = models
UTILS = utils

MAIN_DEPENDENCIES = 

# Test targets
TESTS = test_model_builder

# Build and run all tests
all: $(TESTS) run_tests

build_test_model_builder:
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) $(MAIN_DEPENDENCIES) $(SRC_DIR)/builder/test/modelBuilder_test.cpp $(LIB_DIRS) $(GTEST_LIBS) -o $(BIN_DIR)/test_model_builder

build_simulation:
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) $(MAIN_DEPENDENCIES) main.cpp -fsanitize=address -fsanitize=undefined $(LIB_DIRS) $(MATPLOT_LIBS) -o $(BIN_DIR)/simulation

# Run all tests
run_tests: $(addprefix run_, $(TESTS))

run_simulation: 
	$(BIN_DIR)/simulation

run_test_model_builder:
	$(BIN_DIR)/test_model_builder

.PHONY: all $(TESTS) run_tests clean

clean:
	rm -rf $(BIN_DIR)/*

build_all: $(addprefix build_, $(TESTS)) build_simulation
