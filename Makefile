include sources.mk

PROJ_NAME = pathfinder
BUILD_TYPE = DEBUG
TEST_MODE = OFF

SRC_DIR = ./src
INC_DIR = ./inc
BUILD_DIR = ./build
OBJ_DIR = ./build/obj
BIN_DIR = ./build/bin

OBJECTS = $(SOURCES:%.cu=$(OBJ_DIR)/%.obj)

ifeq ($(BUILD_TYPE), DEBUG)
PP_DEFINE = DEBUG
else ifeq ($(BUILD_TYPE), RELEASE)
PP_DEFINE = NDEBUG
else
$(error Build type undefined. Possible types: DEBUG, RELEASE)
endif

ifeq ($(TEST_MODE), ON)
PP_DEFINE += TEST
else ifeq ($(BUILD_TYPE), OFF)
PP_DEFINE += NTEST
else
$(error Test mode undefined. Possible types: ON, OFF)
endif

NVCC = nvcc
CFLAGS = -arch=sm_75 -I $(INC_DIR)
PPFLAGS = $(PP_DEFINE:%=-D %)

# Build object files
$(OBJ_DIR)/%.obj: $(SRC_DIR)/%.cu
	$(NVCC) -dc $< $(PPFLAGS) $(CFLAGS) -o $@

# Build executable
PHONY: build
build: $(OBJECTS)
	$(NVCC) $^ $(CFLAGS) -o $(BIN_DIR)/$(PROJ_NAME)

PHONY: clean
clean:
	del /Q .\build\bin\* .\build\obj\* > null