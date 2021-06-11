include sources.mk

PROJ_NAME = pathfinder
BUILD_TYPE = DEBUG

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

NVCC = nvcc
CFLAGS = -arch=sm_75 -I $(INC_DIR)
PPFLAGS = -D $(PP_DEFINE)

# Build object files
$(OBJ_DIR)/%.obj: $(SRC_DIR)/%.cu
	$(NVCC) -dc $< $(PPFLAGS) $(CFLAGS) -o $@

# Build executable
PHONY: build
build: $(OBJECTS)
	$(NVCC) $^ $(CFLAGS) -o $(BIN_DIR)/$(PROJ_NAME)

PHONY: clean
clean:
	del /Q .\build\bin\* .\build\obj\*