#ifndef __GENERATOR_CUH__
#define __GENERATOR_CUH__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "common.cuh"

/* Define -------------------------------------------------------------------*/

#define CONTOUR_SIZE_FACTOR ((size_t) 5)
#define MIN_CONTOUR_SIZE    ((unsigned int) 1)
#define MAX_CONTOUR_NUM     ((size_t) 20)

/* Structures ---------------------------------------------------------------*/

typedef struct {
	unsigned int x;
	unsigned int y;
	unsigned int size;
} contour_instance;

typedef struct {
	int* d_field;
	size_t field_size;
	contour_instance* d_contour_list;
	size_t contour_list_size;
} generator_param;

/* Function Definitions -----------------------------------------------------*/

__host__ void generator_new_contour(const size_t field_size, contour_instance* contour);

__host__ cudaError_t generator_exec(generator_param* parameters);

__global__ void generator_main_kernel(int* d_field,
	                                  const size_t field_size,
	                                  contour_instance* d_contour_list,
	                                  const size_t contour_list_size);

#endif /* __GENERATOR_CUH__ */
