#ifndef __PATHFINDER_CUH__
#define __PATHFINDER_CUH__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "common.cuh"

/* Define -------------------------------------------------------------------*/

#define START_VAL ((int) 1)

/* Structures ---------------------------------------------------------------*/

typedef struct {
	unsigned int row;
	unsigned int col;
} point2d;

typedef struct {
	int* d_field_a;
	int* d_field_b;
	size_t field_size;
	point2d* d_start;
	point2d* d_finish;
} pathfinder_param;

/* Function Definitions -----------------------------------------------------*/

__host__ bool pathfinder_check_target_points(int* d_field,
	                                         const size_t field_size,
	                                         point2d* h_start,
	                                         point2d* h_finish);

__host__ cudaError_t pathfinder_set_start_val(int* d_field, const size_t field_size, point2d* h_start);

__host__ bool pathfinder_exec(pathfinder_param* parameters);

__global__ void pathfinder_main_kernel(int* d_field_prev,
	                                   int* d_field_next,
	                                   const size_t field_size,
	                                   point2d* d_start,
	                                   point2d* d_finish,
	                                   unsigned int* d_term_flag);

__device__ unsigned int pathfinder_mark_neigh(point2d* d_cur_point,
	                                          int d_mark,
	                                          int* d_field_prev,
	                                          int* d_field_next,
	                                          unsigned int field_size);

__host__ void pathfinder_backtrace(int* h_field, unsigned int field_size, point2d* h_start, point2d* h_finish);

#endif /* __PATHFINDER_CUH__ */