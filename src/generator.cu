#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

#include "generator.cuh"

extern unsigned int BLOCK_DIM;

__host__ void generator_new_contour(const size_t field_size, contour_instance* contour) {
	if ((contour == NULL) || (field_size < 1)) {
		return;
	}

	static size_t contour_max_size = field_size / CONTOUR_SIZE_FACTOR;
	size_t contour_size = MIN_CONTOUR_SIZE + (unsigned int) (rand() % contour_max_size);
	size_t margin = field_size - contour_size;

	unsigned int x = (unsigned int) (rand() % margin);
	unsigned int y = (unsigned int) (rand() % margin);

	contour->size = contour_size;
	contour->x = x;
	contour->y = y;
}

__host__ cudaError_t generator_exec(generator_param* parameters) {
	cudaError_t cudaStatus;

	/* Set Grid And Block Dimensions */
	dim3 block_dim;
	block_dim.x = BLOCK_DIM;
	block_dim.y = BLOCK_DIM;

	dim3 grid_dim;
	grid_dim.x = parameters->field_size / block_dim.x + 1;
	grid_dim.y = parameters->field_size / block_dim.y + 1;

	/* Call the Main Kernel */
	generator_main_kernel<<<grid_dim, block_dim>>>(parameters->d_field,
		                                           parameters->field_size,
		                                           parameters->d_contour_list,
		                                           parameters->contour_list_size);

	cudaStatus = cudaGetLastError();
	HANDLE_ERROR(cudaStatus, "GENERATOR ERROR: generator_main_kernel launch failed\n");

	cudaStatus = cudaThreadSynchronize();
	HANDLE_ERROR(cudaStatus, "GENERATOR ERROR: failed to synchronize thread\n");

	cudaStatus = cudaDeviceSynchronize();
	HANDLE_ERROR(cudaStatus, "GENERATOR ERROR: failed to synchronize the device\n");

ERROR:
	return cudaStatus;
}

__global__ void generator_main_kernel(int* d_field,
	                                  const size_t field_size,
	                                  contour_instance* d_contour_list,
	                                  const size_t contour_list_size)
{
	unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
	contour_instance contour;

	if ((row < field_size) && (col < field_size)) {
		for (size_t i = 0; i < contour_list_size; i++) {
			contour = d_contour_list[i];
			bool inside_contour_y = (row >= contour.y) && (row < (contour.y + contour.size));
			bool inside_contour_x = (col >= contour.x) && (col < (contour.x + contour.size));
			bool inside_contour = inside_contour_y && inside_contour_x;

			if (inside_contour) {
				d_field[row * field_size + col] = BARRIER_VAL;
				break;
			}
		}
	}
}