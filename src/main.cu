#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <errno.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "common.cuh"
#include "pathfinder.cuh"
#include "generator.cuh"

unsigned int BLOCK_DIM = 0;
unsigned int GRID_DIM = 0;

__host__ int main(int argc, const char* argv[]) {
	/*----------------------------------------------------------------------------
	 * Parameter Check & Inial Setup
	 *--------------------------------------------------------------------------*/

	/* Check Input Parameters */
	if (argc != 4) {
		fprintf(stderr, "ERROR: incorrect numer of arguments\n");
		return EXIT_FAILURE;
	}

	long field_size_l = strtol(argv[1], NULL, 10);
	if ((errno == ERANGE) ||
		(field_size_l < MIN_FIELD_SIZE) ||
		(field_size_l > MAX_FIELD_SIZE))
	{
		fprintf(stderr, "ERROR: incorrect field size parameter\n");
		return EXIT_FAILURE;
	}

	GRID_DIM = strtol(argv[3], NULL, 10);
	if ((errno == ERANGE) ||
		(BLOCK_DIM < MIN_BLOCK_DIM) ||
		(BLOCK_DIM > MAX_BLOCK_DIM))
	{
		fprintf(stderr, "ERROR: incorrect block dim parameter\n");
		return EXIT_FAILURE;
	}

	BLOCK_DIM = strtol(argv[3], NULL, 10);
	if ((errno == ERANGE) ||
		(BLOCK_DIM < MIN_BLOCK_DIM) ||
		(BLOCK_DIM > MAX_BLOCK_DIM))
	{
		fprintf(stderr, "ERROR: incorrect block dim parameter\n");
		return EXIT_FAILURE;
	}

	/* Check Device Properties */
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "ERROR: cudaSetDevice failed\n");
		return EXIT_FAILURE;
	}

	cudaDeviceProp property = {0};
	cudaGetDeviceProperties(&property, 0);
	size_t device_memory = property.totalGlobalMem;
	if (device_memory < (MAX_FIELD_SIZE * MAX_FIELD_SIZE)) {
		fprintf(stderr, "ERROR: not enough global memory\n");
		return EXIT_FAILURE;
	}

	/* Seed Rand */
	srand(time(0));

	/*----------------------------------------------------------------------------
	 * Variables & Constants Declaration
	 *--------------------------------------------------------------------------*/

	/* Common Constants */
	const size_t field_size = (size_t) field_size_l;
	const size_t field_bytes = (size_t) field_size * (size_t) field_size * sizeof(unsigned int);

	#ifndef TEST_MODE
	const size_t contour_num = (size_t) (1 + ((size_t) rand() % MAX_CONTOUR_NUM));
	#else
	const size_t contour_num = 3;
	#endif

	/* Global Host Variables */
	int* h_field = NULL;
	contour_instance* h_contour_list = NULL;
	point2d h_start = { 0 };
	point2d h_finish = { 0 };

	/* Device Variables */
	int* d_field_a = NULL;
	int* d_field_b = NULL;
	contour_instance* d_contour_list = NULL;
	point2d* d_start = NULL;
	point2d* d_finish = NULL;

	/* Generator Variables */
	generator_param gen_parameters;
	memset(&gen_parameters, 0, sizeof(generator_param));

	/* Pathfinder Variables */
	pathfinder_param pathfinder_parameters;
	memset(&pathfinder_parameters, 0, sizeof(pathfinder_param));
	bool path_exists = false;

	/* Timer Instances */
	cudaEvent_t timer_start;
	cudaEventCreate(&timer_start);
	cudaEvent_t timer_stop;
	cudaEventCreate(&timer_stop);
	float generator_time = 0.F;
	float pathfinder_time = 0.F;
	float backtrace_time = 0.F;

	/*----------------------------------------------------------------------------
	 * Memory Management
	 *--------------------------------------------------------------------------*/

	/* Host Memory */
	cudaStatus = cudaHostAlloc(&h_field, field_bytes, cudaHostAllocDefault);
	HANDLE_ERROR(cudaStatus, "Failed to allocate field on the host\n");

	cudaStatus = cudaHostAlloc(&h_contour_list, contour_num * sizeof(contour_instance), cudaHostAllocDefault);
	HANDLE_ERROR(cudaStatus, "Failed to allocate contour list\n");

	/* Device Memory */
	cudaStatus = cudaMalloc(&d_field_a, field_bytes);
	HANDLE_ERROR(cudaStatus, "Failed to allocate field on the device\n");
	cudaStatus = cudaMemset(d_field_a, 0, field_bytes);
	HANDLE_ERROR(cudaStatus, "Memset failed\n");

	cudaStatus = cudaMalloc(&d_field_b, field_bytes);
	HANDLE_ERROR(cudaStatus, "Failed to allocate field on the device\n");

	cudaStatus = cudaMalloc(&d_contour_list, contour_num * sizeof(contour_instance));
	HANDLE_ERROR(cudaStatus, "Failed to allocate contour list on the device\n");

	cudaStatus = cudaMalloc(&d_start, sizeof(point2d));
	HANDLE_ERROR(cudaStatus, "Failed to allocate start point on the device\n");

	cudaStatus = cudaMalloc(&d_finish, sizeof(point2d));
	HANDLE_ERROR(cudaStatus, "Failed to allocate finish point on the device\n");

	/*----------------------------------------------------------------------------
	 * Generate Barrier Contours
	 *--------------------------------------------------------------------------*/

	printf("----------------[LOG]----------------\n\n");
	printf("Pathfinder: Start...\n");
	printf("Pathfinder: Generating field...\n");

	h_contour_list[0].x = field_size / 2;
	h_contour_list[0].y = 0;
	h_contour_list[0].size = field_size / 2;

	h_contour_list[1].x = 0;
	h_contour_list[1].y = field_size / 4;
	h_contour_list[1].size = field_size / 4;

	h_contour_list[2].x = field_size / 4;
	h_contour_list[2].y = 3 * (field_size / 4);
	h_contour_list[2].size = field_size / 4;

	// for (size_t i = 0; i < contour_num; i++) {
	// 	generator_new_contour(field_size, &(h_contour_list[i]));
	// }

	cudaStatus = cudaMemcpy(d_contour_list,
		                    h_contour_list,
		                    contour_num * sizeof(contour_instance),
		                    cudaMemcpyDefault);
	HANDLE_ERROR(cudaStatus, "Failed to copy data Host -> Device\n");

	gen_parameters.d_field = d_field_a;
	gen_parameters.field_size = field_size;
	gen_parameters.d_contour_list = d_contour_list;
	gen_parameters.contour_list_size = contour_num;

	/* Start Timer */
	cudaEventRecord(timer_start, 0);

	/* Run Generator */
	cudaStatus = generator_exec(&gen_parameters);
	HANDLE_ERROR(cudaStatus, "Generator failed\n");

	/* Stop Timer */
	cudaEventRecord(timer_stop, 0);
	cudaEventSynchronize(timer_stop);
	cudaEventElapsedTime(&generator_time, timer_start, timer_stop);

	cudaStatus = cudaMemcpy(h_field, d_field_a, field_bytes, cudaMemcpyDefault);
	HANDLE_ERROR(cudaStatus, "Failed to copy data Device -> Host\n");

	printf("Pathfinder: Field generated\n");

	/*----------------------------------------------------------------------------
	 * Generate Start & Finish points
	 *--------------------------------------------------------------------------*/

	printf("Pathfinder: Generating start/finish points...\n");

	memset(&h_start, 0, sizeof(point2d));
	memset(&h_finish, 0, sizeof(point2d));

GEN_START_FINISH:

	h_start.row = 1;
	h_start.col = 1;

	h_finish.row = field_size - 1;
	h_finish.col = 0;

	// h_start.row = rand() % field_size;
	// h_start.col = rand() % field_size;

	// h_finish.row = rand() % field_size;
	// h_finish.col = rand() % field_size;

	if ((h_start.row == h_finish.row) && (h_start.col == h_finish.col)) {
		goto GEN_START_FINISH;
	}

	/* Check if Target Points are inside the Contour */
	if (!pathfinder_check_target_points(d_field_a, field_size, &h_start, &h_finish)) {
		goto GEN_START_FINISH;
	}

	cudaStatus = cudaMemcpy(d_start, &h_start, sizeof(point2d), cudaMemcpyDefault);
	HANDLE_ERROR(cudaStatus, "Failed to copy data Host -> Device\n");

	cudaStatus = cudaMemcpy(d_finish, &h_finish, sizeof(point2d), cudaMemcpyDefault);
	HANDLE_ERROR(cudaStatus, "Failed to copy data Host -> Device\n");

	printf("Pathfinder: Start/finish points generated\n");

	/*----------------------------------------------------------------------------
	 * Find Path on the Field
	 *--------------------------------------------------------------------------*/

	printf("Pathfinder: Scanning the field...\n");

	cudaStatus = pathfinder_set_start_val(d_field_a, field_size, &h_start);
	HANDLE_ERROR(cudaStatus, "Failed to set start value\n");

	pathfinder_parameters.d_field_a = d_field_a;
	pathfinder_parameters.d_field_b = d_field_b;
	pathfinder_parameters.field_size = field_size;
	pathfinder_parameters.d_start = d_start;
	pathfinder_parameters.d_finish = d_finish;

	/* Start Timer */
	cudaEventRecord(timer_start, 0);

	/* Run Pathfinder*/
	path_exists = pathfinder_exec(&pathfinder_parameters);

	/* Stop Timer */
	cudaEventRecord(timer_stop, 0);
	cudaEventSynchronize(timer_stop);
	cudaEventElapsedTime(&pathfinder_time, timer_start, timer_stop);

	printf("Pathfinder: Field is scanned\n");

	if (path_exists) {
		printf("Pathfinder: Backtracing the path...\n");

		cudaStatus = cudaMemcpy(h_field,
			                    d_field_b,
			                    field_bytes,
			                    cudaMemcpyDefault);
		HANDLE_ERROR(cudaStatus, "Failed to copy data Device -> Host\n");

		/* Start Timer */
		cudaEventRecord(timer_start, 0);

		/* Run Path Backtrace */
		pathfinder_backtrace(h_field, field_size, &h_start, &h_finish);

		/* Stop Timer */
		cudaEventRecord(timer_stop, 0);
		cudaEventSynchronize(timer_stop);
		cudaEventElapsedTime(&backtrace_time, timer_start, timer_stop);

		printf("Pathfinder: Path backtracing done\n");
	}
	else {
		fprintf(stderr, "No path avaliable\n");
		goto ERROR;
	}

	printf("Pathfinder: Finish\n\n");

	/*----------------------------------------------------------------------------
	 * Print Statistics
	 *--------------------------------------------------------------------------*/
	printf("---------------[STATS]---------------\n\n");
	printf("Field generation time: %.3f milliseconds\n", generator_time);
	printf("Path finding time: %.3f milliseconds\n", pathfinder_time);
	printf("Path backtrace time: %.3f milliseconds\n", backtrace_time);
	printf("Total time: %.3f milliseconds\n", generator_time + pathfinder_time + backtrace_time);

#ifndef NDEBUG
	/* Print Out Field */
	printf("+");
	for (unsigned int i = 0; i < field_size * 2; i++) {
		printf("-");
	}
	printf("+\n");

	for (unsigned int i = 0; i < field_size; i++) {
		printf("|");
		for (unsigned int j = 0; j < field_size; j++) {
			if ((i == h_start.row) && (j == h_start.col)) {
				printf("S ");
			}
			else if ((i == h_finish.row) && (j == h_finish.col)) {
				printf("F ");
			}
			else {
				int point_val = h_field[field_size * i + j];
				if (point_val == BARRIER_VAL) {
					printf("# ");
				}
				else if (point_val == TRACE_VAL) {
					printf("+ ");
				}
				else {
					printf("  ");
				}
			}
		}
		printf("|\n");
	}

	printf("+");
	for (unsigned int i = 0; i < field_size * 2; i++) {
		printf("-");
	}
	printf("+\n");
#endif /* !NDEBUG */

ERROR:

	cudaFree(d_field_a);
	cudaFree(d_field_b);
	cudaFree(d_contour_list);
	cudaFree(d_start);
	cudaFree(d_finish);
	cudaFreeHost(h_field);
	cudaFreeHost(h_contour_list);
}
