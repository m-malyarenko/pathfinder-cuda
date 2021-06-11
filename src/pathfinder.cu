#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include "pathfinder.cuh"

extern unsigned int BLOCK_DIM;

__host__ bool pathfinder_check_target_points(int* d_field,
	                                         const size_t field_size,
	                                         point2d* h_start,
	                                         point2d* h_finish)
{
	if ((d_field == NULL) ||
		(h_start == NULL) ||
		(h_finish == NULL))
	{
		fprintf(stderr, "PATHFINDER ERROR: Argument is a NULL pointer\n");
		return false;
	}

	cudaError_t cudaStatus = cudaSuccess;

	int start_val = 0;
	int finish_val = 0;

	cudaStatus = cudaMemcpy(&start_val,
		                    &d_field[h_start->row * field_size + h_start->col],
		                    sizeof(int),
		                    cudaMemcpyDefault);
	HANDLE_ERROR(cudaStatus, "PATHFINDER ERROR: cudaMemcpy Device -> Host failed\n");

	cudaStatus = cudaMemcpy(&finish_val,
		                    &d_field[h_finish->row * field_size + h_finish->col],
		                    sizeof(int),
		                    cudaMemcpyDefault);
	HANDLE_ERROR(cudaStatus, "PATHFINDER ERROR: cudaMemcpy Device -> Host failed\n");

	if ((start_val == BARRIER_VAL) || (finish_val == BARRIER_VAL)) {
		return false;
	}
	else {
		return true;
	}

ERROR:
	return false;
}

__host__ cudaError_t pathfinder_set_start_val(int* d_field, const size_t field_size, point2d* h_start) {
	cudaError_t cudaStatus = cudaSuccess;

	int start_val = START_VAL;
	cudaStatus = cudaMemcpy(&d_field[h_start->row * field_size + h_start->col],
		&start_val,
		sizeof(int),
		cudaMemcpyHostToDevice);

	HANDLE_ERROR(cudaStatus, "PATHFINDER ERROR: cudaMemcpy Host -> Device failed\n");
	return cudaSuccess;

ERROR:
	return cudaStatus;
}

__host__ bool pathfinder_exec(pathfinder_param* parameters) {
	if (parameters == NULL) {
		return false;
	}

	cudaError_t cudaStatus = cudaSuccess;
	bool pathfinder_status = false;
	unsigned int count = 0;

	int* d_field_prev = parameters->d_field_a;
	int* d_field_next = parameters->d_field_b;

	unsigned int* d_term_flag = NULL;
	unsigned int h_term_flag = 0;

	/* Set Blocks and Threads dimensions */
	dim3 block_dim;
	block_dim.x = BLOCK_DIM;
	block_dim.y = BLOCK_DIM;

	dim3 grid_dim;
	grid_dim.x = parameters->field_size / block_dim.x + 1;
	grid_dim.y = parameters->field_size / block_dim.y + 1;

	/* Create Terminating Flag */
	cudaStatus = cudaMalloc(&d_term_flag, sizeof(unsigned int));
	HANDLE_ERROR(cudaStatus, "PATHFINDER ERROR: cudaMalloc failed\n");
	cudaMemset(d_term_flag, 0, sizeof(unsigned int));
	HANDLE_ERROR(cudaStatus, "PATHFINDER ERROR: cudaMemset failed\n");

	HANDLE_ERROR(cudaStatus, "PATHFINDER ERROR: cudaMemcpy Host -> Device failed\n");

	/* Call the Kernel */
	do {
		pathfinder_main_kernel<<<grid_dim, block_dim>>>(d_field_prev,
			                                            d_field_next,
			                                            parameters->field_size,
			                                            parameters->d_start,
			                                            parameters->d_finish,
			                                            d_term_flag);

		cudaStatus = cudaGetLastError();
		HANDLE_ERROR(cudaStatus, "PATHFINDER ERROR: pathfinder_main_kernel launch failed\n");

		cudaStatus = cudaMemcpy(&h_term_flag,
			                    d_term_flag,
			                    sizeof(unsigned int),
			                    cudaMemcpyDeviceToHost);
		HANDLE_ERROR(cudaStatus, "PATHFINDER ERROR: cudaMemcpy Device -> Host failed\n");

		int* tmp = d_field_next;
		d_field_next = d_field_prev;
		d_field_prev = tmp;
		// #ifdef MODE
		// 	if (count > 10 * parameters->field_size) break;
		// 	count++;
		// #endif
	} while (!(h_term_flag & FINISH_REACHED) && (h_term_flag & NEXT_STEP_AVAILABLE));

	cudaStatus = cudaDeviceSynchronize();
	HANDLE_ERROR(cudaStatus, "PATHFINDER ERROR: cudaDeviceSynchronize failed\n");

	if (h_term_flag & FINISH_REACHED) {
		pathfinder_status = true;
	}
	else {
		pathfinder_status = false;
	}

ERROR:
	cudaFree(d_term_flag);

	return pathfinder_status;
}

__global__ void pathfinder_main_kernel(int* d_field_prev,
	                                   int* d_field_next,
	                                   const size_t field_size,
	                                   point2d* d_start,
	                                   point2d* d_finish,
	                                   unsigned int* d_term_flag)
{
	__shared__ unsigned int block_term_flag;
	block_term_flag = POINT_BLOCKED;

	unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

	if ((row < field_size) && (col < field_size)) {
		int cur_mark = d_field_prev[row * field_size + col];
		d_field_next[row * field_size + col] = cur_mark;

		bool is_finish_reached = (cur_mark != BLANK_VAL) && (row == d_finish->row) && (col == d_finish->col);

		if (is_finish_reached) {
			atomicOr(d_term_flag, FINISH_REACHED);
			return;
		}

		unsigned int thread_term_flag = 0;
		bool is_blank = (cur_mark == BLANK_VAL);
		bool is_barrier = (cur_mark == BARRIER_VAL);

		if (!is_blank && !is_barrier) {
			point2d cur_point = { row, col };
			thread_term_flag = pathfinder_mark_neigh(&cur_point,
				                                     cur_mark,
				                                     d_field_prev,
				                                     d_field_next,
				                                     field_size);
		}

		atomicOr(&block_term_flag, thread_term_flag);

		__syncthreads();

		atomicOr(d_term_flag, thread_term_flag);
	}
}

__device__ unsigned int pathfinder_mark_neigh(point2d* d_cur_point,
	                                          int d_mark,
	                                          int* d_field_prev,
	                                          int* d_field_next,
	                                          unsigned int field_size)
{
	unsigned int row = d_cur_point->row;
	unsigned int col = d_cur_point->col;

	bool is_top_boundary = (d_cur_point->row == field_size - 1);
	bool is_bottom_boundary = (d_cur_point->row == 0);
	bool is_right_boundary = (d_cur_point->col == field_size - 1);
	bool is_left_boundary = (d_cur_point->col == 0);

	unsigned int top_offset = (row + 1) * field_size + col;
	unsigned int bottom_offset = (row - 1) * field_size + col;
	unsigned int right_offset = row * field_size + (col + 1);
	unsigned int left_offset = row * field_size + (col - 1);

	bool has_top_neigh = !is_top_boundary && (d_field_prev[top_offset] == 0);
	bool has_bottom_neigh = !is_bottom_boundary && (d_field_prev[bottom_offset] == 0);
	bool has_right_neigh = !is_right_boundary && (d_field_prev[right_offset] == 0);
	bool has_left_neigh = !is_left_boundary && (d_field_prev[left_offset] == 0);

	if (!has_top_neigh && !has_bottom_neigh && !has_right_neigh && !has_left_neigh) {
		return POINT_BLOCKED;
	}

	if (has_top_neigh) {
		d_field_next[top_offset] = d_mark + 1;
	}

	if (has_bottom_neigh) {
		d_field_next[bottom_offset] = d_mark + 1;
	}

	if (has_right_neigh) {
		d_field_next[right_offset] = d_mark + 1;
	}

	if (has_left_neigh) {
		d_field_next[left_offset] = d_mark + 1;
	}

	return POINT_N_BLOCKED;
}

__host__ void pathfinder_backtrace(int* h_field, unsigned int field_size, point2d* h_start, point2d* h_finish) {
	unsigned int cur_row = h_finish->row;
	unsigned int cur_col = h_finish->col;
	int cur_mark = h_field[cur_row * field_size + cur_col];

	while (cur_mark != 0) {
		h_field[cur_row * field_size + cur_col] = TRACE_VAL;
		if ((cur_row + 1 < field_size) && (h_field[(cur_row + 1) * field_size + cur_col] == (cur_mark - 1))) {
			cur_row++;
		}
		else if ((cur_row - 1 > 0) && (h_field[(cur_row - 1) * field_size + cur_col] == (cur_mark - 1))) {
			cur_row--;
		}
		else if ((cur_col + 1 < field_size) && (h_field[cur_row * field_size + (cur_col + 1)] == (cur_mark - 1))) {
			cur_col++;
		}
		else if ((cur_col - 1 > 0) && (h_field[cur_row * field_size + (cur_col - 1)] == (cur_mark - 1))) {
			cur_col--;
		}
		cur_mark--;
	}
}
