#ifndef  __COMMON_CUH__
#define __COMMON_CUH__

/* Define -------------------------------------------------------------------*/

#define MIN_FIELD_SIZE ((long) 4)
#define MAX_FIELD_SIZE ((long) 10000)

#define MIN_GRID_DIM ((unsigned int) 1)
#define MAX_GRID_DIM ((unsigned int) 100)

#define MIN_BLOCK_DIM ((unsigned int) 1)
#define MAX_BLOCK_DIM ((unsigned int) 32)

#define BLANK_VAL   ((int)  0)
#define BARRIER_VAL ((int) -1)
#define TRACE_VAL   ((int) -2)

#define TARGET_IS_BARIER   ((unsigned int) 1)
#define TARGET_IS_N_BARIER ((unsigned int) 0)

#define FINISH_REACHED      ((unsigned int) 0x00000002)
#define NEXT_STEP_AVAILABLE ((unsigned int) 0x00000001)

#define POINT_BLOCKED   ((unsigned int) 0)
#define POINT_N_BLOCKED ((unsigned int) 1)

#define PATHFINDER_NO_RESULT      ((unsigned int) 0)
#define PATHFINDER_RESULT_FIELD_A ((unsigned int) 1)
#define PATHFINDER_RESULT_FIELD_B ((unsigned int) 2)

/* Macro --------------------------------------------------------------------*/

#define FIELD_BYTES(N) ((size_t) (N) * (size_t) (N) * sizeof(int))
#define HANDLE_ERROR(status, message) if ((status) != cudaSuccess) { fprintf(stderr, message); goto ERROR;} 

#endif /* __COMMON_CUH__ */
