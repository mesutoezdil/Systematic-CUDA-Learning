# Lesson 07: Warp IDs

This lesson covers the third level of the CUDA hierarchy: warps. Block and thread IDs were covered in Lesson-01 and Lesson-02. Here we look at the hardware layer beneath those and learn how to calculate which warp a thread belongs to from inside the kernel. Machine: NVIDIA L40S, CUDA 13.0, Ubuntu 24.

## CUDA Hierarchy

The software levels in CUDA are:

```
Grid
  └── Blocks
        └── Warps  (exactly 32 threads each)
              └── Threads
```

You choose how many blocks and how many threads per block using the `<<<num_blocks, threads_per_block>>>` syntax (see Lesson-01, Lesson-02). The warp size is always 32 on NVIDIA GPUs, it is hardcoded into the hardware and cannot be changed. The warp is the actual scheduling unit on the GPU: the GPU does not run threads one by one, it runs them in groups of 32.

Warp count limits depend on the hardware. Values measured on the L40S with `cudaGetDeviceProperties`:

- Max warps per block: 32 (max 1024 threads / 32, applies to all GPUs)
- Max concurrent warps per SM: 48
- SM count: 142
- Max concurrent warps across the entire GPU: 6,816

## warp_id Is Not a Built-in Variable

In regular C code you define a variable before you use it. `blockIdx.x` and `threadIdx.x` are different: you do not define them yourself. When the GPU runs each thread, it fills in these values automatically for that thread. You just read them. But there is no such thing for `warp_id`. The GPU does not fill it in. You calculate it yourself inside the kernel:

```c
int warp_id = threadIdx.x / 32;
```

In a block with 128 threads the result is: threads 0-31 → warp 0, threads 32-63 → warp 1, threads 64-95 → warp 2, threads 96-127 → warp 3. Total: 128 / 32 = 4 warps.

## What Happens with 1024 Threads

I first launched the kernel with 1 block and 1024 threads (`<<<1, 1024>>>`). The warp IDs in the output went from 0 to 31. At first it looked too high, but the math was correct: 1024 / 32 = 32 warps, so warp IDs go from 0 to 31. Each warp ID had exactly 32 threads:

```
Block ID: 0 --- Thread ID:    0 --- Warp ID:  0
Block ID: 0 --- Thread ID:    1 --- Warp ID:  0
...
Block ID: 0 --- Thread ID:   31 --- Warp ID:  0
Block ID: 0 --- Thread ID:   32 --- Warp ID:  1
...
Block ID: 0 --- Thread ID:  992 --- Warp ID: 31
...
Block ID: 0 --- Thread ID: 1023 --- Warp ID: 31
```

Verified on the machine: warps 0-31, exactly 32 threads each, 1024 lines total.

```
32 warp 0
32 warp 1
...
32 warp 31
```

## Warp ID Resets Per Block

Warp ID starts from zero in every block. If there are 2 blocks, you see warp ID 0 and 1 in both of them. So when you see warp ID 0 in the output, you cannot tell which block it belongs to without also looking at the block ID. Launching with `<<<2, 64>>>` gives 64 threads per block = 2 warps per block, and `warp_id=0` appears twice in the output: once for block 0 and once for block 1.

## Lane ID (Exercise)

I also need to look into the modulo operator. `threadIdx.x % 32` does not give the warp ID. Each warp has 32 threads and those threads have a position inside the warp from 0 to 31. That position is called the lane ID. Thread 33 is in warp 1 and its lane ID is 1 (33 % 32 = 1). Threads 0, 32, and 64 are in different warps but all have lane ID 0. To get the warp ID you need division (`/`), modulo alone is not enough.

## Code

Kernel launched with 1 block and 128 threads. The `test01` function runs on the GPU. Each thread calculates its own `warp_id` using `threadIdx.x / 32` and prints the block ID, thread ID, and warp ID. In `main` the kernel is launched with 1 block and 128 threads, then `cudaDeviceSynchronize()` blocks the CPU until the GPU finishes so the output is not lost before the program exits.

```c
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void test01()
{
    int warp_id = threadIdx.x / 32;
    printf("Block ID: %d --- Thread ID: %d --- Warp ID: %d\n",
           blockIdx.x, threadIdx.x, warp_id);
}

int main()
{
    // 1 block, 128 threads -> 4 warps (IDs: 0,1,2,3)
    test01<<<1, 128>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

- `#include "cuda_runtime.h"`: required header for CUDA functions.
- `#include "device_launch_parameters.h"`: defines GPU built-in variables like `blockIdx` and `threadIdx`.
- `#include <stdio.h>`: standard C header required for `printf`.
- `__global__`: marks this function as a kernel, called from the CPU and executed on the GPU.
- `int warp_id = threadIdx.x / 32;`: each thread calculates its own warp ID. Threads 0-31 → 0, threads 32-63 → 1, and so on.
- `printf(...)`: each thread prints its block ID, thread ID, and warp ID.
- `test01<<<1, 128>>>();`: launches the kernel with 1 block and 128 threads.
- `cudaDeviceSynchronize();`: blocks the CPU until all GPU threads finish and the output is written.

## warp_ids_2blocks.cu

Uses the same kernel as `warp_ids.cu`. The only difference is the launch config: `<<<2, 64>>>`. 2 blocks, 64 threads per block. Each block gets 64 / 32 = 2 warps. This file was written to show warp ID resetting per block.

```c
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void test01()
{
    int warp_id = threadIdx.x / 32;
    printf("Block ID: %d --- Thread ID: %d --- Warp ID: %d\n",
           blockIdx.x, threadIdx.x, warp_id);
}

int main()
{
    // 2 blocks, 64 threads/block -> 2 warps per block, warp ID resets per block
    test01<<<2, 64>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

- `test01<<<2, 64>>>();`: launches the kernel with 2 blocks and 64 threads per block. 128 threads total, 4 warps total, split across 2 blocks.
- All other lines are identical to `warp_ids.cu`.

## Compile and Run

Both files are in the `code/` directory:

```bash
# 1 block, 128 threads -> 4 warps
nvcc -arch=sm_89 -o warp_ids warp_ids.cu
./warp_ids

# 2 blocks, 64 threads/block -> 2 warps per block
nvcc -arch=sm_89 -o warp_ids_2blocks warp_ids_2blocks.cu
./warp_ids_2blocks
```

## Output: `<<<1, 128>>>`

128 lines, 4 warps. Real L40S output (thread order is not guaranteed, the listing below is sorted):

```
Block ID: 0 --- Thread ID:  0 --- Warp ID: 0
Block ID: 0 --- Thread ID:  1 --- Warp ID: 0
Block ID: 0 --- Thread ID:  2 --- Warp ID: 0
...
Block ID: 0 --- Thread ID: 31 --- Warp ID: 0
Block ID: 0 --- Thread ID: 32 --- Warp ID: 1
Block ID: 0 --- Thread ID: 33 --- Warp ID: 1
...
Block ID: 0 --- Thread ID: 63 --- Warp ID: 1
Block ID: 0 --- Thread ID: 64 --- Warp ID: 2
...
Block ID: 0 --- Thread ID: 95 --- Warp ID: 2
Block ID: 0 --- Thread ID: 96 --- Warp ID: 3
...
Block ID: 0 --- Thread ID: 127 --- Warp ID: 3
```

## Output: `<<<2, 64>>>`

128 lines, 2 blocks, 2 warps per block. Confirms that warp ID resets per block:

```
Block ID: 0 --- Thread ID:  0 --- Warp ID: 0
...
Block ID: 0 --- Thread ID: 31 --- Warp ID: 0
Block ID: 0 --- Thread ID: 32 --- Warp ID: 1
...
Block ID: 0 --- Thread ID: 63 --- Warp ID: 1
Block ID: 1 --- Thread ID:  0 --- Warp ID: 0   <- resets to zero
...
Block ID: 1 --- Thread ID: 31 --- Warp ID: 0
Block ID: 1 --- Thread ID: 32 --- Warp ID: 1
...
Block ID: 1 --- Thread ID: 63 --- Warp ID: 1
```

Block 1 shows warp_id 0 again because warp IDs start from zero in every block. There is no global warp number across the entire GPU.

## Visual

```
test01<<<1, 128>>>
              |
    1 block --+-- 128 threads/block

Block 0
+-------------------------------------------------------+
|  Warp 0: threads   0 -  31  (0/32 = 0)               |
|  Warp 1: threads  32 -  63  (32/32 = 1)              |
|  Warp 2: threads  64 -  95  (64/32 = 2)              |
|  Warp 3: threads  96 - 127  (96/32 = 3)              |
+-------------------------------------------------------+

test01<<<2, 64>>>

Block 0                          Block 1
+---------------------------+   +---------------------------+
|  Warp 0: threads  0 - 31 |   |  Warp 0: threads  0 - 31 |
|  Warp 1: threads 32 - 63 |   |  Warp 1: threads 32 - 63 |
+---------------------------+   +---------------------------+
  warp_id: 0, 1                   warp_id: 0, 1  (resets)
```

## Glossary

- warp: a group of 32 threads the GPU runs together as one unit. The GPU schedules warps, not individual threads.
- warp size: always 32 on NVIDIA GPUs. Cannot be changed from software.
- warp ID: which warp a thread belongs to within its block. Calculated as `threadIdx.x / 32`.
- lane ID: a thread's position inside its warp. Calculated as `threadIdx.x % 32`, ranges from 0 to 31. Does not give the warp ID.
- warps per block: `(threads per block) / 32`. 128 threads/block → 4 warps/block.
- warp ID reset: warp IDs start from zero in every block, just like `threadIdx.x`. There is no global warp ID across the entire GPU.
