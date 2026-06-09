# Lesson 06: Device Memory

The kernel in this lesson takes a pointer as an argument and writes into an array. This is the first time a kernel does useful work instead of just printing. It introduces the three functions that manage memory on the GPU: `cudaMalloc`, `cudaMemcpy`, and `cudaFree`.

## Why the kernel cannot use CPU memory

GPU and CPU have separate memory spaces. A pointer that is valid on the CPU points into system RAM, which the GPU cannot read or write. Passing a CPU pointer to a kernel compiles fine but causes a segfault at runtime. All data the kernel touches must live in device memory.

## cudaMalloc

`cudaMalloc` allocates memory on the GPU and stores the resulting pointer in a variable you pass by address:

```c
int *d_arr;
cudaMalloc(&d_arr, N * sizeof(int));
```

The `d_` prefix is a convention, not a requirement. It marks the pointer as pointing into device memory so you do not accidentally pass it to regular C functions. The size argument works the same way as `malloc`.

## cudaMemcpy

`cudaMemcpy` copies a block of bytes between CPU and GPU memory. The fourth argument sets the direction:

```c
cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);
```

`cudaMemcpyHostToDevice` copies from CPU to GPU before the kernel runs. `cudaMemcpyDeviceToHost` copies results back after. The copy blocks until it finishes, so there is no need to call `cudaDeviceSynchronize` between the copy and the next CPU statement.

## cudaFree

`cudaFree` releases device memory. It takes the pointer directly, not its address:

```c
cudaFree(d_arr);
```

Not calling it leaks GPU memory for the lifetime of the process.

## The kernel

The kernel receives `d_arr` and each thread writes its global ID into the corresponding slot:

```c
__global__ void fillWithID(int *arr)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    arr[id] = id;
}
```

The global ID formula is from Lesson 02. With `<<<2, 4>>>` there are 8 threads numbered 0 through 7. Thread 0 writes 0, thread 7 writes 7.

## Code

```c
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void fillWithID(int *arr)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    arr[id] = id;
}

int main()
{
    int N = 8;
    int h_arr[8];
    int *d_arr;

    cudaMalloc(&d_arr, N * sizeof(int));

    fillWithID<<<2, 4>>>(d_arr);

    cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);

    for (int i = 0; i < N; i++)
        printf("arr[%d] = %d\n", i, h_arr[i]);

    return 0;
}
```

## Compile and run

```bash
nvcc first_kernel.cu -o first_kernel
./first_kernel
```

Expected output:

```
arr[0] = 0
arr[1] = 1
arr[2] = 2
arr[3] = 3
arr[4] = 4
arr[5] = 5
arr[6] = 6
arr[7] = 7
```

Output is deterministic here. Each thread writes to a different slot, so there is no race and no ordering dependency.

## Visual

```
CPU (host)                         GPU (device)
-----------                        ------------
h_arr[8]  ---cudaMemcpy H2D-->    d_arr[8]
                                       |
                                  fillWithID<<<2,4>>>
                                  Thread 0: d_arr[0] = 0
                                  Thread 1: d_arr[1] = 1
                                  ...
                                  Thread 7: d_arr[7] = 7

          <--cudaMemcpy D2H---    d_arr[8]
h_arr[8]
  [0]=0 [1]=1 [2]=2 [3]=3
  [4]=4 [5]=5 [6]=6 [7]=7
```

## Glossary

- device memory: memory physically on the GPU board. The kernel can read and write it. The CPU cannot access it directly.
- host memory: regular system RAM. Accessible by the CPU. Not directly readable by the GPU.
- `cudaMalloc`: allocates N bytes in device memory. Takes the address of a pointer and fills it in.
- `cudaFree`: releases device memory allocated by `cudaMalloc`.
- `cudaMemcpy`: copies bytes between host and device memory. Direction is set by the fourth argument.
- `cudaMemcpyHostToDevice`: copy direction from CPU to GPU.
- `cudaMemcpyDeviceToHost`: copy direction from GPU to CPU.
- `d_` prefix: naming convention for pointers into device memory. Not enforced by the compiler.
- `h_` prefix: naming convention for pointers into host memory. Often used alongside `d_` to keep the two clear.
