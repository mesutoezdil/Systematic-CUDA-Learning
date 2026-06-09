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
