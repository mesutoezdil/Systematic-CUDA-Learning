#include <cstdio>
#include <cuda_runtime.h>

__global__ void vectorAdd(float *a, float *b, float *c) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main() {
    int n = 8;
    size_t bytes = n * sizeof(float);

    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    for (int i = 0; i < n; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, bytes);
    cudaMalloc((void**)&d_b, bytes);
    cudaMalloc((void**)&d_c, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    vectorAdd<<<1, 8>>>(d_a, d_b, d_c);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    int success = 1;
    for (int i = 0; i < n; ++i) {
        if (h_c[i] != (h_a[i] + h_b[i])) {
            printf("Error at index %d: Got %f, expected %f\n",
                   i, h_c[i], (h_a[i] + h_b[i]));
            success = 0;
            break;
        }
    }
    if (success) {
        printf("All elements are correct.\n");
    }

    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}