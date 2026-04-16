#include <cstdio>
#include <cuda_runtime.h>

/*
 * CUDA Kernel: vectorAdd
 * ----------------------
 * Runs on the GPU. Each thread handles exactly one element of the array.
 * threadIdx.x gives the thread's position within its block (0..7 here).
 * With 8 threads and 8 elements, every addition happens in parallel —
 * no loops needed on the GPU side.
 *
 * __global__ means: called from CPU, executed on GPU.
 * Pointers a, b, c point to GPU (device) memory, not CPU memory.
 */
__global__ void vectorAdd(float *a, float *b, float *c) {
    int i = threadIdx.x; // unique thread ID → maps directly to array index
    c[i] = a[i] + b[i]; // each thread does its own independent addition
}

int main() {
    int n = 8;
    size_t bytes = n * sizeof(float); // total bytes needed for one vector (32 bytes)

    /*
     * Host memory = regular RAM, accessible by the CPU.
     * The GPU cannot read this directly — data must be explicitly copied.
     * malloc returns a raw pointer; we cast to float*.
     */
    float *h_a = (float*)malloc(bytes); // input vector A (host)
    float *h_b = (float*)malloc(bytes); // input vector B (host)
    float *h_c = (float*)malloc(bytes); // output vector C (host, filled after GPU run)

    /*
     * Fill input vectors with simple test values:
     *   h_a = [0, 1, 2, 3, 4, 5, 6, 7]
     *   h_b = [0, 2, 4, 6, 8, 10, 12, 14]
     * Expected result:
     *   h_c = [0, 3, 6, 9, 12, 15, 18, 21]
     */
    for (int i = 0; i < n; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }

    /*
     * Device memory = GPU's own VRAM, not accessible by the CPU.
     * cudaMalloc works like malloc but allocates on the GPU.
     * We pass a pointer-to-pointer (**) because cudaMalloc writes
     * the allocated address into our pointer variable.
     */
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, bytes); // allocate GPU memory for A
    cudaMalloc((void**)&d_b, bytes); // allocate GPU memory for B
    cudaMalloc((void**)&d_c, bytes); // allocate GPU memory for C (output)

    /*
     * cudaMemcpy transfers data between CPU and GPU.
     * Direction is controlled by the last argument:
     *   cudaMemcpyHostToDevice → CPU → GPU
     *   cudaMemcpyDeviceToHost → GPU → CPU
     * This is a blocking call: CPU waits until transfer is complete.
     */
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice); // send A to GPU
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice); // send B to GPU

    /*
     * Kernel launch syntax: kernelName<<<gridDim, blockDim>>>(args)
     *   gridDim  = number of blocks  → 1
     *   blockDim = threads per block → 8
     * Total threads = 1 × 8 = 8, one per element.
     *
     * For larger arrays you'd scale up: e.g. <<<(n+255)/256, 256>>>
     * and use: int i = blockIdx.x * blockDim.x + threadIdx.x;
     *
     * Kernel launch is ASYNCHRONOUS — CPU continues immediately.
     * The following cudaMemcpy acts as an implicit synchronization point.
     */
    vectorAdd<<<1, 8>>>(d_a, d_b, d_c);

    /*
     * Copy the GPU result back to CPU memory.
     * This also implicitly waits for the kernel to finish (sync point).
     * After this call, h_c contains the computed sums.
     */
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    /*
     * Verify correctness on the CPU.
     * We recompute the expected value and compare with GPU output.
     * Floating-point equality (!=) is safe here because the values
     * are small integers — no rounding errors expected.
     */
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

    /*
     * Always free both host and device memory to avoid leaks.
     * free()     → releases CPU (heap) memory
     * cudaFree() → releases GPU (VRAM) memory
     * Skipping cudaFree() in longer programs causes GPU memory leaks.
     */
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}