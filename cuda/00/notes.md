# Running Your First CUDA Program

## My environment
- Machine (ubuntu@nebius-tarantula) in Nebius Cloud
- CUDA version: 13.0 (V13.0.88)
- Compiler: `nvcc`, NVIDIA CUDA Compiler Driver

```bash
ubuntu@nebius-tarantula:~/Systematic-CUDA-Learning$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Wed_Aug_20_01:58:59_PM_PDT_2025
Cuda compilation tools, release 13.0, V13.0.88
Build cuda_13.0.r13.0/compiler.36424714_0
```

## Steps

### 1 > Go to the code directory
```bash
cd ~/Systematic-CUDA-Learning/cuda/00/code
```

### 2 > Compile and run `first_kernel.cu`
```bash
ubuntu@nebius-tarantula:~/Systematic-CUDA-Learning/cuda/00/code$ nvcc first_kernel.cu -o first_kernel && ./first_kernel
All elements are correct.
```

### 3 > Compile and run `first_kernel_with_commentary.cu`
```bash
ubuntu@nebius-tarantula:~/Systematic-CUDA-Learning/cuda/00/code$ nvcc first_kernel_with_commentary.cu -o first_kernel_with_commentary && ./first_kernel_with_commentary
All elements are correct.
```

### 4 > Check the files
```bash
ubuntu@nebius-tarantula:~/Systematic-CUDA-Learning/cuda/00/code$ ls
first_kernel  first_kernel.cu  first_kernel_with_commentary  first_kernel_with_commentary.cu
```

## What happened
- `nvcc` compiled the `.cu` source file into a binary
- `&&` runs the binary only if compilation succeeded
- `-o first_kernel` sets the output binary name
- Both files produce: `All elements are correct.`

## Why two steps (compile + run)?
Unlike Python, CUDA/C++ must be compiled first into machine code.
`nvcc` does the compilation, then you execute the binary with `./`.