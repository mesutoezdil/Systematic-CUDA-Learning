# Compute Capability

If you spend any time with CUDA or NVIDIA GPUs, you will run into one term constantly "compute capability". 

Sometimes written as CC, sometimes called a version number, but always the same idea. Let's break it down.

Compute capability is NVIDIA's classification system for describing the features and processing power of a GPU. 

It is not a marketing score or a benchmark number. 

It is a precise indicator of what a specific GPU architecture can and cannot do.

Think of it as a specification sheet compressed into a single number.

## How the Numbering Works

Compute capability is expressed as a version number, like 3.0, 5.1, or 7.5.

The logic is consistent across all generations:

- The number before the dot signals a major architectural change
- The number after the dot represents minor improvements or extensions

So going from 7.x to 8.x is not just a speed bump, it marks a fundamentally different architecture with new hardware units and new capabilities.

## Walking Through the Architectures

Volta → CC 7.x

Volta's compute capability of 7.x was significant for one reason above all else: it introduced Tensor Cores. These specialized units dramatically accelerated the kinds of matrix operations that power AI and deep learning workloads. Before Volta, those operations ran on general-purpose CUDA cores. After Volta, they had dedicated hardware.

Ampere → CC 8.x

Ampere moved the needle further. The jump to 8.x brought more powerful and efficient Tensor Cores, larger memory bandwidth, and better energy efficiency. The same ideas from Volta, refined and expanded.

Hopper → CC 9.x

Hopper represented another major step, introducing new execution models and pushing AI performance forward. The minimum supported CUDA toolkit version for Hopper is 11.8, anything lower will throw a compatibility error.

Blackwell → CC 10.0 (B200/GB200) and 12.0 (RTX PRO / RTX 50 series)

Blackwell is the current generation as of 2026. It introduces 5th-generation Tensor Cores and an entirely new precision format -NVFP4- which doubles throughput compared to FP8 for large model inference. This is not an incremental improvement. FP4 acceleration simply does not exist on previous architectures. To build natively for Blackwell, you need CUDA Toolkit 12.8.

## Feature Support

CUDA's official documentation includes detailed tables mapping features to compute capability versions. 

Scanning that table reveals patterns quickly:

- GPUs at CC 5.0 do not support half-precision (FP16) operations
- Tensor Cores appear only from CC 7.x onward
- FP8 acceleration kicks in at CC 9.x
- NVFP4 requires CC 10.0 or higher

This matters because the absence of a feature is absolute. 

If your GPU does not have Tensor Cores, you cannot use them.

There is no software workaround, no emulation path. 

The hardware either has the unit or it does not.

So the first question to ask before writing any performance-sensitive CUDA code is not "is my GPU fast enough?" but "does my GPU support what I need?"

## Software Compatibility

Compute capability also determines which CUDA toolkit versions you can use. 

Higher compute capability unlocks newer toolkit versions, which in turn unlock more features and better optimizations.

A few concrete examples:

- Maxwell (CC 5.x) - requires CUDA 6.5 or higher
- Hopper (CC 9.x) - requires CUDA 11.8 or higher
- Blackwell (CC 10.0) - requires CUDA 12.8 for native cubin support

Using a toolkit version below the minimum for your architecture will produce a hard error. 

The code will not compile, or it will fail at runtime. There is no gray area here.

The practical workflow is always the same: identify your GPU's compute capability first, then choose your CUDA version accordingly, then write your code.

## The Low-Level Layer (PTX)

Going deeper: CUDA code does not run directly on the GPU. 

It compiles down to PTX, a low-level intermediate representation that functions roughly like an assembly language for NVIDIA GPUs.

PTX is where compute capability becomes most concrete. 

Certain PTX instructions require specific hardware units that only exist from a particular compute capability onward. 

The warp shuffle functions, which let threads within a warp share data without touching shared or global memory, are only available on devices with CC higher than 5.0.

If your GPU does not meet the minimum, those instructions simply cannot execute. 

The hardware for them does not exist on that chip.

## The Practical Takeaway

Whether you are building a machine learning pipeline, running a physics simulation, or writing a custom CUDA kernel, the same principle applies: your GPU's compute capability is the contract between your hardware and your code.

Know your CC number. Cross-reference it with the CUDA documentation. 

Choose the right toolkit version. Then build.

Everything else, performance tuning, optimization, feature selection, flows from that single starting point.

> Compute capability is not just a version number. It is the definition of what your GPU can actually do.