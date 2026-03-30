# Getting Started with CUDA

We are starting to learn CUDA. For now, we’ll just focus on what we need.

Before diving into deep topics, let’s clarify one basic question:

Is it enough to just write code and run it on a GPU to get optimal performance?

The answer is: No.

So why?

Bcs even though CPUs and GPUs may look similar at first glance, they are designed for completely different purposes.

## CPU vs GPU

Both have similar components, but they are used differently:

### Memory (DRAM / global memory)

- CPU → uses system RAM  
- GPU → has its own dedicated memory (VRAM)

### Cache

- CPU → L1, L2, and L3 cache  
- GPU → usually L1 and L2 cache  
- GPU also has shared memory, which is critical for parallel processing  

**Note:**  
L1, L2, and L3 cache are small but very fast memory layers inside the CPU used to access data quickly.

### Processing Units

- CPU → few but powerful processing units (ALU, etc.)  
- GPU → many simpler cores (for example, CUDA cores in NVIDIA architecture)

## The real difference

- CPU → optimized for low-latency, sequential tasks  
- GPU → optimized for high parallelism  

In simple terms:

- CPU → does one task very fast  
- GPU → does many tasks at the same time  

## ALU vs GPU Core

If we compare them directly:

- CPU processing units → more powerful and complex  
- GPU cores → simpler but much more numerous  

What is an ALU?

ALU (Arithmetic Logic Unit) is the part of the processor that performs basic operations like addition, subtraction, and comparisons.

## Frequency difference (important detail)

- Modern CPU → around 3–4 GHz  
- NVIDIA A100 GPU core → ~765–1200 MHz  

So at a single-core level, CPUs are faster.

But that is not the main strength of GPUs.

## Why are GPUs powerful?

Bcs:

→ They can execute many operations in parallel at the same time!

This makes GPUs ideal for:

- graphics processing  
- artificial intelligence  
- large-scale data computations  

## Again

- CPU → sequential tasks  
- GPU → parallel tasks  

That’s why this statement is wrong:

> "GPUs are always better than CPUs"

## How CPU and GPU work together

In a system, you have: CPU, RAM and PCI interface  

The GPU connects to the system via PCI.

So:

- CPU → uses system RAM  
- GPU → uses its own memory  

## GPU Architecture (NVIDIA-focused)

Inside a GPU, the most important structure is:

### Streaming Multiprocessor (SM)

An SM is the basic working unit of a GPU.

A GPU is essentially made up of many SMs working together, and all parallel computation happens through them.

---

## What is inside an SM?

Each SM works like a small processor and contains:

- L1 cache  
  Small fast memory for frequently used data  

- Registers 
  The fastest memory, used directly by each thread  

- Scheduler  
  Decides which tasks run and when  

- Dispatcher  
  Sends tasks to execution units  

## Execution units (the actual workers)

These units perform the computations:

- Floating-point cores  
  Handle decimal calculations (AI, graphics, etc.)

- Integer cores  
  Handle integer operations  

- Tensor cores  
  Specialized for fast matrix operations (very important for AI)

- Special Function Units (SFU)  
  Handle complex math operations like log, sin, cos  

- Load/Store units  
  Move data between memory and registers  

## L2 cache (global level)

L2 cache belongs to the entire GPU, not a single SM.

It is larger than L1 but slower, and shared across SMs.
