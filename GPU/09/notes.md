# Understanding Compute Capability

At some point in my CUDA learning journey, I kept seeing this term everywhere: "compute capability"

At first, I ignored it. Just another version number, I thought.

But after a while, I realized something important. 

This number is not just a label. It actually defines what your GPU can and cannot do.

## What "Compute Capability" Really Means

Compute capability (often written as CC) is NVIDIA's way of describing the feature set of a GPU.

It tells you which hardware features exist inside the GPU, which instructions are supported, which CUDA features you can use, and which CUDA versions are compatible.

So it is not just about speed. It is about capability.

Compute capability is written like this: 5.0, 6.1, 7.5, 8.0, 9.0...

There is a simple logic behind it. The first number marks a major architecture change, and the second marks smaller improvements. 

So when you go from 7.x → 8.x → 9.x, you are not just getting faster hardware. You are getting new features and new instructions.

## Mapping It to Real Architectures

When I started connecting compute capability to real architectures, everything became clearer:

| Architecture                      | Compute Capability  |
|-----------------------------------|---------------------|
| Maxwell                           | 5.x                 |
| Pascal                            | 6.x                 |
| Volta                             | 7.x                 |
| Ampere                            | 8.x                 |
| Hopper                            | 9.x                 |
| Blackwell (B200 / GB200)          | 10.0                |
| Blackwell RTX PRO / RTX 50 series | 12.0                |
| Rubin (2026 and beyond)           | Continuing...       |

Each step adds something meaningful. Volta introduced Tensor Cores. Ampere improved them and strengthened the memory architecture. Hopper brought new execution models and better AI performance. Blackwell then introduced 5th-generation Tensor Cores along with entirely new precision formats like FP4 (NVFP4), not just a speed bump, but new capability that simply did not exist before.

Compute capability is basically a shortcut: one number tells you the architecture level.

## Why This Matters (The Part I Missed at First)

At the beginning, I thought: "If my GPU is powerful, I can run everything."

That is not true.

Some features simply do not exist on older GPUs, no matter how fast they are. Tensor Cores are a good example. If your GPU does not support them, you cannot use them. No workaround.

So compute capability answers a very practical question: "Can my GPU actually run this feature?"

## Feature Support (Concrete Examples)

When you look at CUDA documentation, you will find tables showing which features are supported at which compute capability level. This is extremely useful, instead of guessing, you can check directly.

A few examples:

- Older GPUs (5.x) → limited FP16 support
- Tensor Cores → available from 7.x and above
- FP8 → accelerated from 9.x (Hopper) onward
- NVFP4 → only on 10.0 (Blackwell) and above

So instead of asking "Why is my code not working?", you ask: "Does my GPU even support this?"

## Software Compatibility

Compute capability also determines which CUDA version you need. This matters a lot in practice because CUDA toolkits are tied to GPU capabilities.

For example, working natively with Blackwell requires CUDA Toolkit 12.8. 

Earlier versions cannot generate native cubins for Blackwell. They can only fall back to PTX with JIT compilation at runtime.

If you mismatch them, your code may fail to compile or break at runtime.

The workflow is simple:
1. Check GPU compute capability
2. Choose a compatible CUDA version
3. Write code accordingly

## The Low-Level View (PTX and Instructions)

Going deeper, I discovered something else. CUDA is not just high-level code. Under the hood, it uses something called PTX, a low-level instruction layer, something like an assembly language for GPUs.

Here, compute capability becomes even more important. Some instructions only exist on certain GPU versions. 

Warp-level operations (fast thread communication within a warp) are a good example. They require a minimum compute capability.

Hardware defines what is possible. That is the bottom line.

## A Better Way to Think About GPUs

At this point, my thinking shifted.

Instead of asking "Which GPU is faster?", I started asking: "What can this GPU actually do?"

Because two GPUs can have similar specs on paper but support completely different features. This is especially visible with modern GPUs in 2026. Blackwell's FP4 compute capability simply does not exist on Hopper. That is not a performance gap. That is a capability gap.

## Insight for todays section

Compute capability is not just a number. It is a bridge between hardware, CUDA software, and your code.

If they do not match, things either break or run inefficiently.

So before writing any CUDA code, always check compute capability first. That one number tells you which features you can use, which optimizations are possible, and which CUDA version you need.

Once I understood this, everything got easier. Reading documentation, debugging issues, understanding performance.

And as GPUs continue evolving through Rubin and beyond, the concept stays the same:

> Compute capability defines the real limits of your GPU.