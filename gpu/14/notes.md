# Running Linux on Windows (A Practical Setup with WSL)

At some point, working seriously with CUDA naturally leads you toward Linux.

This is not bcs Windows is unusable, but bcs the GPU ecosystem has been built around Linux for years. Most tooling, docs, and real-world deployments assume a Linux env. If you look at how modern GPU systems are used in 2026, especially in AI and high-performance computing, Linux is almost always part of the stack.

You can use WSL too.

WSL, or Windows Subsystem for Linux, allows you to run a real Linux env inside Windows. This is not an emulation layer like older solutions. With WSL2, you are running an actual Linux kernel, which makes a significant difference in behavior, compatibility, and performance.

Because of this, WSL has become a standard part of many development workflows.

The setup itself looks simple on the surface.

You open a terminal on Windows and run a single command: `wsl --install`

As of 2026, one detail is especially important.

You should always use WSL2.

The earlier version, WSL1, does not provide the same level of compatibility and does not support GPU acceleration in a meaningful way. WSL2, on the other hand, is designed with modern workloads in mind and is the foundation for CUDA support on Windows.

If the system is not using WSL2, many GPU-related features will simply not work as expected.

When you launch your Linux distribution for the first time, you are asked to create a username and password.

This is not a minor step.

You are not entering your Windows env. You are entering a separate Linux env that runs on the same machine. It has its own users, its own file system structure, and its own package management.

From this point on, you are effectively working in two systems at once.

The real importance of WSL becomes clear when you look at GPU access.

With WSL2, the Linux env can access the GPU through the Windows driver. This means CUDA apps can run inside WSL almost as if they were running on a native Linux system.

This is a major shift. It allows you to develop in a Linux environment while still using Windows as your main operating system.

There is one detail that often causes confusion.

The GPU driver is installed on the Windows side, not inside WSL.

WSL does not require a separate NVIDIA driver installation. Instead, it uses the driver that is already installed on the host system. Trying to install a Linux GPU driver inside WSL usually leads to conflicts and should be avoided.

Understanding this separation is important for maintaining a stable setup.

Another point that becomes relevant in practice is how CUDA is installed.

Inside WSL, you install the Linux version of the CUDA Toolkit, not the Windows one. However, even here there is a distinction. The packages used for WSL are designed to work with the shared driver model and avoid conflicts with the host system.

This means the installation process looks similar to Linux, but it is not identical.

Looking at the situation in 2026, WSL is no longer just a convenience tool.

It has matured into a serious development env.

With recent CUDA versions, including the 12.x and emerging 13.x series, support for modern architectures like Hopper and Blackwell is fully integrated into workflows that run inside WSL. GPU access is stable, memory handling is improved, and integration with container systems has become more consistent.

In many cases, the difference between WSL and a native Linux setup is no longer as significant as it used to be.

At the same time, it is important to keep expectations realistic.

WSL is still a layered system. When sth goes wrong, the issue can come from multiple places: Windows config, WSL itself, the Linux distribution, or the CUDA setup.

This is part of the process.

Troubleshooting and understanding these layers is not a distraction. It is part of learning how the system actually works.

In the end, WSL represents a practical bridge.

It allows you to stay in a Windows env while working with Linux-based GPU tools in a way that is close to real production systems.

It is one of the most natural ways to start.

This was just for general info. The name “windows” will not be used in this repo under any circumstances.
