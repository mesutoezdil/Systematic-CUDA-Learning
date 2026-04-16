# Installing CUDA Toolkit on Linux

This is where most people waste time.

First rule:

You don’t always install CUDA.

## 1. Start with one question: what environment are you in?

### If you are using a cloud GPU VM

If you selected an image like:

* Ubuntu for NVIDIA GPUs
* any image that mentions CUDA

Do not install CUDA.

It is already installed.

This is the setup you’re using.

What you should do instead:

```bash
nvidia-smi
```

```bash
which nvcc
nvcc --version
```

If these work, you’re done. Move on.

Reinstalling CUDA here usually just breaks things.

## 2. If you are on local Linux / WSL / bare metal

This is the only case where you actually install CUDA.

## Step 1 > Check GPU first

```bash
nvidia-smi
```

If this fails, stop.

CUDA won’t fix it. Your driver setup is broken.

## Step 2 > Install CUDA (WSL example)

Do not use:

```bash
apt install nvidia-cuda-toolkit
```

It’s outdated.

Use NVIDIA’s repo:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-13-2
```

## Step 3 > Verify

```bash
nvcc --version
```

If that works, you’re good.

---

## Step 4 > Fix PATH (only if needed)

```bash
export PATH=/usr/local/cuda/bin:$PATH
```

## What actually got installed

* `nvcc` (compiler)
* CUDA runtime
* core libraries

It does NOT install GPU drivers.

## Reality check

CUDA is tightly tied to hardware.

If your version is too old, your code still runs, just not efficiently.

