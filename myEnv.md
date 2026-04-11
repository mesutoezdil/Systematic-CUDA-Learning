# My Remote CUDA Dev Env (just for beginning)

> Stack: Cloud (L40S GPU) · CLion Remote SSH · GitHub · CUDA 13 · Ubuntu 24.04

## 1 > Architecture Overview

My setup uses a three-layer model. Each layer has a fixed, non-negotiable role:

```
┌──────────────────────────────────────────────────────┐
│              THE LOCAL MACHINE (Mac)                 │
│   CLion IDE — edit code, browse files, view UI       │
│   ⚠ Do NOT compile locally — editor only             │
└────────────────────┬─────────────────────────────────┘
                     │ SSH (port 22)
┌────────────────────▼─────────────────────────────────┐
│            CLOUD VM (<THE_VM_IP>)                    │
│   NVIDIA L40S GPU  — all compute happens here        │
│   nvcc / CUDA 13.0 — compile + run                   │
│   gdb              — debugging                       │
│   cmake            — build system                    │
│   ~/Systematic-CUDA-Learning/  ← git repo lives here │
└────────────────────┬─────────────────────────────────┘
                     │ git push (SSH)
┌────────────────────▼─────────────────────────────────┐
│            GITHUB — source of truth                  │
│   <the-username>/Systematic-CUDA-Learning            │
└──────────────────────────────────────────────────────┘
```

> Golden rule: Never compile on your local machine. Write code → build on the VM → run on the VM. Local = editor only.

## 2 > Cloud (Creating the GPU VM)

### Recommended Config (for beginning)

| Field   | Value                                      |
|---------|--------------------------------------------|
| GPU     | NVIDIA L40S Intel                          |
| CPU     | 8 vCPU                                     |
| RAM     | 32 GB                                      |
| Disk    | 80 GB SSD                                  |
| OS      | Ubuntu 24.04 LTS for NVIDIA GPUs (CUDA 13) |
| Network | Auto-assign dynamic public IP              |

### What to avoid

- H100 NVLink > overkill and expensive for learning. L40S is the right starting point.
- 1000+ GiB disk > CUDA kernels and experiments use less than 10 GB. 80 GB is generous.
- No public IP > you won't be able to SSH in at all.

> Cost tip: Always STOP the VM when you're done for the day, don't DELETE it. STOP = GPU halts = billing stops. DELETE = everything is gone.

## 3 > SSH Connection Setup

Once the VM is running, grab the Public IP from the Cloud panel.

### First connection

```bash
ssh ubuntu@<THE_VM_PUBLIC_IP>

# You'll see a fingerprint prompt — type "yes"
# Successful output:
# Welcome to Ubuntu 24.04.4 LTS (GNU/Linux 6.11.0-1016-nvidia x86_64)
# ubuntu@computeinstance-...:~$
```

### Load your SSH key into the agent

```bash
# See available keys
ls ~/.ssh

# Add key to agent
ssh-add ~/.ssh/id_ed25519

# Verify
ssh-add -l
```

### Prevent wrong-key issues with `~/.ssh/config`

If you have multiple SSH keys (e.g., one for work, one personal), create a config file to pin which key is used for GitHub:

```
# ~/.ssh/config
Host github.com
  HostName       github.com
  User           git
  IdentityFile   ~/.ssh/id_ed25519
  IdentitiesOnly yes
```

## 4 > VM Env Setup

SSH into the VM and run these in order:

### Step 1 > Update packages and install essentials

```bash
sudo apt update
sudo apt install -y gdb build-essential cmake
```

### Step 2 > Verify the GPU

```bash
nvidia-smi

# Expected output:
# | 0  NVIDIA L40S     On  | 00000000:8D:00.0 Off |
# | N/A  30C  P8  35W / 325W | 0MiB / 46068MiB |
```

### Step 3 > Verify the CUDA toolchain

```bash
which nvcc
# /usr/local/cuda-13.0/bin/nvcc

nvcc --version
# Cuda compilation tools, release 13.0, V13.0.88

gdb --version
# GNU gdb 15.1
```

### Step 4 > Create the nvcc symlink

CMake looks for `/usr/local/cuda` by default. Fix this with a symlink:

```bash
sudo ln -s /usr/local/cuda-13.0 /usr/local/cuda

# Verify
ls -la /usr/local/cuda
# lrwxrwxrwx ... /usr/local/cuda -> /usr/local/cuda-13.0
```

## 5 > Repo Structure & CMake

### Directory layout

The key principle here is out-of-source builds: source code and compiled output are always kept separate.

```
Systematic-CUDA-Learning/
├── CMakeLists.txt            ← root build config
├── .gitignore                ← build/ is listed here
├── CUDA/
│   ├── 00/
│   │   ├── code/
│   │   │   ├── first_kernel.cu    ← the code you write
│   │   │   └── CMakeLists.txt     ← build config for this lesson
│   │   └── notes.md               ← learning notes
│   ├── 01/
│   │   ├── code/
│   │   │   ├── main.cu
│   │   │   └── CMakeLists.txt
│   │   └── notes.md
│   └── 02/ ...
└── build/                    ← in .gitignore, never committed
    ├── CUDA/00/code/run_00   ← compiled binary you run
    └── CUDA/01/code/run_01
```

> Rule: `CUDA/` = what you write, what goes to git. `build/` = what the machine produces, never goes to git.

### Clone the repo (inside the VM)

```bash
git clone https://github.com/<the-username>/Systematic-CUDA-Learning.git
cd Systematic-CUDA-Learning
```

### Root `CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.18)
project(cuda_learning LANGUAGES CUDA)

add_subdirectory(CUDA/00/code)
add_subdirectory(CUDA/01/code)
# Add a new line here for each new lesson
```

### Per-lesson `CMakeLists.txt` (example: lesson 00)

```cmake
# CUDA/00/code/CMakeLists.txt
cmake_minimum_required(VERSION 3.18)
project(cuda_00 LANGUAGES CUDA)

add_executable(run_00 first_kernel.cu)

set_target_properties(run_00 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
```

### `.gitignore`

```gitignore
build/
cmake-build-debug/
cmake-build-release/
*.o
*.a
*.out
```

> If you accidentally committed `build/`:
> ```bash
> git rm -r --cached build
> git commit -m "fix: remove build artifacts"
> git push
> ```

### First build

```bash
mkdir -p build && cd build

cmake ..
cmake --build .

# Expected output:
# [ 50%] Built target run_00
# [100%] Built target run_01

# Run it
./CUDA/00/code/run_00
# Hello from GPU!
```

## 6 > CLion Remote Dev

CLion's Remote SSH mode connects directly to the VM, all files stay on the VM and CLion acts as the UI only.

### Step 1 > Open Remote Dev

CLion Welcome Screen → Remote Development → SSH → add connection:

```
Host:      <THE_VM_PUBLIC_IP>
Port:      22
Username:  ubuntu
Auth:      OpenSSH config and authentication agent
```

Test the connection. If it fails, make sure you've run `ssh-add ~/.ssh/id_ed25519` first.

### Step 2 > Open the remote project

After connecting → Open Remote Project → path:

```
/home/ubuntu/Systematic-CUDA-Learning
```

### Step 3 > Configure the Toolchain

"Settings → Build, Execution, Deployment → Toolchains"

Select "Remote Host". CLion will auto-detect:

```
C Compiler:    /usr/bin/gcc
C++ Compiler:  /usr/bin/g++
Debugger:      /usr/bin/gdb
```

### Step 4 > Set CMake options

Settings → Build, Execution, Deployment → CMake → CMake options:

```
-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

### Step 5 > Create a Run Config

Top bar → Current File ▼ → Edit Configurations → + → CMake Application:

```
Name:       run_00
Target:     run_00
Executable: run_00   (auto-filled)
```

> Sanity check: Open CLion's built-in terminal and run `nvidia-smi`. If you see the L40S, you're in the right env.

## 7 > GitHub SSH Authentication (VM → GitHub)

GitHub no longer accepts passwords over HTTPS. You must set up SSH from the VM.

### Step 1 > Generate an SSH key inside the VM

```bash
ssh-keygen -t ed25519 -C "the@email.com"
# Press Enter three times (accept defaults, no passphrase)

# Print the public key
cat ~/.ssh/id_ed25519.pub
# Copy this output
```

### Step 2 > Add the public key to GitHub

Go to github.com → Settings → SSH and GPG Keys → New SSH Key, paste the output.

### Step 3 > Create SSH config to avoid key conflicts

```bash
vim ~/.ssh/config
```

```
Host github.com
  HostName       github.com
  User           git
  IdentityFile   ~/.ssh/id_ed25519
  IdentitiesOnly yes
```

### Step 4 > Load the key and test

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# Test
ssh -T git@github.com
# Hi <the-username>! You've successfully authenticated...
```

### Step 5 > Switch remote URL to SSH and push

```bash
cd ~/Systematic-CUDA-Learning

git remote set-url origin git@github.com:<the-username>/Systematic-CUDA-Learning.git

git config --global user.name "The Name"
git config --global user.email "the@email.com"

git add .
git commit -m "setup: cuda remote dev environment"
git push
# To github.com:<the-username>/Systematic-CUDA-Learning.git
#    abc1234..def5678  main -> main
```

## 8 > Final Env Check

Run all of these inside the VM. Everything should pass:

```bash
# 1. GPU visible?
nvidia-smi
# ✓ NVIDIA L40S, 0MiB usage

# 2. CUDA compiler path correct?
which nvcc && nvcc --version
# ✓ /usr/local/cuda-13.0/bin/nvcc
# ✓ release 13.0, V13.0.88

# 3. Debugger available?
gdb --version
# ✓ GNU gdb 15.1

# 4. Symlink in place?
ls -la /usr/local/cuda
# ✓ /usr/local/cuda -> /usr/local/cuda-13.0

# 5. Build works?
cd ~/Systematic-CUDA-Learning/build && cmake --build .
# ✓ [100%] Built target run_00
# ✓ [100%] Built target run_01

# 6. CUDA actually runs on the GPU?
./CUDA/00/code/run_00
# ✓ Hello from GPU!

# 7. Git push works?
git push
# ✓ main -> main

# 8. Repo clean?
git status
# ✓ nothing to commit, working tree clean
```

## 9 > Daily Workflow

### Session start

```bash
# 1. Start the VM from the Cloud panel

# 2. SSH in
ssh ubuntu@<THE_VM_PUBLIC_IP>

# 3. Load SSH key (required every new session)
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# 4. Go to repo
cd ~/Systematic-CUDA-Learning
```

### Dev loop

```bash
# Edit code (via CLion or vim)
# CUDA/0X/code/main.cu

# Rebuild
cd build && cmake --build .

# Run
./CUDA/0X/code/run_0X

# Write notes
vim ~/Systematic-CUDA-Learning/CUDA/0X/notes.md

# Commit and push
cd ~/Systematic-CUDA-Learning
git add . && git commit -m "lesson 0X: <what you learned>" && git push
```

### Session end

```bash
# Stop the VM from the Cloud panel.
```

### Adding a new lesson

```bash
# Create the folder structure
mkdir -p CUDA/03/code
touch CUDA/03/notes.md

# Add a CMakeLists.txt for the new lesson
cat << 'EOF' > CUDA/03/code/CMakeLists.txt
cmake_minimum_required(VERSION 3.18)
project(cuda_03 LANGUAGES CUDA)

add_executable(run_03 main.cu)
set_target_properties(run_03 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
EOF

# Register it in the root CMakeLists.txt
echo "add_subdirectory(CUDA/03/code)" >> CMakeLists.txt

# Rebuild
cd build && cmake .. && cmake --build .
```

### `notes.md` template

```markdown
# CUDA 0X
## Output
```
Hello from GPU!
```

## Key Concepts
- `threadIdx.x` —> each thread's ID within a block
- `blockIdx.x`  —> each block's ID within the grid
- `<<<grid, block>>>` —> kernel launch config
```

## 10 > Common Errors & Fixes

| Error                                   | Cause                                                | Fix                                                                          | 
|-----------------------------------------|------------------------------------------------------|------------------------------------------------------------------------------|
| `Authentication failed` (HTTPS push)    | GitHub no longer accepts passwords                   | Set up SSH key and switch remote URL to `git@github.com:...`                 |
| `Permission denied to <wrong-username>` | Wrong SSH key is being selected                      | Add `IdentityFile` to `~/.ssh/config`, run `ssh-add`                         |
| `nvcc: command not found`               | PATH missing or symlink absent                       | `sudo ln -s /usr/local/cuda-13.0 /usr/local/cuda`                            |
| CLion toolchain red / undetected        | `gdb` not installed or SSH agent not running         | `sudo apt install gdb` + `ssh-add ~/.ssh/id_ed25519`                         |
| `CMake: No SOURCES given to target`     | `CMakeLists.txt` points to a file that doesn't exist | Create a placeholder `main.cu` or remove that `add_subdirectory` line        |
| `build/` appeared in git diff           | `.gitignore` added after first commit                | `git rm -r --cached build && git commit -m "fix: remove build artifacts"`    |
| `error: expected a ";"` in `.cu` file   | Placeholder text like `soon...` in source file       | Replace file contents with valid CUDA code                                   |
| CLion SSH timeout / won't connect       | SSH agent key not loaded                             | Change Auth type to "OpenSSH config and authentication agent" in CLion       |

## My Architecture

| Layer           | Role                                      |
|-----------------|-------------------------------------------|
| Local machine   | Editor only -> CLion UI, file browsing    |
| Cloud VM        | All compute -> compile, run, debug, git   |
| GitHub          | Source of truth -> every commit goes here |

Do not compile locally. Do not commit `build/`. Stop the VM when done.
