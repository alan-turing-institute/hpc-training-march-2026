# One-Hour Training: Containers on HPC with Apptainer

## Session Details
- Duration: 60 minutes
- Audience: HPC users running Python/ML/science workloads on Isambard-AI (or similar systems)
- Assumed knowledge: Linux shell basics, Slurm basics, basic idea of Docker images
- Format: 35 minutes guided demo + 20 minutes hands-on + 5 minutes wrap-up

## Learning Outcomes
By the end of the session, participants should be able to:
- Explain when Apptainer is a better choice than bare-metal environments or Docker runtime on HPC.
- Pull/build/run a container image for interactive and batch jobs.
- Use bind mounts and `--fakeroot` safely.
- Run GPU and multi-node workloads using host MPI/NCCL integration.
- Avoid common performance and portability pitfalls.

## Why Apptainer on HPC (And When It Is a Good Choice)
Use Apptainer when you need:
- Reproducibility: fixed user-space stack (Python/CUDA libs/framework versions).
- Portability: move one image between login and compute nodes or between sites.
- User-level operation: no Docker daemon and no root daemon model required.
- Compatibility with OCI sources: pull from Docker registries and run as SIF.
- Safer multi-user HPC operation: integrates better with shared clusters and schedulers.

Less ideal when:
- You need frequent in-container mutation in production (prefer immutable images + rebuild).
- You can use curated site modules with no dependency conflicts (containers may add overhead/complexity).
- Your workflow depends on architecture-unsupported images (for Isambard, check `aarch64` support).

## 60-Minute Agenda
### 0-5 min: Framing
- What containers do and what they do not do on HPC.
- Apptainer/Singularity naming: command may be `singularity` on site docs, concepts map directly to Apptainer.

### 5-15 min: Core Commands and Mental Model
- `build` from OCI source to `.sif`.
- `run` (entrypoint/runscript) vs `exec` (explicit command) vs `shell` (interactive).
- Read-only SIF image + host bind mounts for data.

Demo commands:
```bash
mkdir -p $HOME/sif-images && cd $HOME/sif-images
singularity build ubuntu.sif docker://ubuntu:24.04
singularity shell ubuntu.sif
singularity exec ubuntu.sif grep PRETTY /etc/os-release
```

### 15-25 min: Building for Real Workloads
- Choosing a base image (e.g., NGC PyTorch if GPU ML stack is needed).
- Definition files (`.def`) for repeatable builds.
- Rootless builds with `--fakeroot` when package install steps are needed.

Demo commands:
```bash
singularity shell --fakeroot ubuntu.sif
# in container:
whoami
```

### 25-35 min: GPUs, Binds, and Batch Jobs
- GPU pass-through with `--nv`.
- Bind behavior and explicit binds for scratch/work dirs (`--bind`).
- Batch job pattern using `sbatch` / `srun`.

Demo commands:
```bash
srun --gpus=1 --ntasks=1 --time=5 \
  singularity exec --nv ubuntu.sif nvidia-smi --list-gpus
```

### 35-50 min: Multi-Node Performance Pattern (Key Section)
- Why special setup is required: high performance needs NIC visibility and host MPI/NCCL integration.
- Isambard pattern:
  - `module load brics/apptainer-multi-node`
  - run container with `/host/adapt.sh`
  - this wires `/host/` libs and env for MPI/NCCL/Slingshot performance

Canonical command pattern:
```bash
module load brics/apptainer-multi-node
srun -N 2 --gpus 8 --ntasks-per-node 1 --cpus-per-task 72 \
  singularity exec --nv --bind $PWD:$PWD pytorch_25.05-py3.sif \
  /host/adapt.sh bash -lc "python train.py --config conf.yaml"
```

If dropping into shell first:
```bash
singularity exec --nv image.sif /host/adapt.sh bash
```

### 50-55 min: Gotchas and Debugging Checklist
- Architecture mismatch (`x86_64` image on Arm): verify image supports `aarch64`.
- Forgetting `--nv`: GPU not visible in container.
- Forgetting `/host/adapt.sh` on multi-node: poor or broken MPI/NCCL behavior.
- Running with `run` when you needed `exec`: wrong entrypoint behavior.
- Missing binds: code/data not visible where expected.
- Writing heavy temp files to shared home: redirect to `$TMPDIR`/scratch.
- Image pulls at scale during job start: pre-stage `.sif` before large runs.

### 55-60 min: Recommended Working Patterns
- Development loop:
  - iterate in a sandbox/dev context
  - encode into `.def`
  - build immutable `.sif`
  - run with explicit command + explicit binds
- Team reproducibility:
  - version `.def` in git
  - tag images by date/version
  - record exact launch command in job scripts
- Performance-first launch:
  - use site multi-node module setup
  - keep container startup simple
  - benchmark communication (e.g., NCCL tests) before long training runs

## Hands-On Exercise (20 min within session)
Participants do:
1. Pull/build a base image from Docker registry.
2. Run `shell`, `run`, and `exec` and explain differences.
3. Execute a GPU visibility check with `--nv`.
4. Launch a 2-node smoke test with site multi-node adaptation.
5. Capture one “gotcha” they hit and how they fixed it.

## Instructor Notes
- Keep all examples in `singularity` CLI to match site docs; mention `apptainer` naming equivalence once.
- Emphasize that containers package user space, not the kernel/interconnect stack.
- Reuse same training code as prior distributed training lesson so focus stays on container mechanics.

## Optional Appendix: Minimal `.def` Skeleton
```def
Bootstrap: docker
From: nvcr.io/nvidia/pytorch:25.05-py3

%post
    pip install --no-cache-dir -U pip
    pip install --no-cache-dir pyyaml

%runscript
    exec python -u train.py "$@"
```

## Source References
- Isambard Singularity guide: https://docs.isambard.ac.uk/user-documentation/guides/containers/singularity/
- Isambard containers overview: https://docs.isambard.ac.uk/user-documentation/guides/containers/
- Isambard Singularity multi-node guide: https://docs.isambard.ac.uk/user-documentation/guides/containers/apptainer-multi-node/
