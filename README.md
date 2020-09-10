# The wavefront temporal blocking rabbit-hole

# Prerequisites:
  - A working devito installation https://www.devitoproject.org/devito/download.html
  - 
  
# Steps:

  - `cd devito`
  - `git checkout timetiling_on_cd`
  - Set env variables and pinning.
DEVITO_LANGUAGE=openmp
OMP_PROC_BIND=?
DEVITO_LOGGING=DEBUG

  - Run `DEVITO_JIT_BACKDOOR=0 python3 examples/seismic/acoustic/demo_temporal_sources.py -so 4` 
  This will generate code for a space order 4 acoustic devito kernel and another kernel that we will modify.
  You will notice difference between `norm(usol)` and `norm(uref)`. This is expected. Ignore that for now.
The generated log will end executing a kernel under `===Temporal blocking================================`

[![Build Status for the Core backend](https://github.com/devitocodes/devito/workflows/CI-core/badge.svg)](https://github.com/devitocodes/devito/actions?query=workflow%3ACI-core)
[![Build Status with MPI](https://github.com/devitocodes/devito/workflows/CI-mpi/badge.svg)](https://github.com/devitocodes/devito/actions?query=workflow%3ACI-mpi)
[![Build Status on GPU](https://github.com/devitocodes/devito/workflows/CI-gpu/badge.svg)](https://github.com/devitocodes/devito/actions?query=workflow%3ACI-gpu)
[![Code Coverage](https://codecov.io/gh/devitocodes/devito/branch/master/graph/badge.svg)](https://codecov.io/gh/devitocodes/devito)
[![Slack Status](https://img.shields.io/badge/chat-on%20slack-%2336C5F0)](https://join.slack.com/t/devitocodes/shared_invite/zt-gtd2yxj9-Y31YKk_7lr9AwfXeL2iMFg)
[![asv](http://img.shields.io/badge/benchmarked%20by-asv-blue.svg?style=flat)](https://devitocodes.github.io/devito-performance)
[![PyPI version](https://badge.fury.io/py/devito.svg)](https://badge.fury.io/py/devito)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/devitocodes/devito/master)

[Devito](http://www.devitoproject.org) is a Python package to implement
optimized stencil computation (e.g., finite differences, image processing,
machine learning) from high-level symbolic problem definitions.  Devito builds
on [SymPy](http://www.sympy.org/en/index.html) and employs automated code
generation and just-in-time compilation to execute optimized computational
kernels on several computer platforms, including CPUs, GPUs, and clusters
thereof.

- [About Devito](#about-devito)
- [Installation](#installation)
- [Resources](#resources)
- [Performance](#performance)
- [Get in touch](#get-in-touch)
- [Interactive jupyter notebooks](#interactive-jupyter-notebooks)

## About Devito

Devito provides a functional language to implement sophisticated operators that
can be made up of multiple stencil computations, boundary conditions, sparse
operations (e.g., interpolation), and much more.  A typical use case is
explicit finite difference methods for approximating partial differential
equations. For example, a 2D diffusion operator may be implemented with Devito
as follows

```python
>>> grid = Grid(shape=(10, 10))
>>> f = TimeFunction(name='f', grid=grid, space_order=2)
>>> eqn = Eq(f.dt, 0.5 * f.laplace)
>>> op = Operator(Eq(f.forward, solve(eqn, f.forward)))
```
===Temporal blocking======================================
Allocating memory for n(1,)
Allocating memory for usol(3, 236, 236, 236)
gcc -O3 -g -fPIC -Wall -std=c99 -march=native -Wno-unused-result -Wno-unused-variable -Wno-unused-but-set-variable -ffast-math -shared -fopenmp /tmp/devito-jitcache-uid1000/xxx-hash-xxx.c -lm -o /tmp/devito-jitcache-uid1000/xxx-hash-xxx.so
Operator `Kernel` jit-compiled `/tmp/devito-jitcache-uid1000/xxx-hash-xxx.c` in 0.48 s with `GNUCompiler`
Operator `Kernel` run in 0.49 s
```

Copy the space order 4 kernel from `devito/examples/seismic/acoustic/kernels/` to the `xxx-hash-xxx.c`
`cp kernels/kernel_so8_acoustic.c /tmp/devito-jitcache-uid1000/d3cee7726ce639b303537a387aa51cd42d9feecd.c`

Then try `DEVITO_JIT_BACKDOOR=1 python3 examples/seismic/acoustic/demo_temporal_sources.py -so 4`
to re-run. Now the norms should match. If not contact me ASAP :-).

Use arguments `-d nx ny nx` , `-tn timesteps` to pass as arguments domain size and number of timesteps.
e.g.:
`DEVITO_JIT_BACKDOOR=1 python3 examples/seismic/acoustic/demo_temporal_sources.py -d 200 200 200 --tn 100 -so 8`

There exist available kernels for space orders 4, 8, 12.

# Tuning Devito
Run Devito with `DEVITO_LOGGING=aggressive` so as to ensure that one of the best space-blocking configurations are selected.

# Tuning time-tiled kernel
In order to manually tune the Devito time-tiled kernel one should use an editor and jit-backdoor.
You can and should manually play around tile and block size values.
```
  int xb_size = 32;
  int yb_size = 32; // to fix as 8/16 etc
  int num_threads = 8;
  int x0_blk0_size = 8;
  int y0_blk0_size = 8;
```
You should also change accordingly the number of threads manually.
According to experiments x0_blk0_size, y0_blk0_size are best at 8
while xb_size, yb_size are nice in {32, 48, 64} depending on the platform.

`xb_size, yb_size == tiles`
`x0_blk0_size, y0_blk0_size == blocks`

When on Skylake: you may also need to change SIMD parallelism from 32 to 64.

As of YASK:
`From YASK: Although the terms "block" and "tile" are often used interchangeably, in
this section, we [arbitrarily] use the term "block" for spatial-only grouping
and "tile" when multiple temporal updates are allowed.`

Let me know your findings and your performance results. https://opesci-slackin.now.sh/ at #time-tiling

Depending on platform, I would expect speed-ups along the following lines:
![Perf results](https://github.com/devitocodes/devito/blob/timetiling_on_cd/examples/seismic/acoustic/kernels/temporal_performance.pdf)
