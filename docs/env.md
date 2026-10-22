# Environment fingerprint

Verified 2026-09-30 on the reference machine (the hardware under test).

> **This is the measurement-time snapshot, not the current machine.** As of 2026-10-22 the
> same box runs kernel 7.2.10-1 and Mesa 26.2.4 (RADV `STRIX_HALO`). Every tracked result
> was measured on the versions below; re-checked cells on the newer stack are 8–11% lower on
> `pp2048` with `tg512` unchanged (see [RUNNING.md §7](RUNNING.md)). Do not compare a new
> run against these tables without re-measuring the rest.
>
> **The GPU memory split has changed too.** `mem_info_vram_total` now reads **1 GiB** (the
> table below records the 96 GiB UMA carve-out the numbers were measured with), so model
> weights are now resident in **GTT**, not in the VRAM pool: `mem_info_vram_used` sits at
> ~0.1–1 GiB during a 59 GiB model's run while `MemAvailable` drops by the model's size.
> The sampler records both pools and the rollup publishes `mem_peak_gib` (VRAM + GTT);
> `vram_peak_gib` alone is not residency on this configuration. Restoring the carve-out
> restores comparability with the table below.

| Item | Value |
|---|---|
| Host | reference host |
| OS | CachyOS (Arch-based), kernel 7.2.8-1-cachyos, x86_64 |
| CPU | AMD Ryzen AI Max+ 395 (Strix Halo), 16C/32T, boost to 5.19 GHz |
| iGPU | Radeon 8060S (RDNA 3.5, 40 CUs), PCI `c5:00.0` |
| iGPU VRAM | 103,079,215,104 B (96 GiB) via `mem_info_vram_total`; ~30 GiB left to system |
| NPU | aie2p (XDNA 2) visible in rocminfo — **not used**, per README |
| Storage | `/dev/nvme0n1p1` 1.9 TB, ~1.6 TB free at fingerprint time |
| Vulkan | Mesa 26.2.3-arch3.1, RADV `STRIX_HALO`, ICD `radeon_icd.json` |
| ROCm | 7.2.4 at `/opt/rocm`; `rocminfo` sees **gfx1151 natively** (no `HSA_OVERRIDE_GFX_VERSION` needed) |
| llama.cpp | `~/.local/opt/llama.cpp` @ commit **5b59b83f4** (pinned, decision D2) |
|  · Backend A | `build-vulkan/` — `GGML_VULKAN=ON`, Release (reference) |
|  · Backend B | `build/` — `GGML_HIP=ON`, Release, `ROCWMMA_FATTN=ON`, `GRAPHS=ON`, `NO_VMM=ON` |
|  · (ref) | `build-probe/` — earlier HIP probe build |
| Models | `~/.local/share/models/` — 5 files, ~152 GB (see `matrix.yaml`) |
| Python | 3.14.7 (harness uses stdlib + CLI only; no `llama-cpp-python`) |
| Tools | git 2.55, cmake, gcc/g++, clang, hipcc/amdclang, vulkaninfo, rocm-smi |

## Power measurement
- **Source of truth during a run:** `scripts/sampler.py` reads the amdgpu **hwmon via sysfs**
  (resolved by name, not a hardcoded index) at 5 Hz to CSV per cell — `power1_average`
  (microwatts → W) and `temp1_input` (millidegrees → °C). No `rocm-smi` CLI per sample (fast).
- **Cross-check:** `rocm-smi --showpower --showtemp` → socket graphics package power
  (verified working, e.g. 54.6 W at 58% busy). Sample one cell against it to confirm the
  hwmon units line up.
- Caveat: GPU-only power. Under CPU-heavy load (prompt processing) SoC power is higher;
  `amd-power-tune` (platform level) not installed — revisit if tok/s/W needs the full picture.

## CPU power state (measurement condition)
**Decision 2026-10-02: leave the default state and record it**, rather than pinning
`performance` — the numbers then describe the box as it actually behaves, and the cell
already measured stays comparable.

| Item | Value | Read from |
|---|---|---|
| governor | `powersave` | `/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor` |
| EPP | `balance_performance` | `/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference` |
| driver | `amd_pstate=active` | `/sys/devices/system/cpu/amd_pstate/status` |
| power daemon | none (ppd / tlp / tuned all inactive) | `systemctl is-active` |

`run_sweep.py` stamps this into `run.json` (`cpu_power`) and `sweep.log` for every run.
Runs captured before 2026-10-02 — including `run-20261002-1719` — were taken in exactly
this state (it has not changed since), but only from 2026-10-02 is it recorded in the
run metadata itself.

## Contention
- A long-running `llama-server` can hold ~23 GB of VRAM. **Stop any such server before
  sweeps**; the driver verifies the GPU is idle before each cell and refuses to start otherwise.
