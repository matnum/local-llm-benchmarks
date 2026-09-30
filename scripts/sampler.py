#!/usr/bin/env python3
"""Resource sampler: poll GPU power/temp/busy, VRAM, system memory at HZ to CSV.

Usage: sampler.py OUT_CSV [HZ]

Reads sysfs only (fast; no rocm-smi CLI per sample).

I4: the amdgpu hwmon is resolved BY NAME, not by a hardcoded hwmon index
(numbering is not stable across reboots/driver updates). The run fails loudly at
start if the amdgpu hwmon entry or any required sensor file is missing. A
per-sample GPU read failure is logged to stderr and that sample is SKIPPED (row
omitted) rather than silently defaulting to 0 — silent zero power in a benchmark
trace is worse than a flagged gap.

I1: temp1_input is in MILLI-degrees (raw 64000 -> 64.0 C), power1_average in
MICROWATTS. Cross-check one sample against `rocm-smi --showpower --showtemp`
per cell.
"""
import csv
import glob
import os
import sys
import time

HWMON_ROOT = "/sys/class/hwmon"
DRM = "/sys/class/drm/card0/device"
GIB = 1024 ** 3


def resolve_hwmon(name="amdgpu"):
    """Return the /sys/class/hwmon/hwmonN dir whose name == `name`, else None."""
    for d in sorted(glob.glob(f"{HWMON_ROOT}/hwmon*")):
        try:
            with open(os.path.join(d, "name")) as f:
                if f.read().strip() == name:
                    return d
        except OSError:
            continue
    return None


def rd(path):
    """Read an int sensor file; return None on any read error (caller decides)."""
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "resources.csv"
    hz = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    dt = 1.0 / hz

    hw = resolve_hwmon("amdgpu")
    if hw is None:
        sys.exit("sampler: no hwmon entry named 'amdgpu' under /sys/class/hwmon — "
                 "refusing to sample (hwmon indexing may have changed; see I4)")
    power_file = f"{hw}/power1_average"
    temp_file = f"{hw}/temp1_input"
    for req in (power_file, temp_file,
                f"{DRM}/gpu_busy_percent", f"{DRM}/mem_info_vram_used"):
        if not os.path.isfile(req):
            sys.exit(f"sampler: required sensor file missing: {req}")
    print(f"[sampler] hwmon={hw} @ {hz:.2f} Hz -> {out}", file=sys.stderr)

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "power_w", "temp_c", "gpu_busy_pct", "vram_used_gib", "sys_avail_gib"])
        while True:
            p = rd(power_file)
            t = rd(temp_file)
            b = rd(f"{DRM}/gpu_busy_percent")
            v = rd(f"{DRM}/mem_info_vram_used")
            # I4: any failed GPU read -> warn + skip the sample (no silent zeros).
            failed = [name for name, val in
                      (("power", p), ("temp", t), ("busy", b), ("vram", v)) if val is None]
            if failed:
                sys.stderr.write(f"[sampler] t={time.time():.1f} read failed "
                                 f"({', '.join(failed)}); skipping sample\n")
                time.sleep(dt)
                continue
            assert p is not None and t is not None and b is not None and v is not None
            try:
                with open("/proc/meminfo") as mi:
                    avail = int(next(l for l in mi if l.startswith("MemAvailable")).split()[1]) / (1024 ** 2)
            except Exception:
                avail = -1.0
            w.writerow([f"{time.time():.3f}",
                        f"{p / 1e6:.2f}",     # microwatts -> W
                        f"{t / 1000.0:.1f}",  # millidegrees -> C (I1)
                        b, f"{v / GIB:.2f}", f"{avail:.2f}"])
            f.flush()
            time.sleep(dt)


if __name__ == "__main__":
    main()
