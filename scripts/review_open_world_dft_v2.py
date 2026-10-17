"""O2 single-item runner, version 2: record the GPU through the CUDA runtime.

Version 1 (scripts/review_open_world_dft.py, frozen at bb09c34) runs
`nvidia-smi --query-gpu=name,driver_version --format=csv,noheader` with
check=True before any quantum-chemistry stage, only to write environment.json.
Since 2026-10-10 the n2 NVML user-space library (595.99) has not matched the
loaded kernel module (595.91.07), so nvidia-smi exits 18 although CUDA works:
on 2026-10-17 Slurm job 3371 recomputed the stored 2752 anchor energy with the
frozen qc_protocol and agreed to 5.7e-14 Hartree.

The owner approved this versioned change on 2026-10-17, and nothing else:
- that one query is answered from the CUDA runtime (cupy): the name of each
  visible device and the loaded kernel-module version read from
  /proc/driver/nvidia/version, tagged `source=cuda-runtime`;
- a small CUDA computation must succeed first; otherwise the runner fails
  before any QC stage, exactly as a failing nvidia-smi did.
All other behaviour is version 1 itself: this file imports it unchanged and
runs its main(). qc_protocol, the frozen plan and sample, SCF/optimizer
settings, input-hash checks and the clean-source and Slurm guards are untouched.

`gpu-check` prints the same GPU record without touching any plan item; the
version-4 dispatcher runs it inside Slurm as its precheck.
"""
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_open_world_dft as v1  # noqa: E402

# The exact argument list version 1 passes to subprocess.run (see v1.run).
NVIDIA_SMI_QUERY = ('nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader')
PROC_DRIVER_VERSION = Path('/proc/driver/nvidia/version')


def kernel_module_version(text):
    """Driver version from /proc/driver/nvidia/version (the value nvidia-smi reported).

    Example first line: 'NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  595.91.07  Release Build ...'.
    """
    match = re.search(r'NVRM version:.*?\s(\d+\.\d+(?:\.\d+)?)\s', text)
    if not match:
        raise ValueError('Unrecognised /proc/driver/nvidia/version')
    return match.group(1)


def cuda_gpu_info(cupy=None, proc_driver_version=PROC_DRIVER_VERSION):
    """One 'name, driver_version, source=...' line per visible device.

    Raises when no device is visible or a trivial CUDA reduction gives a wrong
    answer, so an unusable GPU still stops the item before any QC stage.
    """
    if cupy is None:
        import cupy
    runtime = cupy.cuda.runtime
    count = runtime.getDeviceCount()
    if count < 1:
        raise RuntimeError('No visible CUDA device')
    if int(cupy.arange(8).sum()) != 28:
        raise RuntimeError('CUDA self-test returned a wrong sum')
    driver = kernel_module_version(Path(proc_driver_version).read_text())
    source = f'source=cuda-runtime (cupy {cupy.__version__}, CUDA driver API {runtime.driverGetVersion()})'
    lines = []
    for index in range(count):
        name = runtime.getDeviceProperties(index)['name']
        lines.append(f"{name.decode() if isinstance(name, bytes) else name}, {driver}, {source}")
    return '\n'.join(lines)


class SubprocessWithCudaGpuQuery:
    """Stand-in for version 1's `subprocess` module.

    Answers only the exact nvidia-smi GPU query from `info()`; every other call
    (e.g. the git checks in v1.source_state) goes to the real module.
    """

    def __init__(self, real=subprocess, info=cuda_gpu_info):
        self._real, self._info = real, info

    def run(self, args, *pargs, **kwargs):
        if tuple(args) == NVIDIA_SMI_QUERY:
            return self._real.CompletedProcess(list(args), 0, stdout=self._info() + '\n', stderr='')
        return self._real.run(args, *pargs, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)


def main():
    if sys.argv[1:] == ['gpu-check']:
        print(cuda_gpu_info(), flush=True)
        return
    v1.subprocess = SubprocessWithCudaGpuQuery()
    v1.main()


if __name__ == '__main__':
    main()
