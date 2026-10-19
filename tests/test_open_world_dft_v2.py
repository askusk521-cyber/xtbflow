"""Version-2 runner and version-4 dispatcher: GPU record via the CUDA runtime only."""
import inspect
import subprocess
import sys
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import review_open_world_dft as v1  # noqa: E402
import review_open_world_dft_v2 as v2  # noqa: E402
from review_open_world_dispatch_v4 import item_wrap, precheck_command  # noqa: E402

PROC = ('NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  595.91.07  Release Build  '
        '(dvs-builder@U22-I3-B08-02-2)  Wed Jul 29 03:01:16 UTC 2026\nGCC version:  gcc version 12.3.0\n')


def fake_cupy(count=1, total=28):
    runtime = SimpleNamespace(getDeviceCount=lambda: count, driverGetVersion=lambda: 13020,
                              getDeviceProperties=lambda i: {'name': b'NVIDIA RTX PRO 6000 Blackwell'})
    return SimpleNamespace(__version__='14.2.0', cuda=SimpleNamespace(runtime=runtime),
                           arange=lambda n: SimpleNamespace(sum=lambda: total))


def test_kernel_module_version_matches_what_nvidia_smi_reported():
    assert v2.kernel_module_version(PROC) == '595.91.07'
    with pytest.raises(ValueError):
        v2.kernel_module_version('no driver here')


def test_cuda_gpu_info_keeps_the_name_driver_prefix_and_tags_its_source(tmp_path):
    (tmp_path / 'version').write_text(PROC)
    line = v2.cuda_gpu_info(fake_cupy(), tmp_path / 'version')
    assert line == ('NVIDIA RTX PRO 6000 Blackwell, 595.91.07, '
                    'source=cuda-runtime (cupy 14.2.0, CUDA driver API 13020)')


@pytest.mark.parametrize('cupy', [fake_cupy(count=0), fake_cupy(total=0)])
def test_unusable_gpu_still_fails_before_any_qc(tmp_path, cupy):
    (tmp_path / 'version').write_text(PROC)
    with pytest.raises(RuntimeError):
        v2.cuda_gpu_info(cupy, tmp_path / 'version')


def test_shim_answers_only_the_exact_query_and_delegates_the_rest():
    calls = []
    real = SimpleNamespace(CompletedProcess=subprocess.CompletedProcess,
                           run=lambda *a, **k: calls.append(('run', a, k)) or 'real',
                           check_output=lambda *a, **k: calls.append(('check_output', a, k)) or 'abc\n')
    shim = v2.SubprocessWithCudaGpuQuery(real, info=lambda: 'GPU, 595.91.07, source=cuda-runtime')
    out = shim.run(list(v2.NVIDIA_SMI_QUERY), capture_output=True, text=True, check=True)
    assert out.returncode == 0 and out.stdout.strip() == 'GPU, 595.91.07, source=cuda-runtime'
    assert calls == []
    assert shim.run(['git', 'status']) == 'real'
    assert shim.check_output(['git', 'rev-parse', 'HEAD'], text=True) == 'abc\n'
    assert [c[0] for c in calls] == ['run', 'check_output']


def test_query_constant_is_exactly_what_the_frozen_runner_calls():
    source = inspect.getsource(v1.run)
    assert "subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader']," in source
    assert list(v2.NVIDIA_SMI_QUERY) == ['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader']


def test_dispatcher_v4_uses_the_v2_runner_in_slurm(tmp_path):
    pre = precheck_command(PurePosixPath('/src'))
    assert pre[:3] == ['srun', '--partition=main', '--gres=gpu:pro6000:1']
    assert 'nvidia-smi' not in ' '.join(pre) and pre[-1].endswith('scripts/review_open_world_dft_v2.py gpu-check')
    wrap = item_wrap(PurePosixPath('/src'), PurePosixPath('/root'), 'abcd')
    assert 'scripts/review_open_world_dft_v2.py run --plan /root/o2-plan.json --item abcd --work /root/o2-run' in wrap
