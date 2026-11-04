"""Preserve replay values/RNG while changing scheduling resources and batch assembly."""
import ast
import hashlib
from pathlib import Path

base = Path(__file__).resolve().parent
source = base / 'recover_one.py'
assert hashlib.sha256(source.read_bytes()).hexdigest() == '7a9a3d3287581626f94143f080116b637a46ae0039e0441421e3ca0dd08533b6'
text = source.read_text()
text = text.replace('available < 10 * 1024**2', 'available < 8 * 1024**2')
text = text.replace("original = torch.load(record['downstream_weight_path']",
                    "from fast_batch_fetch import install_contiguous_batch_fetch\n"
                    "install_contiguous_batch_fetch(runner.train_data)\n"
                    "install_contiguous_batch_fetch(runner.dev_data)\n"
                    "original = torch.load(record['downstream_weight_path']")
text = text.replace('runner.test_data = runner.data_loader.load_test_data()',
                    'runner.test_data = runner.data_loader.load_test_data()\n'
                    'install_contiguous_batch_fetch(runner.test_data)')
text = text.replace("'retained_runtime_zip': digest(BASE / 'runtime.zip')",
                    "'retained_runtime_zip': digest(BASE / 'runtime.zip'), "
                    "'batch_fetch': digest(BASE / 'fast_batch_fetch.py')")
text = text.replace("'gpu_allocator_fraction': .08, 'peak_gpu_allocated_bytes'",
                    "'gpu_allocator_fraction': .08, 'minimum_available_ram_gib': 8, "
                    "'cpu_threads': 1, 'cpu_affinity': sorted(os.sched_getaffinity(0)), "
                    "'niceness': os.getpriority(os.PRIO_PROCESS, 0), 'peak_gpu_allocated_bytes'")
ast.parse(text)
(base / 'recover_one_parallel.py').write_text(text)
for name in ('parallel_queue.py', 'handover_to_parallel.py', 'fast_batch_fetch.py'):
    ast.parse((base / name).read_text())
print(hashlib.sha256((base / 'recover_one_parallel.py').read_bytes()).hexdigest())
names = ('recover_one_parallel.py', 'parallel_queue.py', 'parallel_config.json',
         'handover_to_parallel.py', 'launch_parallel.sh', 'fast_batch_fetch.py')
(base / 'parallel_deployment.sha256').write_bytes(''.join(
    hashlib.sha256((base / name).read_bytes()).hexdigest() + '  ' + name + '\n' for name in names).encode())
(base / 'parallel_batch_fetch.sha256').write_bytes((
    hashlib.sha256((base / 'recover_one_parallel.py').read_bytes()).hexdigest() + '  recover_one_parallel.py.next\n' +
    hashlib.sha256((base / 'fast_batch_fetch.py').read_bytes()).hexdigest() + '  fast_batch_fetch.py\n').encode())
