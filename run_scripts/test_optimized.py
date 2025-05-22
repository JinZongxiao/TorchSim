import time
import os
import torch

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:3950"

data_path_xyz = r'C:\Users\Thinkstation2\Desktop\MYMD_testdata\1.Liquid Argon\Ar_2.xyz'
# data_path_xyz = r'C:\Users\Thinkstation2\Desktop\MYMD_testdata\1.Liquid Argon\LiquidAr.xyz'
# data_path_xyz = r'C:\Users\Thinkstation2\Desktop\MYMD_testdata\1.Liquid Argon\Ar1e4.xyz'

from io_utils.reader import AtomFileReader
from io_utils.output_logger import Logger
import sys

param_pairs = {
                          #ev              #A
    '[0 0]' : {'epsilon': 0.0104, 'sigma': 3.405},
}

box_length = 17# 36.4# 80 #
cut_off = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"GPU Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

molecular = AtomFileReader(data_path_xyz,
                           box_length=box_length,
                           cutoff=cut_off,
                           device=device,
                           parameter=param_pairs,
                           skin_thickness=3.0)

print(f"Atom count: {molecular.atom_count}")
print(f"Atom types: {set(molecular.atom_types)}")

from core.force.lennard_jones_force import LennardJonesForce
lj_force = LennardJonesForce(molecular)

from core.md_model import SumBackboneInterface
force_field = SumBackboneInterface([lj_force], molecular)

from core.integrator.integrator import VerletIntegrator
vi = VerletIntegrator(molecular, 1000, lj_force, ensemble='NVT', temperature=[94.4,94.4], gamma=1)

from core.md_model import BaseModel
MDModel = BaseModel(lj_force, vi, molecular)

from core.md_simulation import MDSimulator
num_steps = 100
print_interval = 1
save_to_graph_dataset = False
Simulation = MDSimulator(MDModel, num_steps, print_interval, save_to_graph_dataset=save_to_graph_dataset)

output_dir = r"C:\\Users\\Thinkstation2\\Desktop\\MYMD_testdata"
sys.stdout = Logger(sys.stdout, log_dir=output_dir)
sys.stderr = Logger(sys.stderr, log_dir=output_dir)

USE_PROFILER = True
if USE_PROFILER:
    with torch.autograd.profiler.profile(enabled=True, use_cuda=True) as prof:
        Simulation.model()
    print(prof.table(sort_by="self_cuda_time_total"))
    prof.export_chrome_trace(f"{output_dir}/trace_{time.strftime('%Y%m%d_%H%M%S')}.json")

start_time = time.time()
Simulation.run(enable_minimize_energy=False)
end_time = time.time()
total_time = end_time - start_time
print(f"Total runtime: {total_time:.2f} seconds")
print(f"Time per step: {total_time/num_steps*1000:.2f} ms")

now_time = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())
output_prefix = f"{output_dir}/optimized_"

Simulation.save_energy_curve(f"{output_prefix}MD_energy_curve_{now_time}.png")
Simulation.save_xyz_trajectory(f"{output_prefix}MD_traj_{now_time}.xyz",
                              atom_types=molecular.atom_types)
Simulation.save_forces_grad(f"{output_prefix}forces_{now_time}.xyz",
                            with_no_ele=True,
                            atom_types=molecular.atom_types)
if save_to_graph_dataset:
    Simulation.save_graph_dataset(f"{output_prefix}MD_graph_dataset_{now_time}.pt") 