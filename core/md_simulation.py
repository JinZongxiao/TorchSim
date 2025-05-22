import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:3950"

import torch
from torch.amp import autocast

from typing import Dict, List
import time
from matplotlib import rcParams
from torch_geometric.data import Data
from graph_diffusion.graph_utils import calc_rho

from core.energy_minimizer import minimize_energy_bfgs_scipy

matplotlib_config = {
    'font.family': 'Times New Roman',
    'axes.unicode_minus': False,
}
rcParams.update(matplotlib_config)


class MDSimulator:

    def __init__(self, model, num_steps: int, print_interval: int = 10, save_to_graph_dataset = False):
        self.model = model
        self.num_steps = num_steps
        self.print_interval = print_interval

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.model.to(self.device)

        self.save_to_graph_dataset = save_to_graph_dataset
        if self.save_to_graph_dataset:
            self.dataset = []

        self.trajectory = []  
        self.energy_list = []  
        self.force_list = []  

        self.gpu_trajectory = torch.empty((num_steps, model.molecular.atom_count, 3), device='cuda')
        self.gpu_forces = torch.empty((num_steps, model.molecular.atom_count, 3), device='cuda')
        self.gpu_energies = torch.empty(num_steps, device='cuda')

    def run(self, enable_minimize_energy: bool = True):
        start_time = time.time()

        if self.device == torch.device('cuda'):
            gpu_name = torch.cuda.get_device_name(self.device.index if self.device.index is not None else 0)
            print(f"Simulation will run on GPU: {gpu_name}")
        else:
            print("Simulation will run on CPU")

        if enable_minimize_energy:
            print("=== Minimizing energy before simulation ===")
            minimize_energy_bfgs_scipy(self.model)
            print("=== Energy Minimization Completed ===\n")
        self.model = self.model.to(torch.float16)
        for step in range(self.num_steps):
            with autocast('cuda'):
                out = self.model()

            self.gpu_forces[step] = out['forces']
            self.gpu_energies[step] = out['energy']
            self.gpu_trajectory[step] = out['updated_coordinates']
            rho = calc_rho(self.model.molecular.graph_data,self.model.molecular.box_length)
            if (step + 1) % self.print_interval == 0:
                current_energy = self.gpu_energies[step].cpu().item()
                current_kinetic_energy = out['kinetic_energy'].cpu().item()
                total_energy = current_energy + current_kinetic_energy
                current_temperature = out['temperature'].cpu().item()
                print(
                    #f"Step {step + 1}/{self.num_steps}: Energy = {current_energy}, First atom pos = {current_coords[0]}, First atom vel = {(self.model.molecular.atom_velocities[0].cpu().detach()).numpy()}"
                    f"Step {step + 1}/{self.num_steps}:Tot_E={total_energy:.4f}, Pot_E = {current_energy:.4f}, Kin_E = {current_kinetic_energy:.4f}, T = {current_temperature:.4f}, Density = {rho: .4f}"
                )
            if self.save_to_graph_dataset:
                graph_data = self.model.molecular.graph_data
                data = Data(
                    x=graph_data.get('x', None),
                    pos = graph_data.get('pos', None),
                    edge_index=graph_data.get('edge_index', None),
                    edge_attr=graph_data.get('edge_attr', None),
                    forces=out['forces'].detach(),
                    energy=out['energy'].detach()
                )
                self.dataset.append(data)

        self.trajectory = self.gpu_trajectory.detach().cpu().numpy()
        self.force_list = self.gpu_forces.detach().cpu().numpy()
        self.energy_list = self.gpu_energies.detach().cpu().numpy()

        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"{self.num_steps} steps simulation finished.")
        print(f"Total simulation time: {elapsed_time:.2f} seconds.")

        return {
            'trajectory': self.trajectory,
            'energies': self.energy_list,
            'forces': self.force_list
        }

    def save_forces_grad(self, filename: str, with_no_ele=False, atom_types=None):
        if len(self.force_list)==0:
            print("No forces to save.")
            return
        num_atoms = self.trajectory[0].shape[0]
        with open(filename, 'w') as f:
            for step_idx, forces in enumerate(self.force_list):
                energy = self.energy_list[step_idx]
                f.write(f"{num_atoms}\n")
                f.write(f"Step {step_idx}, Energy = {energy}\n")

                for i in range(num_atoms):
                    x, y, z = forces[i]
                    if with_no_ele:
                        f.write(f"{x:.2f} {y:.2f} {z:.2f}\n")
                    else:
                        f.write(f"{atom_types[i]} {x:.2f} {y:.2f} {z:.2f}\n")

        print(f"forces saved to {filename}")

    def save_xyz_trajectory(self, filename: str, atom_types=None):

        if not self.trajectory.any():
            print("No trajectory to save.")
            return

        num_atoms = self.trajectory[0].shape[0]

        if not atom_types or len(atom_types) != num_atoms:
            atom_types = ["X"] * num_atoms

        with open(filename, 'w') as f:
            for step_idx, coords in enumerate(self.trajectory):
                energy = self.energy_list[step_idx]
                f.write(f"{num_atoms}\n")
                f.write(f"Step {step_idx}, Energy = {energy}\n")

                for i in range(num_atoms):
                    x, y, z = coords[i]
                    f.write(f"{atom_types[i]} {x:.2f} {y:.2f} {z:.2f}\n")

        print(f"XYZ trajectory saved to {filename}")

    def save_energy_curve(self, filename: str):
        if not self.energy_list.any():
            print("No energy data to save.")
            return

        from matplotlib import pyplot as plt

        plt.plot(self.energy_list)
        plt.xlabel("Step")
        plt.ylabel("Energy")
        plt.title("Energy curve")
        plt.savefig(filename)
        print(f"Energy curve saved to {filename}")

    def save_graph_dataset(self, filename: str):
        if not self.dataset:
            print("No graph dataset to save.")
            return
        torch.save(self.dataset, filename) # md_simulation_dataset.pth
        print(f"Saved {len(self.dataset)} graph snapshots to dataset")