import torch
import torch.nn as nn
from abc import ABCMeta, abstractmethod
from typing import Dict, List

main_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BackboneInterface(nn.Module, metaclass=ABCMeta):

    @abstractmethod
    def forward(self) -> Dict[str, torch.Tensor]:
        pass

class IntegratorInterface(nn.Module, metaclass=ABCMeta):

    @abstractmethod
    def forward(self, force: torch.Tensor) -> Dict[str, torch.Tensor]:
        pass

class SumBackboneInterface(nn.Module):
    def __init__(self, backbones,molecular):
        super(SumBackboneInterface, self).__init__()
        self.backbones = nn.ModuleList(backbones)
        self.molecular = molecular
        self.device = self.molecular.device
        self.atom_num = self.molecular.atom_count

    def forward(self):
        total_forces = torch.zeros(self.atom_num, 3, device=self.device)
        total_energy = torch.tensor(0.0, device=self.device)
        for backbone in self.backbones:
            output = backbone()
            total_forces.add_(output['forces'])
            total_energy.add_(output['energy'])
        results = {
            'forces': total_forces,
            'energy': total_energy
        }
        return results


class BaseModel(nn.Module):
    
    def __init__(self,
                 sum_bone,
                 integrator: IntegratorInterface,
                 molecular):
        super(BaseModel, self).__init__()
        self.sum_bone = sum_bone
        self.Integrator = integrator
        self.molecular = molecular

        self.force_cache = torch.empty_like(molecular.coordinates, device=main_device)
        self.energy_cache = torch.empty(1, device=main_device)

    def forward(self):
        # current_coords = self.molecular.coordinates.detach().clone()
        # current_coords.requires_grad_(True)

        out = self.sum_bone()

        self.force_cache = out['forces']
        self.energy_cache = out['energy']
        #with torch.autograd.profiler.profile(enabled=True, use_device='cuda') as prof:
        integrator_output = self.Integrator.forward(self.force_cache)
        #print(prof.table(sort_by="self_cuda_time_total"))

        del out
        torch.cuda.empty_cache() 
        return {
            'forces': self.force_cache,
            'energy': self.energy_cache,
            'updated_coordinates': integrator_output['update_coordinates'],
            'kinetic_energy': integrator_output['kinetic_energy'],
            'temperature': integrator_output['temperature']
        }
