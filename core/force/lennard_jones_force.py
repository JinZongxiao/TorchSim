import torch.nn as nn
import torch

from core.md_model import BackboneInterface
from core.auto_force_calculator import auto_force_calculator as afc

class LennardJonesForce(BackboneInterface, nn.Module):
    def __init__(self, molecular):
        super(LennardJonesForce, self).__init__()
        self.molecular = molecular

        self.pair_force = torch.empty_like(self.molecular.coordinates, device=self.molecular.device)


    def forward(self):
        r = self.molecular.graph_data.edge_attr
        pos = self.molecular.graph_data.pos
        epsilon = self.molecular.pair_params[0]
        sigma = self.molecular.pair_params[1]
        # epsilon = self.molecular.get_parameter('epsilon').contiguous()
        # sigma = self.molecular.get_parameter('sigma').contiguous()

        # v_r = 4*epsilon*((sigma/r)**12 - (sigma/r)**6)
        v_r = self.calc(r, sigma, epsilon)

        self.pair_force = afc(v_r, pos, self.molecular.atom_count, self.molecular.device, self.molecular.graph_data.edge_index[0])
        total_energy = v_r.sum() # ev
        self.pair_force = -self.pair_force # ev/A
        return {'energy': total_energy, 'forces': self.pair_force}


    @staticmethod
    def calc(r, sigma, epsilon):

        # v_r = torch.mul(4*epsilon,torch.sub(torch.pow(torch.div(sigma,r),12), torch.pow(torch.div(sigma, r), 6)))
        v_r = 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)
        return v_r