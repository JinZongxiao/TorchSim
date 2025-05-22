import torch.nn as nn
import torch
from core.md_model import IntegratorInterface
import time

class VerletIntegrator(IntegratorInterface, nn.Module):
    def __init__(self,
                 molecular,
                 dt,
                 force_field,
                 ensemble=None, temperature=None, gamma=None):
        super().__init__()
        self.molecular = molecular
        self.box_length = molecular.box_length

        self.force_field = force_field
        self.BOLTZMAN = 8.617333262e-5  # ev/K

        self.dt = dt * 0.001  # ps

        self.atom_mass = self.molecular.atom_mass.unsqueeze(-1).expand_as(self.molecular.atom_velocities)

        if ensemble == 'NVT' and temperature is not None and gamma is not None:
            self.init_temperature = torch.tensor(temperature[0], device=molecular.device)
            self.temperature = torch.tensor(temperature[1], device=molecular.device)
            self.gamma = torch.tensor(gamma, device=molecular.device)

            # self.gamma = self.gamma / 1e4 * 48.88821
            # self.gamma = self.gamma / 1e4

            self.is_langevin_thermostat = True

            # self.molecular.create_velocity_gaussian(temperature, 773)
            self.molecular.set_maxwell_boltzmann_velocity(self.init_temperature)

            self.random_force_factor = torch.sqrt(
                2 * self.gamma * self.BOLTZMAN / self.atom_mass * self.temperature * self.dt
            )
            # self.csi = torch.randn_like(self.random_force_factor)*self.random_force_factor
        else:
            self.is_langevin_thermostat = False

        self.new_coords = torch.empty_like(molecular.coordinates,device=molecular.device)
        self.vel_half = torch.zeros_like(molecular.coordinates,device=molecular.device)


    def apply_pbc(self, coordinates):
        return coordinates - torch.floor(coordinates / self.box_length) * self.box_length

    def forward(self, force):

        # mass : amu
        # force : eV/A
        # vel : A/ps
        vel = self.molecular.atom_velocities
        accel = force / self.atom_mass

        ## first_vv
        self.new_coords = self.molecular.coordinates + vel * self.dt + 0.5 * accel * (self.dt ** 2)
        self.vel_half = vel + 0.5 * accel * self.dt

        self.molecular.update_coordinates(self.new_coords)
        force = self.force_field()['forces']

        if self.is_langevin_thermostat:
            csi = torch.randn_like(self.random_force_factor) * self.random_force_factor
            self.vel_half = self.vel_half - self.gamma * self.vel_half * self.dt + csi

        ## second_vv
        accel = force / self.atom_mass
        vel = self.vel_half + 0.5 * accel * self.dt
        self.molecular.update_velocities(vel)

        kin_energy = (0.5 * self.atom_mass * vel.pow(2)).sum()/2
        T = (2.0 / 3.0) * kin_energy / (self.molecular.atom_count * self.BOLTZMAN)


        return {'update_coordinates': self.new_coords,
                'kinetic_energy': kin_energy,
                'temperature': T}
