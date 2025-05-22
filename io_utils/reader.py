import itertools
from collections import defaultdict

import torch

from core.element_info import get_element_mass, get_element_iron_num

from pymatgen.core import Structure, Lattice

from torch_geometric.data import Data
from torch_geometric.nn import radius_graph

from scipy.spatial import cKDTree
from core.neighbor_search.gpu_kdtree import find_neighbors_gpu_pbc


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class AtomFileReader:

    def __init__(self,
                 filename,
                 box_length,
                 cutoff,
                 device=DEVICE,
                 parameter = None,
                 skin_thickness = 5.0,
                 is_mlp = False
                 ):
        self.device = device
        self.box_length_cpu = box_length
        self.box_length = torch.tensor(box_length,device=self.device)
        self.cutoff = torch.tensor(cutoff,device=self.device)
        self.parameter = parameter
        self.is_mlp = is_mlp

        self.atom_count = 0
        self.atom_types = []
        self.coordinates = []
        self.atom_mass = []
        self.atom_iron_num = []

        self.read_file(filename)

        self.atom_set = self.get_atom_set()

        self.atom_velocities = torch.zeros(self.atom_count, 3,device=self.device)

        self.element_to_id = []
        element_to_id_result_temp = {}
        for element in self.atom_types:
            if element not in element_to_id_result_temp:
                element_to_id_result_temp[element] = len(element_to_id_result_temp)
            self.element_to_id.append(element_to_id_result_temp[element])
        self.element_ids = torch.tensor(self.element_to_id, device=self.device)

        self.skin_thickness = torch.tensor(skin_thickness,device=device)
        self.verlet_cutoff = self.cutoff + self.skin_thickness
        self.last_positions = None
        self.needs_update = True


        self.graph_data = self.initialize_pyg_data(self.verlet_cutoff)
        if not is_mlp:
            self.pair_params = self.initial_parameters()

    def get_atom_set(self):
        count_dict = {elem: self.atom_types.count(elem) for elem in self.atom_types}
        duplicated_set = {elem for elem, count in count_dict.items() if count > 1}
        return list(duplicated_set)

    def get_atom_mass(self):
        return self.atom_mass

    def get_atom_num(self):
        return self.atom_count

    def get_atom_type_array(self):
        return self.atom_types

    def get_atom_coordinates(self):
        return self.coordinates

    def get_parameter(self, param_name:str):
        return torch.tensor(
            [self.parameter[str(index.cpu().numpy())][param_name] for index in self.graph_data.index_pairs],
            device=self.device
        )

    def update_coordinates(self, coordinates):
        if self.last_positions is not None:
            displacement = coordinates - self.last_positions
            displacement -= torch.round(displacement / self.box_length) * self.box_length
            max_displacement = torch.max(torch.norm(displacement, dim=1))

            if max_displacement > self.skin_thickness / 2:
                self.needs_update = True
                self.last_positions = coordinates.clone()
            else:
                self.needs_update = False
        else:
            self.needs_update = True
            self.last_positions = coordinates.detach().clone()

        self.coordinates = coordinates
        self.graph_data.pos = coordinates

        if self.needs_update:
            self.update_neighbor_list()  
            if not self.is_mlp:
                self.pair_params = self.initial_parameters()
        else:
            self.graph_data.edge_attr = self.calculate_edge_attr(
                coordinates,
                self.graph_data.edge_index,
                self.box_length
            )

    def initial_parameters(self):
        first_key = next(iter(self.parameter))
        first_value = self.parameter[first_key]
        param_list = []
        for key in first_value:
            param_list.append(torch.tensor(
            [self.parameter[str(index.cpu().numpy())][key] for index in self.graph_data.index_pairs],
            device=self.device
        ))
        return param_list



    def update_neighbor_list(self):
        # edge_index, edge_attr = self.find_neighbors(self.coordinates, self.verlet_cutoff)
        edge_index, edge_attr = self.find_neighbors_kdtree(self.coordinates, self.verlet_cutoff)
        self.graph_data.edge_index = edge_index
        self.graph_data.edge_attr = edge_attr
        element_ids = torch.tensor(self.element_to_id, device=self.device)
        self.graph_data.element_edge_0 = element_ids[edge_index[0]]
        self.graph_data.element_edge_1 = element_ids[edge_index[1]]
        self.graph_data.index_pairs = torch.stack(
            [self.graph_data.element_edge_0, self.graph_data.element_edge_1],
            dim=1
        )
        self.needs_update = False
        print(f"Neighbor list has been updated, edge number change to: {self.graph_data.edge_index.shape[1]}")

    def update_velocities(self, velocities):
        self.atom_velocities = velocities

    def read_file(self, filename):
        with open(filename, 'r') as file:
            lines = file.readlines()
            if not lines:
                raise ValueError(f"Error: The file {filename} is empty.")
            skip_lines = 0
            if filename.lower().endswith('.xyz'):
                skip_lines = 1
            try:
                self.atom_count = int(lines[0].strip())
                for line in lines[1 + skip_lines:]:
                    parts = line.split()
                    if len(parts) >= 4:
                        atom_type = parts[0]
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        self.atom_types.append(atom_type)
                        self.atom_mass.append(get_element_mass(atom_type))
                        self.atom_iron_num.append(get_element_iron_num(atom_type))
                        self.coordinates.append([x,y,z])
            except Exception as e:
                raise ValueError(f"Error: {e}")
            self.atom_mass = torch.tensor(self.atom_mass, device=self.device)
            self.atom_iron_num = torch.tensor(self.atom_iron_num, device=self.device)
            self.coordinates = torch.tensor(self.coordinates, device=self.device)

    def to_pymatgen_structure(self):
        coords = self.coordinates.detach().cpu().numpy()
        lattice = Lattice([[self.box_length_cpu, 0, 0],
                           [0, self.box_length_cpu, 0],
                           [0, 0, self.box_length_cpu]])
        structure = Structure(lattice,
                              self.atom_types,
                              coords,
                              coords_are_cartesian=True)
        return structure

    def create_velocity_gaussian(self, temperature, seed):
        torch.manual_seed(seed)
        if self.device == 'cuda':
            torch.cuda.manual_seed(seed)

        k_b_ev = 8.617333262e-5  # eV/(K·atom)
        ev_to_gA2ps2 = 1.60218e-20  # 1 eV -> g·Å²/ps²

        #g/atom
        # 1 amu = 1 g/mol
        # /N_A
        # mass_g_per_atom = self.atom_mass / 6.02214076e23
        mass_g_per_atom = self.atom_mass


        # A/ps
        sigma_squared = (k_b_ev * temperature * ev_to_gA2ps2) / mass_g_per_atom
        sigma = torch.sqrt(sigma_squared)

        velocities = torch.randn((self.atom_count, 3), device=self.device) * sigma.view(-1, 1)

        total_momentum = torch.sum(velocities * mass_g_per_atom.view(-1, 1), dim=0)
        total_mass = torch.sum(mass_g_per_atom)
        velocities -= total_momentum / total_mass

        self.atom_velocities = velocities

    def set_maxwell_boltzmann_velocity(self,temperature):
        natoms = self.atom_count
        BOLTZMAN = 8.617333262e-5
        mass = (self.atom_mass).unsqueeze(-1).expand_as(self.atom_velocities)
        velocities = torch.sqrt(temperature * BOLTZMAN / mass) * torch.randn((natoms, 3),device=self.device)
        self.atom_velocities = velocities

    def initialize_pyg_data(self, cutoff):
        atom_types_index = torch.tensor(list(range(0, len(self.atom_types))), device=self.device)
        pos = self.coordinates
        pos.requires_grad_(True)
        # edge_index, edge_attr = self.find_neighbors(pos, cutoff)
        edge_index, edge_attr = self.find_neighbors_kdtree(pos, cutoff)
        element_ids = torch.tensor(self.element_to_id, device=self.device)
        element_edge_0 = element_ids[edge_index[0]]  # 直接索引映射
        element_edge_1 = element_ids[edge_index[1]]
        index_pairs = torch.stack([element_edge_0,element_edge_1], dim=1)
        # edge_attr.requires_grad_(True)
        return Data(
            x=torch.stack([atom_types_index,self.atom_iron_num, self.atom_mass], dim=1).float().to(self.device),
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            # box_size=torch.tensor([box_length] * 3, device=self.device),
            index_pairs = index_pairs,
            element_edge_0 = element_edge_0,
            element_edge_1 = element_edge_1,
            energy = torch.zeros(0, device=self.device),
            forces = torch.empty((self.atom_count, 3), device=self.device)
        )

    @staticmethod
    def calculate_edge_attr(pos,edge_index,box_length):
        row, col = edge_index[0], edge_index[1]
        pos_i = pos[row]
        pos_j = pos[col]
        rij = pos_j - pos_i
        rij = rij - box_length * torch.round(rij / box_length)
        distances = torch.norm(rij, dim=1, keepdim=False)
        return distances

    @staticmethod
    def build_cell_list(pos, box_length, cutoff):
        cell_size = cutoff  
        n_cells = int(box_length // cell_size)
        cell_indices = (pos / cell_size).floor().long() % n_cells
        return cell_indices, n_cells, cell_size

    @staticmethod
    def assign_particles_to_cells(cell_indices):
        cell_dict = defaultdict(list)
        for particle_idx, idx in enumerate(cell_indices):
            cell_dict[tuple(idx.tolist())].append(particle_idx)
        return cell_dict

    def local_neighbor_search(self,pos, cell_dict, n_cells, cutoff, box_length):
        neighbors = []
        edge_attr_list  = []
        for i in range(len(pos)):
            cell_idx = tuple((pos[i] / (box_length / n_cells)).floor().long().tolist())
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        neighbor_cell = (
                            (cell_idx[0] + dx) % n_cells,
                            (cell_idx[1] + dy) % n_cells,
                            (cell_idx[2] + dz) % n_cells
                        )
                        for j in cell_dict.get(neighbor_cell, []):
                            if j <= i: continue
                            rij = pos[j] - pos[i]
                            # rij -= torch.round(rij / box_length) * box_length
                            rij = rij - box_length * torch.round(rij / box_length)
                            dist = torch.norm(rij)
                            if dist < cutoff:
                                neighbors.append((i, j))
                                edge_attr_list.append(dist)
        edge_index = torch.tensor(neighbors,device=self.device).t().contiguous()
        edge_attr  = torch.stack(edge_attr_list)
        return edge_index, edge_attr

    def find_neighbors(self, pos, cutoff):
        cell_indices, n_cells, cell_size = self.build_cell_list(pos, self.box_length, cutoff)

        cell_dict = self.assign_particles_to_cells(cell_indices)

        edge_index,edge_attr = self.local_neighbor_search(pos, cell_dict, n_cells, cutoff,self.box_length)
        return edge_index,edge_attr

    def expand_pos_pbc(self,pos):
        expanded_pos = []
        expanded_indices = []
        expanded_shifts = [] 
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    shift = torch.tensor([dx, dy, dz], device=self.device) * self.box_length
                    expanded_pos.append(pos + shift)
                    expanded_indices.extend(range(len(pos)))
                    expanded_shifts.extend([[dx, dy, dz]] * len(pos))
        expanded_pos = torch.cat(expanded_pos, dim=0)
        expanded_indices = torch.tensor(expanded_indices, device=self.device)
        expanded_shifts = torch.tensor(expanded_shifts, device=self.device)
        return expanded_pos, expanded_indices, expanded_shifts

    def find_neighbors_kdtree(self, pos, cutoff):
        print("Using GPU-accelerated neighbor search")
        edge_index, edge_attr = find_neighbors_gpu_pbc(pos, cutoff, self.box_length)
        return edge_index, edge_attr
        
    def find_neighbors_kdtree_cpu(self, pos, cutoff):
        expanded_pos, expanded_indices, expanded_shifts = self.expand_pos_pbc(pos)
        tree = cKDTree(expanded_pos.detach().cpu().numpy())
        pairs = tree.query_pairs(cutoff.cpu().numpy())
        pair_list = []
        for pair in pairs:
            if (99 >= pair[0] >= 0) or (99 >= pair[1] >= 0):
                pair_list.append(pair)
        edge_index = torch.tensor(pair_list, device=self.device).t().contiguous()
        edge_index = expanded_indices[edge_index]
        edge_attr = self.calculate_edge_attr(expanded_pos, edge_index, self.box_length)
        return edge_index, edge_attr

