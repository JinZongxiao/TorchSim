import torch

def load_graph(filename: str):

    return torch.load(filename)

# dataset = load_graph("C:\\Users\\Thinkstation2\\Desktop\\MD_graph_dataset.pt")
# print(dataset)

def calc_rho(data,box_length):
    v = box_length * 3
    Nm = data.x[:, 2].sum()
    Na = 6.02e23
    rho = Nm / (v * Na * 1e-22)

    return rho