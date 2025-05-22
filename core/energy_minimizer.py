import torch
from scipy.optimize import minimize


def minimize_energy_bfgs_scipy(model,
                               max_inter=1000,
                               grad_tolerance=0.5):
    def eval_func(coords, info):
        n_atoms = model.molecular.coordinates.shape[0]
        coords = torch.tensor(coords.reshape(n_atoms, 3), dtype=model.molecular.coordinates.dtype,
                              device=model.molecular.coordinates.device)
        for backbone in model.sum_bone.backbones:
            out_backbone = backbone()
            energy_backbone = out_backbone['energy']
            grad_backbone = out_backbone['forces']
            energy_zero = torch.zeros_like(energy_backbone)
            grad_zero = torch.zeros_like(grad_backbone)
            break
        for backbone in model.sum_bone.backbones:
            out_backbone = backbone()
            energy_backbone = out_backbone['energy']
            grad_backbone = out_backbone['forces']
            energy_zero += energy_backbone
            grad_zero += grad_backbone
        current_energy = energy_zero.detach().cpu().numpy().flatten()
        grad = grad_zero.detach().cpu().numpy().flatten()

        if info["iteration_step"] % 1 == 0:
            print(info["iteration_step"], current_energy)
        info["iteration_step"] += 1

        return current_energy, grad

    print("Iter", "Epot")
    x0 = model.molecular.coordinates.detach().cpu().numpy().flatten()
    n_atoms = model.molecular.coordinates.shape[0]
    res = minimize(
        eval_func,
        x0,
        method="L-BFGS-B",
        jac=True,
        options={"gtol": grad_tolerance, "maxiter": max_inter, "maxls": 50, "ftol": 1e-10,"disp": True
                 },
        args=({"iteration_step": 0},)
    )
    model.molecular.coordinates = torch.tensor(
        res.x.reshape(n_atoms, 3),
        dtype=model.molecular.coordinates.dtype,
        device=model.molecular.coordinates.device,
        requires_grad=model.molecular.coordinates.requires_grad,
    )


def minimize_energy_bfgs_pytorch(model,
                                 max_iter: int = 100,
                                 tolerance: float = 1e-6,
                                 print_progress: bool = True):

    original_coords = model.molecular.coordinates.detach().requires_grad_(True)
    optimizer = torch.optim.LBFGS([original_coords],
                                  lr=0.1,
                                  max_iter=20,  # 每个外步的内迭代次数
                                  tolerance_grad=tolerance)

    energy_history = []

    def closure():
        optimizer.zero_grad()

        out = model()
        energy = out['energy']
        # grad = out['forces']
        return energy

    for epoch in range(max_iter):
        out = model()
        current_energy = out['energy']
        grad = out['forces']
        optimizer.step(closure)
        energy_history.append(current_energy)
        grad_norm = torch.norm(grad).item()

        if print_progress:
            print(f"Minimization Step {epoch + 1}/{max_iter}, Energy: {current_energy:.4f}")

        if grad_norm < tolerance:
            break

    model.molecular.coordinates.copy_(original_coords.data)

    return energy_history
