#include <torch/extension.h>

template <typename scalar_t>
__global__ void lj_force_kernel(
    const scalar_t* r,
    const int64_t* edge_index,
    const scalar_t* sigma,
    const scalar_t* epsilon,
    scalar_t* forces,
    scalar_t* energy,
    int64_t num_edges,
    int64_t num_atoms
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_edges) {
        scalar_t ratio = sigma[idx] / r[idx];
        scalar_t ratio6 = pow(ratio, 6);
        scalar_t ratio12 = ratio6 * ratio6;

        // Lennard-Jones力计算
        scalar_t force = 24 * epsilon[idx] * (2*ratio12 - ratio6) / r[idx];

        // 原子索引
        int64_t i = edge_index[idx * 2];
        int64_t j = edge_index[idx * 2 + 1];

        // 原子力累加（原子级原子操作）
        atomicAdd(&forces[i], force);
        atomicAdd(&forces[j], -force);

        // 能量累加
        atomicAdd(energy, 4 * epsilon[idx] * (ratio12 - ratio6));
    }
}

std::tuple<torch::Tensor, torch::Tensor> forward(
    torch::Tensor r,
    torch::Tensor edge_index,
    torch::Tensor sigma,
    torch::Tensor epsilon,
    int64_t num_atoms
) {
    // 预分配结果内存
    auto forces = torch::zeros({num_atoms}, r.options());
    auto energy = torch::zeros({1}, r.options());

    const int64_t num_edges = r.size(0);
    const int threads = 256;
    const int blocks = (num_edges + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(r.type(), "lj_forward", ([&] {
        lj_force_kernel<scalar_t><<<blocks, threads>>>(
            r.data<scalar_t>(),
            edge_index.data<int64_t>(),
            sigma.data<scalar_t>(),
            epsilon.data<scalar_t>(),
            forces.data<scalar_t>(),
            energy.data<scalar_t>(),
            num_edges,
            num_atoms
        );
    }));

    return std::make_tuple(forces, energy.sum());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LJ force CUDA kernel");
}