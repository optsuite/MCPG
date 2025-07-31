#include <torch/extension.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <iostream>

#define MAX_BATCH_SIZE 32

namespace py = pybind11;

__global__ void _naive_local_search_kernel(float* input_sols, float* Q, float* lin, 
        int32_t flip_idx, size_t num_vars, size_t num_sols) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_sols) {
        float cost = lin[flip_idx];
        for (size_t i = 0; i < num_vars; ++i) {
            cost += 2.0f * Q[flip_idx * num_vars + i] * input_sols[i * num_sols + tid];
        }
        input_sols[flip_idx * num_sols + tid] = cost > 0.01f ? 1.0f: 0.0f;
    }
}

void naive_local_search_kernel(torch::Tensor& input_sols, torch::Tensor& Q, torch::Tensor& lin, 
        torch::Tensor& perm){
    size_t num_vars = input_sols.size(0);
    size_t num_sols = input_sols.size(1);
    auto perm_accessor = perm.accessor<int32_t, 1>();
    for (size_t i = 0; i < num_vars; ++i) {
        size_t flip_idx = perm_accessor[i];
        _naive_local_search_kernel<<<(num_sols + 255) / 256, 256>>>(
            input_sols.data_ptr<float>(), Q.data_ptr<float>(), lin.data_ptr<float>(),
            flip_idx, num_vars, num_sols);
    }
}

int naive_local_search(torch::Tensor& input_sols, torch::Tensor& Q, torch::Tensor& lin, torch::Tensor& perm) {
    TORCH_CHECK(input_sols.is_cuda(), "input_sols must be a CUDA tensor");
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(lin.is_cuda(), "lin must be a CUDA tensor");
    TORCH_CHECK(!perm.is_cuda(), "perm must be a CPU tensor");
    TORCH_CHECK(input_sols.dtype() == torch::kFloat32, "input_sols must be float32");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Q must be float32");
    TORCH_CHECK(lin.dtype() == torch::kFloat32, "lin must be float32");
    TORCH_CHECK(perm.dtype() == torch::kInt32, "perm must be int32");

    naive_local_search_kernel(input_sols, Q, lin, perm);
    return 0;
}

// __global__ void _batch_local_search_kernel(float* input_sols, float* Q, float* lin, 
//         int* flip_idx_list, size_t num_vars, size_t num_sols, size_t batch_size) {
//     size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid < num_sols) {
//         for (size_t k = 0; k < batch_size; ++k){
//             int flip_idx = flip_idx_list[k];
//             // if (flip_idx < 0 || flip_idx >= num_vars) {
//             //     continue;
//             float cost = lin[flip_idx];
//             for (size_t i = 0; i < num_vars; ++i) {
//                 cost += 2.0f * Q[flip_idx * num_vars + i] * input_sols[i * num_sols + tid];
//             }
//             input_sols[flip_idx * num_sols + tid] = cost > 0.0f ? 1.0f: 0.0f;
//         }
//     }
// }
__global__ void _batch_local_search_kernel(float* input_sols, float* Q, float* lin, 
        int* flip_idx_list, size_t num_vars, size_t num_sols, size_t batch_size) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_sols) {
        int final_flip_idx = -1;
        float final_cost = -1.0f;
        for (size_t k = 0; k < batch_size; ++k){
            int flip_idx = flip_idx_list[k];
            float cost = lin[flip_idx];
            for (size_t i = 0; i < num_vars; ++i) {
                cost += 2.0f * Q[flip_idx * num_vars + i] * input_sols[i * num_sols + tid];
            }
            // cost *= -(2 * input_sols[flip_idx * num_sols + tid] - 1.0f);
            cost *= input_sols[flip_idx * num_sols + tid] > 0.0f ? -1.0f : 1.0f;
            if (cost > 0.0f) {
                final_flip_idx = flip_idx;
            }
        }
        if (final_flip_idx >= 0) {
            input_sols[final_flip_idx * num_sols + tid] = 1.0f - input_sols[final_flip_idx * num_sols + tid];
        }
    }
}
void batch_local_search_kernel(torch::Tensor& input_sols, torch::Tensor& Q, torch::Tensor& lin, 
        torch::Tensor& perm, size_t batch_size) {
    size_t num_vars = input_sols.size(0);
    size_t num_sols = input_sols.size(1);
    int* perm_d;
    cudaMalloc((void**)&perm_d, num_vars * sizeof(int));
    cudaMemcpy(perm_d, perm.data_ptr<int>(), num_vars * sizeof(int), cudaMemcpyHostToDevice);
    
    size_t batch_num = (num_vars + batch_size - 1) / batch_size;
    for (size_t i = 0; i < batch_num; ++i) {
        size_t start_bidx = i * batch_size;
        size_t end_bidx = std::min(start_bidx + batch_size, num_vars);
        size_t current_batch_size = end_bidx - start_bidx;
        if (current_batch_size <= 0) {
            continue;
        }
        int* flip_idx_list = perm_d + start_bidx;
        _batch_local_search_kernel<<<(num_sols + 255) / 256, 256>>>(
            input_sols.data_ptr<float>(), Q.data_ptr<float>(), lin.data_ptr<float>(),
            flip_idx_list, num_vars, num_sols, current_batch_size);
    }
    cudaFree(perm_d);
    cudaDeviceSynchronize();
}

int batch_local_search(torch::Tensor& input_sols, torch::Tensor& Q, torch::Tensor& lin, torch::Tensor& perm, size_t batch_size) {
    TORCH_CHECK(input_sols.is_cuda(), "input_sols must be a CUDA tensor");
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(lin.is_cuda(), "lin must be a CUDA tensor");
    TORCH_CHECK(!perm.is_cuda(), "perm must be a CPU tensor");
    TORCH_CHECK(input_sols.dtype() == torch::kFloat32, "input_sols must be float32");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Q must be float32");
    TORCH_CHECK(lin.dtype() == torch::kFloat32, "lin must be float32");
    TORCH_CHECK(perm.dtype() == torch::kInt32, "perm must be int32");
    TORCH_CHECK(batch_size > 0 && batch_size <= MAX_BATCH_SIZE, "batch_size must be between 1 and MAX_BATCH_SIZE");

    batch_local_search_kernel(input_sols, Q, lin, perm, batch_size);
    return 0;
}


__global__ void _assign_flip(float* sols_ptr, float* rand_ptr, float prob, size_t n_sols) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n_sols) {
        sols_ptr[tid] = (rand_ptr[tid] < prob) ? 1.0f : 0.0f;
    }
}
void sampling_kernel(torch::Tensor& input_sols, torch::Tensor& prob, torch::Tensor& all_flip_idx) {
    // input_sols: (num_vars, num_sols)
    size_t num_vars = input_sols.size(0);
    size_t num_sols = input_sols.size(1);
    size_t flip_num = all_flip_idx.size(0);

    auto prob_accessor = prob.accessor<float, 1>();
    auto flip_idx_accessor = all_flip_idx.accessor<int32_t, 1>();

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    float* all_rand;
    cudaMalloc((void**)&all_rand, flip_num * num_sols * sizeof(float));
    curandGenerateUniform(gen, all_rand, flip_num * num_sols);

    for (size_t i = 0; i < flip_num; ++i) {
        size_t idx = flip_idx_accessor[i];
        float* all_rand_ptr = all_rand + i * num_sols;
        float* input_sols_ptr = input_sols.data_ptr<float>() + idx * num_sols;
        _assign_flip<<<(num_sols + 255) / 256, 256>>>(input_sols_ptr, all_rand_ptr, prob_accessor[i], num_sols);
    }
    cudaDeviceSynchronize();
    cudaFree(all_rand);
    curandDestroyGenerator(gen);
}

int sampling(torch::Tensor& input_sols, torch::Tensor& prob, torch::Tensor& flip_idx) {
    TORCH_CHECK(input_sols.is_cuda(), "input_sols must be a CUDA tensor");
    TORCH_CHECK(!prob.is_cuda(), "prob must be a CPU tensor");
    TORCH_CHECK(!flip_idx.is_cuda(), "flip_idx must be a CPU tensor");
    TORCH_CHECK(input_sols.dtype() == torch::kFloat32, "input_sols must be float32");
    TORCH_CHECK(prob.dtype() == torch::kFloat32, "prob must be float32");
    TORCH_CHECK(flip_idx.dtype() == torch::kInt32, "flip_idx must be int32");
    
    sampling_kernel(input_sols, prob, flip_idx);
    return 0;
}