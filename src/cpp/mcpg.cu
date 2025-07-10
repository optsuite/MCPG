#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <stdexcept>
#include <curand.h>

extern "C" {

struct d_mat_t{
    size_t rows;
    size_t cols;
    double* data;
    d_mat_t(size_t r, size_t c) : rows(r), cols(c) {
        cudaMalloc(&data, r * c * sizeof(double));
    }
    ~d_mat_t() {
        if (data) {
            cudaFree(data);
        }
    }
    d_mat_t(const d_mat_t&) = delete; // Disable copy constructor
    d_mat_t& operator=(const d_mat_t&) = delete; // Disable copy assignment
};
struct d_vec_t{
    size_t size;
    double* data;
    d_vec_t(size_t s) : size(s) {
        cudaMalloc(&data, s * sizeof(double));
    }
    ~d_vec_t() {
        if (data) {
            cudaFree(data);
        }
    }
    d_vec_t(const d_vec_t&) = delete; // Disable copy constructor
    d_vec_t& operator=(const d_vec_t&) = delete; // Disable copy assignment
};

std::vector<double> set_col_major(const std::vector<double>& vec, size_t rows, size_t cols) {
    std::vector<double> col_major(rows * cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            col_major[j * rows + i] = vec[i * cols + j];
        }
    }
    return col_major;
}
std::vector<double> set_row_major(const std::vector<double>& vec, size_t rows, size_t cols) {
    std::vector<double> row_major(rows * cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            row_major[i * cols + j] = vec[j * rows + i];
        }
    }
    return row_major;
}

inline void cpy_h2d(const std::vector<double>& vec, d_mat_t& d_mat) {
    if (vec.size() != d_mat.rows * d_mat.cols) {
        throw std::runtime_error("Size mismatch between vector and device matrix.");
    }
    cudaMemcpy(d_mat.data, vec.data(), d_mat.rows * d_mat.cols * sizeof(double), cudaMemcpyHostToDevice);
}
inline void cpy_d2h(const d_mat_t& d_mat, std::vector<double>& vec) {
    vec.resize(d_mat.rows * d_mat.cols);
    cudaMemcpy(vec.data(), d_mat.data, d_mat.rows * d_mat.cols * sizeof(double), cudaMemcpyDeviceToHost);
}
inline void cpy_h2d_vec(const std::vector<double>& vec, d_vec_t& d_vec) {
    if (vec.size() != d_vec.size) {
        throw std::runtime_error("Size mismatch between vector and device vector.");
    }
    cudaMemcpy(d_vec.data, vec.data(), d_vec.size * sizeof(double), cudaMemcpyHostToDevice);
}
inline void cpy_d2h_vec(const d_vec_t& d_vec, std::vector<double>& vec) {
    vec.resize(d_vec.size);
    cudaMemcpy(vec.data(), d_vec.data, d_vec.size * sizeof(double), cudaMemcpyDeviceToHost);
}

inline void matmul(cublasHandle_t handle, 
    const d_mat_t& A, const d_mat_t& B, d_mat_t& C,
    double alpha = 1.0, double beta = 0.0) {
    // Ensure that A, B, and C are in column-major order
    // A: m * k, B: k*n, C: m*n
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                A.rows, B.cols, A.cols,
                &alpha,
                A.data, A.rows,
                B.data, B.rows,
                &beta,
                C.data, C.rows);
}
__global__ void set_sol_by_delta(double* sol, const double* delta, 
    size_t n_sol, size_t n_var, size_t idx) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n_sol) {
        double* col_ptr = sol + idx * n_sol;
        col_ptr[tid] = (delta[tid] > 0.0) ? 1.0 : 0.0;
    }
}

void local_search(cublasHandle_t handle, const d_mat_t& sols, const d_mat_t& q_mat) {
    // sol: n_sol x n_var
    // q_mat: n_var x n_var
    size_t n_sol = sols.rows;
    size_t n_var = q_mat.rows;

    if (sols.cols != n_var || q_mat.cols != n_var) {
        throw std::runtime_error("Matrix dimensions do not match for multiplication.");
    }
    std::vector<double> h_ones(n_sol, 1.0);
    d_mat_t d_Qx(n_sol, n_var); // n_sol * n_var matrix to store Qx
    d_vec_t d_delta(n_sol); // n_sol vector to store Qx column
    // Copy h_ones to d_ones for each iteration
    for (size_t idx = 0; idx < n_var; ++idx) {
        matmul(handle, sols, q_mat, d_Qx);
        cublasDcopy(handle, n_sol, d_Qx.data + idx * n_sol, 1, d_delta.data, 1);
        set_sol_by_delta<<<(n_sol + 255) / 256, 256>>>(sols.data, d_delta.data, n_sol, n_var, idx);
    }
}
void sampling(cublasHandle_t handle, const d_mat_t& sols, const d_vec_t& prob, size_t flip_num){
    if (sols.rows != prob.size) {
        throw std::runtime_error("sols.rows must equal prob.size");
    }
    if (flip_num == 0) {
        return;
    }
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, time(NULL));
    d_vec_t d_flip_idx(flip_num);
    // Generate random indices for flipping
    curandGenerateUniformDouble(gen, d_flip_idx.data, flip_num);
    // Scale indices to the range [0, sols.rows)
    double scale = static_cast<double>(sols.rows);
    for (size_t i = 0; i < flip_num; ++i) {
        d_flip_idx.data[i] = static_cast<size_t>(d_flip_idx.data[i] * scale);
    }
    // Copy the generated indices to the device
    std::vector<double> h_flip_idx(flip_num);
    cpy_d2h_vec(d_flip_idx, h_flip_idx);

    // Perform the flipping operation
    d_mat_t d_rand(sols.rows, flip_num);
    curandGenerateUniformDouble(gen, d_rand.data, d_rand.rows * d_rand.cols);
    for (size_t i = 0; i < flip_num; ++i) {
        size_t idx = static_cast<size_t>(h_flip_idx[i]);
        d_vec_t temp(sols.rows);
        double* current_rand_ptr = d_rand.data + idx * d_rand.rows;
        cudaMemcpy(temp.data, prob.data, sols.rows * sizeof(double), cudaMemcpyDeviceToDevice);
        double alpha = -1.0;
        cublasDaxpy(handle, sols.rows, &alpha, current_rand_ptr, 1, temp.data, 1);
        for (size_t j = 0; j < sols.rows; ++j) {
            double val;
            cudaMemcpy(&val, temp.data + j, sizeof(double), cudaMemcpyDeviceToHost);
            val = (val > 0.0) ? 1.0 : 0.0;
            cudaMemcpy(temp.data + j, &val, sizeof(double), cudaMemcpyHostToDevice);
        }
        cudaMemcpy(sols.data + idx * sols.rows, temp.data, sols.rows * sizeof(double), cudaMemcpyDeviceToDevice);
    }
    return;
}
} // extern "C"
