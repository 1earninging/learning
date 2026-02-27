#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Day1 交付：
// 1) 写：vector add kernel（含边界检查）+ 正确性验证
// 2) 测：记录 kernel time；对比不同 blockDim（128/256/512）

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err__ = (call);                                         \
        if (err__ != cudaSuccess) {                                         \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__,        \
                         __LINE__, cudaGetErrorString(err__));              \
            std::exit(1);                                                   \
        }                                                                   \
    } while (0)

__global__ void vec_add(const float* __restrict__ a,
                        const float* __restrict__ b,
                        float* __restrict__ c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

static void init_host(std::vector<float>& a, std::vector<float>& b) {
    for (size_t i = 0; i < a.size(); ++i) {
        // 简单初始化，便于 debug
        a[i] = static_cast<float>(i % 1024) * 0.001f;
        b[i] = static_cast<float>((i * 7) % 1024) * 0.001f;
    }
}

static bool check_correct(const std::vector<float>& a, const std::vector<float>& b,
                          const std::vector<float>& c) {
    if (a.size() != b.size() || a.size() != c.size()) return false;
    double max_abs_err = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double ref = static_cast<double>(a[i]) + static_cast<double>(b[i]);
        double err = std::fabs(static_cast<double>(c[i]) - ref);
        if (err > max_abs_err) max_abs_err = err;
        if (err > 1e-5) {
            std::fprintf(stderr, "Mismatch at i=%zu: got=%f ref=%f err=%g\n", i,
                         c[i], static_cast<float>(ref), err);
            return false;
        }
    }
    std::printf("Correctness OK. max_abs_err=%g\n", max_abs_err);
    return true;
}

static float time_kernel_ms(int n, int block_dim, const float* d_a,
                            const float* d_b, float* d_c) {
    int grid = (n + block_dim - 1) / block_dim;

    // warmup
    for (int i = 0; i < 5; ++i) {
        vec_add<<<grid, block_dim>>>(d_a, d_b, d_c, n);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int iters = 200;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        vec_add<<<grid, block_dim>>>(d_a, d_b, d_c, n);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / iters;
}

int main(int argc, char** argv) {
    int n = 1 << 24; // 默认 ~16M 元素
    if (argc >= 2) n = std::atoi(argv[1]);
    if (n <= 0) {
        std::fprintf(stderr, "Usage: %s [n]\n", argv[0]);
        return 1;
    }

    std::printf("N=%d\n", n);

    std::vector<float> h_a(n), h_b(n), h_c(n);
    init_host(h_a, h_b);

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), n * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), n * sizeof(float),
                          cudaMemcpyHostToDevice));

    const int blocks[] = {128, 256, 512};
    for (int bd : blocks) {
        float t_ms = time_kernel_ms(n, bd, d_a, d_b, d_c);
        // 理论字节数：读 a+b 写 c -> 3 * n * sizeof(float)
        double bytes = 3.0 * n * sizeof(float);
        double gbps = bytes / (t_ms * 1e-3) / 1e9;
        std::printf("blockDim=%d: time=%.4f ms, effective_bw=%.2f GB/s\n", bd,
                    t_ms, gbps);
    }

    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, n * sizeof(float),
                          cudaMemcpyDeviceToHost));
    bool ok = check_correct(h_a, h_b, h_c);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return ok ? 0 : 2;
}

