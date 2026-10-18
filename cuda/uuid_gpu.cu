#include <cuda_runtime.h>
#include <chrono>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(_WIN32)
#include <io.h>
#include <fcntl.h>
#endif

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ':' << __LINE__ << '\n'; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

struct RawUuid {
    unsigned long long hi;
    unsigned long long lo;
};
static_assert(sizeof(RawUuid) == 16, "RawUuid size");

__host__ __device__ inline unsigned long long splitmix64(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// Same two splitmix64 calls and bit layout as src/fair.rs.
// Deterministic synthetic UUIDv7 workload; NOT cryptographically secure.
__host__ __device__ inline RawUuid generate_id(unsigned long long seed,
                                                unsigned long long index,
                                                unsigned long long unix_ms) {
    const auto a = splitmix64(seed ^ (index * 0x9E3779B97F4A7C15ULL));
    const auto b = splitmix64(seed ^ (index * 0xBF58476D1CE4E5B9ULL) ^ 0xD1B54A32D192ED03ULL);
    return {
        ((unix_ms & 0xFFFFFFFFFFFFULL) << 16) | (7ULL << 12) | (a & 0xFFFULL),
        (2ULL << 62) | (b & 0x3FFFFFFFFFFFFFFFULL)
    };
}

__global__ void throughput_kernel(unsigned long long* checksum,
                                  unsigned long long seed, unsigned long long start_index,
                                  unsigned long long unix_ms, unsigned int iterations) {
    const unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long offset = start_index + idx * iterations;
    unsigned long long local = 0;
    for (unsigned int i = 0; i < iterations; ++i) {
        auto id = generate_id(seed, offset + i, unix_ms);
        local ^= id.hi ^ id.lo;
    }
    atomicXor(checksum, local);
}

__global__ void stream_kernel(RawUuid* out, unsigned int count,
                              unsigned long long seed, unsigned long long start_index,
                              unsigned long long unix_ms) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = gridDim.x * blockDim.x;
    for (unsigned int i = idx; i < count; i += stride) {
        out[i] = generate_id(seed, start_index + i, unix_ms);
    }
}

int stream_main(int argc, char** argv) {
    if (argc != 8) {
        std::cerr << "Usage: uuid-gpu stream SAMPLES BLOCKS TPB BATCH SEED TIMESTAMP_MS\n";
        return 2;
    }
    try {
        const auto samples = std::stoull(argv[2]);
        const auto blocks = static_cast<unsigned int>(std::stoul(argv[3]));
        const auto tpb = static_cast<unsigned int>(std::stoul(argv[4]));
        const auto batch = static_cast<unsigned int>(std::stoul(argv[5]));
        const auto seed = std::stoull(argv[6]);
        const auto ms = std::stoull(argv[7]);
        if (!samples || !blocks || !tpb || tpb > 1024 || !batch || batch > 262144)
            throw std::runtime_error("Invalid stream parameters");
        CUDA_CHECK(cudaSetDevice(0));
        RawUuid* gpu = nullptr;
        CUDA_CHECK(cudaMalloc(&gpu, batch * sizeof(RawUuid)));
        std::vector<RawUuid> host(batch);
#if defined(_WIN32)
        if (_setmode(_fileno(stdout), _O_BINARY) == -1)
            throw std::runtime_error("Cannot enable binary stdout");
#endif
        unsigned long long offset = 0;
        while (offset < samples) {
            const auto count = static_cast<unsigned int>(
                std::min<unsigned long long>(batch, samples - offset));
            stream_kernel<<<blocks, tpb>>>(gpu, count, seed, offset, ms);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(host.data(), gpu, count * sizeof(RawUuid), cudaMemcpyDeviceToHost));
            if (std::fwrite(host.data(), sizeof(RawUuid), count, stdout) != count)
                throw std::runtime_error("Broken CUDA stdout pipe");
            offset += count;
        }
        if (std::fflush(stdout) != 0) throw std::runtime_error("Cannot flush CUDA stdout");
        CUDA_CHECK(cudaFree(gpu));
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "CUDA stream error: " << e.what() << '\n';
        return 2;
    }
}

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "stream") return stream_main(argc, argv);
    if (argc != 7) {
        std::cerr << "Usage: uuid-gpu SECONDS BLOCKS TPB ITERATIONS SEED TIMESTAMP_MS\n";
        return 2;
    }
    try {
        const unsigned long seconds = std::stoul(argv[1]);
        const unsigned int blocks = static_cast<unsigned int>(std::stoul(argv[2]));
        const unsigned int tpb = static_cast<unsigned int>(std::stoul(argv[3]));
        const unsigned int iterations = static_cast<unsigned int>(std::stoul(argv[4]));
        const unsigned long long seed = std::stoull(argv[5]);
        const unsigned long long timestamp_ms = std::stoull(argv[6]);
        if (!seconds || !blocks || !tpb || tpb > 1024 || !iterations)
            throw std::runtime_error("Invalid benchmark parameters");
        CUDA_CHECK(cudaSetDevice(0));
        cudaDeviceProp props{};
        CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
        unsigned long long* checksum = nullptr;
        CUDA_CHECK(cudaMalloc(&checksum, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(checksum, 0, sizeof(unsigned long long)));
        using Clock = std::chrono::steady_clock;
        auto start = Clock::now();
        unsigned long long total = 0;
        const auto per_kernel = (unsigned long long)blocks * tpb * iterations;
        do {
            throughput_kernel<<<blocks, tpb>>>(checksum, seed, total, timestamp_ms, iterations);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            total += per_kernel;
        } while (Clock::now() - start < std::chrono::seconds(seconds));
        double elapsed = std::chrono::duration<double>(Clock::now() - start).count();
        unsigned long long host_checksum = 0;
        CUDA_CHECK(cudaMemcpy(&host_checksum, checksum, sizeof(host_checksum), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(checksum));
        std::cout.precision(9);
        std::cout << "GPU_RESULT|" << total << '|' << elapsed << '|' << host_checksum
                  << '|' << props.name << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 2;
    }
}