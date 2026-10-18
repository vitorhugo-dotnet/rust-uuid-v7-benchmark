#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <random>
#if defined(_WIN32)
#include <io.h>
#include <fcntl.h>
#endif
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ':' << __LINE__ << '\n'; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

using State = curandStatePhilox4_32_10_t;

__global__ void setup_rng(State* states, unsigned long long seed) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, static_cast<unsigned long long>(idx), 0ULL, &states[idx]);
}

__global__ void generate_v7(State* states, unsigned long long* checksum,
                            unsigned long long unix_ms, unsigned int iterations) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    State rng = states[idx];
    unsigned long long local = 0;
    for (unsigned int i = 0; i < iterations; ++i) {
        uint4 rnd = curand4(&rng);
        // RFC 9562 layout: 48-bit UNIX milliseconds, version 7, 74 random bits, RFC variant.
        // Philox is NOT a CSPRNG. These are synthetic v7-shaped values for a compute benchmark.
        unsigned long long hi = ((unix_ms & 0xFFFFFFFFFFFFULL) << 16)
                              | (7ULL << 12) | (static_cast<unsigned long long>(rnd.x) & 0xFFFULL);
        unsigned long long lo = (0x2ULL << 62)
                              | ((static_cast<unsigned long long>(rnd.y) << 32) | rnd.z) & 0x3FFFFFFFFFFFFFFFULL;
        // Depend on every generated value so the optimizer can't remove generation.
        local ^= hi ^ lo;
    }
    states[idx] = rng;
    atomicXor(checksum, local);
}


// Raw binary stream format: per UUID, little-endian uint64_t high bits followed
// by little-endian uint64_t low bits. Consumer converts to u128 and checks UUIDs.
struct RawUuid {
    unsigned long long hi;
    unsigned long long lo;
};
static_assert(sizeof(RawUuid) == 16, "RawUuid must occupy 16 bytes");

__global__ void stream_v7(State* states, RawUuid* out,
                          unsigned long long unix_ms, unsigned int count) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;
    State rng = states[idx];
    for (unsigned int i = idx; i < count; i += stride) {
        uint4 rnd = curand4(&rng);
        unsigned long long hi = ((unix_ms & 0xFFFFFFFFFFFFULL) << 16)
                              | (7ULL << 12)
                              | (static_cast<unsigned long long>(rnd.x) & 0xFFFULL);
        unsigned long long lo = (0x2ULL << 62)
                              | (((static_cast<unsigned long long>(rnd.y) << 32)
                              | rnd.z) & 0x3FFFFFFFFFFFFFFFULL);
        out[i] = {hi, lo};
    }
    states[idx] = rng;
}

int stream_main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "Usage: uuid-gpu stream SAMPLES BLOCKS THREADS_PER_BLOCK BATCH\n";
        return 2;
    }
    try {
        const unsigned long long samples = std::stoull(argv[2]);
        const unsigned long blocks = std::stoul(argv[3]);
        const unsigned long tpb = std::stoul(argv[4]);
        const unsigned long batch = std::stoul(argv[5]);
        if (!samples || !blocks || !tpb || tpb > 1024 || !batch || batch > 262144) {
            throw std::runtime_error("Invalid stream arguments");
        }
        CUDA_CHECK(cudaSetDevice(0));
        const size_t nthreads = static_cast<size_t>(blocks) * tpb;
        State* states = nullptr;
        RawUuid* device_ids = nullptr;
        CUDA_CHECK(cudaMalloc(&states, nthreads * sizeof(State)));
        CUDA_CHECK(cudaMalloc(&device_ids, batch * sizeof(RawUuid)));
        std::vector<RawUuid> host_ids(batch);
        std::random_device random;
        const unsigned long long seed = static_cast<unsigned long long>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count())
            ^ (static_cast<unsigned long long>(random()) << 32) ^ random();
        setup_rng<<<static_cast<unsigned int>(blocks), static_cast<unsigned int>(tpb)>>>(states, seed);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
#if defined(_WIN32)
        if (_setmode(_fileno(stdout), _O_BINARY) == -1) {
            throw std::runtime_error("Cannot enable binary stdout on Windows");
        }
#endif
        unsigned long long remaining = samples;
        while (remaining > 0) {
            const unsigned int count = static_cast<unsigned int>(
                std::min(remaining, static_cast<unsigned long long>(batch)));
            const auto unix_ms = static_cast<unsigned long long>(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count());
            stream_v7<<<static_cast<unsigned int>(blocks), static_cast<unsigned int>(tpb)>>>(
                states, device_ids, unix_ms, count);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(host_ids.data(), device_ids,
                                  count * sizeof(RawUuid), cudaMemcpyDeviceToHost));
            // fwrite can write fewer bytes on a broken pipe; treat it as failure.
            const size_t written = std::fwrite(host_ids.data(), sizeof(RawUuid), count, stdout);
            if (written != count || std::fflush(stdout) != 0) {
                throw std::runtime_error("Failed to stream CUDA UUIDs to parent process");
            }
            remaining -= count;
        }
        CUDA_CHECK(cudaFree(device_ids));
        CUDA_CHECK(cudaFree(states));
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "CUDA stream error: " << e.what() << '\n';
        return 2;
    }
}

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "stream") {
        return stream_main(argc, argv);
    }
    if (argc != 5) {
        std::cerr << "Usage: uuid-gpu SECONDS BLOCKS THREADS_PER_BLOCK ITERATIONS\n";
        return 2;
    }
    try {
        const unsigned long seconds = std::stoul(argv[1]);
        const unsigned long blocks = std::stoul(argv[2]);
        const unsigned long tpb = std::stoul(argv[3]);
        const unsigned long iterations = std::stoul(argv[4]);
        if (!seconds || !blocks || !tpb || tpb > 1024 || !iterations) {
            throw std::runtime_error("Invalid benchmark parameters");
        }
        cudaDeviceProp props{};
        CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
        CUDA_CHECK(cudaSetDevice(0));
        const size_t nthreads = static_cast<size_t>(blocks) * tpb;
        State* states = nullptr;
        unsigned long long* checksum = nullptr;
        CUDA_CHECK(cudaMalloc(&states, nthreads * sizeof(State)));
        CUDA_CHECK(cudaMalloc(&checksum, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(checksum, 0, sizeof(unsigned long long)));
        using Clock = std::chrono::steady_clock;
        const auto seed = static_cast<unsigned long long>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
        setup_rng<<<static_cast<unsigned int>(blocks), static_cast<unsigned int>(tpb)>>>(states, seed);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto start = Clock::now();
        const auto duration = std::chrono::seconds(seconds);
        unsigned long long total = 0;
        do {
            const auto timestamp = static_cast<unsigned long long>(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count());
            generate_v7<<<static_cast<unsigned int>(blocks), static_cast<unsigned int>(tpb)>>>(
                states, checksum, timestamp, static_cast<unsigned int>(iterations));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            total += static_cast<unsigned long long>(nthreads) * iterations;
        } while (Clock::now() - start < duration);
        const double elapsed = std::chrono::duration<double>(Clock::now() - start).count();
        unsigned long long host_checksum = 0;
        CUDA_CHECK(cudaMemcpy(&host_checksum, checksum, sizeof(host_checksum), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(states));
        CUDA_CHECK(cudaFree(checksum));
        // Machine-readable output consumed by the Rust CLI.
        std::cout.precision(9);
        std::cout << "GPU_RESULT|" << total << '|' << elapsed << '|' << host_checksum
                  << '|' << props.name << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 2;
    }
}
