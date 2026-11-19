#include "llama.h"
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <dlfcn.h>
#include <thread>

// llama.cpp a956a85 schedules a llama_process call per bounded prefill or decode batch.
extern "C" int32_t llama_process(llama_context *ctx, llama_process_type type, llama_batch_ext *batch) {
    static auto process = reinterpret_cast<decltype(&llama_process)>(dlsym(RTLD_NEXT, "llama_process"));
    static auto sync = reinterpret_cast<decltype(&llama_synchronize)>(dlsym(RTLD_NEXT, "llama_synchronize"));
    static const double duty = [] {
        const char *setting = std::getenv("MEMSKILL_DECODE_DUTY");
        const double value = setting ? std::atof(setting) : 0.4;
        if (!(value > 0 && value <= 0.65)) std::abort();
        return value;
    }();
    if (!process || !sync) std::abort();
    static unsigned long calls = 0;
    if (calls++ % 4096 == 0) std::fprintf(stderr, "Limited compute: duty=%.2f calls=%lu\n", duty, calls);
    const auto start = std::chrono::steady_clock::now();
    const int32_t result = process(ctx, type, batch);
    sync(ctx);
    const auto busy = std::chrono::steady_clock::now() - start;
    std::this_thread::sleep_for(busy * (1.0/duty - 1.0));
    return result;
}
