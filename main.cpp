#include "llama.h"
#include <chrono>
#include <thread>
#include <iostream>

import LlamaServer;
import LlamaServer.Samplers;
import LlamaServer.ReadbackBuffer;
import LlamaServer.Processor;

using namespace LlamaServer;

void run_example() {
    const auto model_path = "/home/blackroot/Desktop/Llama-Agents/Llama-3.2-1B-Instruct-IQ4_XS.gguf";

    float tensor_split = 0.0f;
    const auto model = model_load(model_path, -1, &tensor_split, nullptr);
    if (!model) {
        std::cerr <<"Failed to load model.";
        return;
    }

    const auto ctx = ctx_make(model, 4096, -1, 512, false, -1, false, 0, 0, 0.0f);
    if (!ctx) {
        std::cerr <<"Failed to make a context";
        model_free(model);
        return;
    }

    const auto processor = processor_make(model, ctx, 4);

    llama_sampler* sampler = sampler_make();
    sampler = sampler_temp(sampler, 0.7f);      // Temperature of 0.7
    sampler = sampler_top_p(sampler, 0.9f, 1);  // Top-p sampling
    sampler = sampler_dist(sampler, 1337);      // Set random seed

    ReadbackBuffer* readback = readback_create_buffer();

    const auto prompt = "Write a short poem about programming in C++:";
    std::cout << prompt;
    std::flush(std::cout);

    const int job_id = processor_submit_work(
        processor,
        prompt,
        sampler,
        readback,
        200,         // max tokens to generate
        0,           // min tokens to generate
        1337,        // seed
        nullptr, 0,  // rewind strings
        nullptr, 0,  // stopping strings
        nullptr, 0   // stopping tokens
    );

    bool job_finished = false;
    while (!job_finished) {
        job_finished = readback_is_buffer_finished(readback);

        char* token_text;
        llama_token token_id;
        while (readback_read_next(readback, &token_text, &token_id)) {
            std::cout << token_text;
            std::flush(std::cout);
        }

        if (!job_finished) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    if (const char* status = readback_read_status(readback)) {
        std::cout << status;
        std::flush(std::cout);
    }

    // Clean up resources
    sampler_free(sampler);
    ctx_free(ctx);
    model_free(model);
}

int main() {
    llama_backend_init();
    run_example();
    llama_backend_free();
    return 0;
}