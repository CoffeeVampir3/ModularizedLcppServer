module;
#include <utility>
#include <vector>
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <iostream>
#include <thread>
#include <ranges>
#include <algorithm>
#include <optional>
#include <span>
#include <functional>
#include "llama.h"
export module LlamaServer.Processor;

import LlamaServer.Slot;
import LlamaServer.Presampler;
import LlamaServer.Request;
import LlamaServer.SequenceStream;
import LlamaServer.ReadbackBuffer;
import LlamaServer.InferenceArgs;
import LlamaServer.Tokenization;

/*
 * Primary server processor. Controls the overall flow. This processes in slot-order and does not
 * guarantee fairness in processing, to avoid overly shuffling the kv-cache.
 *
 * Provides:
 * The primary job-submit interface
 * Continuous batching aka High-efficiency Multi-user inference
 * Slot state management (Idle, Processing Prompt, Generating)
 * Slot Rewinding
 * Runs the actual llama model forward
 * Job cancellation
 *
 * Mechanism:
 * It's a server.
 */

export namespace LlamaServer {

class Processor {
    llama_model* model;
    llama_context* ctx;
    llama_batch batch{};
    std::atomic<bool> abort_inference = false;

    std::vector<Slot> slots;
    uint32_t batch_size;

    std::queue<Request> queue_tasks;
    std::mutex mutex_tasks;
    std::condition_variable cv_tasks;

    std::thread worker_thread;
    std::atomic<bool> should_exit{false};

    int current_job_index = 0;
    Tokenizer tokenizer;

    // nearly eq to common_add_to_batch from lcpp server
    void add_to_batch(Slot& slot, const llama_token token, const bool compute_logits) {
        slot.i_batch = batch.n_tokens;

        batch.token[batch.n_tokens] = token;
        batch.pos[batch.n_tokens] = slot.n_past;
        batch.n_seq_id[batch.n_tokens] = 1;
        batch.seq_id[batch.n_tokens][0] = slot.job_index;
        batch.logits[batch.n_tokens] = static_cast<int8_t>(compute_logits);

        batch.n_tokens++;
        slot.n_past++;
    }

    static llama_pos common_longest_prefix(const std::vector<llama_token>& a, const std::vector<llama_token>& b) {
        llama_pos i;
        for (i = 0; i < a.size() && i < b.size() && a[i] == b[i]; i++) {}
        return i;
    }

    //Tasks are not processed in fairness.
    //A task assigned to a slot sticks to it until finished to avoid shuffling the cache.
    //This is not a fair processing scheme, however it is more optimal
    void process_tasks() {
        // Check if any idle slot exists
        const bool has_idle_slot = std::ranges::any_of(slots, [](const auto& slot) {
            return slot.state == Slot::State::IDLE;
        });

        if (!has_idle_slot) {
            return;
        }

        std::unique_lock lock(mutex_tasks);
        if (queue_tasks.empty()) {
            return;
        }

        auto [id, prompt_tokens, inference_args, readback_buffer] = queue_tasks.front();
        queue_tasks.pop();
        lock.unlock();

        // Find the best slot with the longest prefix
        auto idle_slots = slots | std::views::filter([](const auto& slot) {
            return slot.state == Slot::State::IDLE;
        });

        // Find oldest idle slot
        const auto oldest_idle_it = std::ranges::min_element(idle_slots, {}, &Slot::job_index);
        Slot* oldest_idle_slot = oldest_idle_it != std::ranges::end(idle_slots) ? &*oldest_idle_it : nullptr;

        // Find best slot based on prefix
        std::pair<Slot*, llama_pos> best_match = {nullptr, 0};
        for (auto& slot : idle_slots) {
            const llama_pos prefix_len = common_longest_prefix(prompt_tokens, slot.prompt_tokens);
            const bool is_better = prefix_len > best_match.second ||
                                   (prefix_len == best_match.second &&
                                    (!best_match.first || slot.job_index < best_match.first->job_index));

            if (is_better) {
                best_match = {&slot, prefix_len};
            }
        }

        // If we do not have any prefix matches, pick the oldest idle slot
        Slot* best_slot = best_match.second > 0 ? best_match.first : oldest_idle_slot;
        if (!best_slot) return;

        if (best_match.second > 0) {
            // Reuse prefix, cut the KV to the prefix size and adjust to gen or prompt appropriately
            llama_kv_self_seq_rm(ctx, best_slot->job_index, best_match.second, -1);
            best_slot->prompt_tokens_processed = best_match.second;
            best_slot->tokens_generated = 0;
            best_slot->generated_text.clear();
            best_slot->state =
                best_match.second == prompt_tokens.size() ?
                Slot::State::GENERATING :
                Slot::State::PROMPT;
        } else {
            // Nothing to reuse, clear the kv and start fresh
            best_slot->clear(ctx);
            best_slot->prompt_tokens_processed = 0;
            best_slot->state = Slot::State::PROMPT;
        }

        best_slot->request_id = id;
        best_slot->prompt_tokens = prompt_tokens;
        best_slot->inference_args = inference_args;
        best_slot->readback_buffer = readback_buffer;

        best_slot->sequence_stream->bind_sequences(inference_args.stopping_strings, inference_args.rewind_strings);
        best_slot->rewind_snapshot = Slot::SlotSnapshot::snapshot_slot(*best_slot, ctx);

        // Ban the EOS tokens immediately before starting generation if we have min tokens
        if (inference_args.min_tokens_to_gen > 0 && inference_args.min_tokens_to_gen < inference_args.max_tokens_to_gen) {
            const std::vector terminal_token_bans {llama_vocab_eos(llama_model_get_vocab(model)), llama_vocab_eot(llama_model_get_vocab(model))};
            best_slot->presampler.add_eos_ban(model, terminal_token_bans);
        }
    }

    void update_prompt_slots() {
        for (auto& slot : slots) {
            if (slot.is_processing_prompt() && slot.prompt_tokens_processed < slot.prompt_tokens.size()) {
                while (slot.prompt_tokens_processed < slot.prompt_tokens.size() &&
                       batch.n_tokens < batch_size) {

                    const llama_token token = slot.prompt_tokens[slot.prompt_tokens_processed];
                    const bool is_last_prompt_token = (slot.prompt_tokens_processed == slot.prompt_tokens.size() - 1);

                    slot.i_batch = batch.n_tokens;
                    add_to_batch(slot, token, is_last_prompt_token);
                    slot.prompt_tokens_processed++;
                }

                if (slot.prompt_tokens_processed >= slot.prompt_tokens.size()) {
                    slot.state = Slot::State::GENERATING;
                }
            }
        }
    }

    // Processes the next sequence token. Finalizes the request if gen is finished.
    bool process_token(Slot& slot, const llama_token token) const {
        const auto piece_opt = slot.detokenizer->process_token(token, true);
        if (!piece_opt) {
            std::cerr << "Error: Failed to process token " << token << std::endl;
            return false;
        }

        slot.tokens_generated++;

        const bool is_eos = tokenizer.is_eos_token(token);
        bool is_complete = is_eos || slot.tokens_generated >= slot.inference_args.max_tokens_to_gen;
        bool yield_final = false;
        const std::string& piece = piece_opt.value_or("");

        if (!piece.empty()) {
            std::string out_string;

            switch (slot.sequence_stream->append(piece, token, out_string)) {
                case SequenceStream::Continuation::ACCEPT:
                    slot.generated_text += out_string;
                    slot.presampler.clear_rewind_bans(model);
                    slot.rewind_snapshot = Slot::SlotSnapshot::snapshot_slot(slot, ctx);
                    yield_final = true;

                    if (slot.inference_args.min_tokens_to_gen > 0
                        && slot.tokens_generated >= slot.inference_args.min_tokens_to_gen
                        && slot.inference_args.min_tokens_to_gen < slot.inference_args.max_tokens_to_gen) {
                        slot.presampler.clear_eos_bans(model);
                    }
                    break;
                case SequenceStream::Continuation::REWIND: {
                    //Restore the slot to whatever the last accepted snapshot was.
                    //Then delete the part of the KV we're rewinding
                    const int32_t prev_kv_pos = slot.rewind_snapshot.rewind_slot(slot);
                    llama_kv_self_seq_rm(ctx, slot.job_index, prev_kv_pos, -1);

                    //Ban every token in the buffer.
                    const auto tokens = tokenizer.tokenize(out_string, false, false);
                    if (!tokens || tokens->empty()) {
                        std::cerr << "Error: Failed to tokenize rewind buffer" << std::endl;
                        return false;
                    }
                    slot.presampler.add_rewind_bans(model, tokens.value());

                    //It's possible we rewind to before the min token threshold, so we need to ensure the eos tokens are actually banned.
                    if (slot.inference_args.min_tokens_to_gen > 0 && slot.inference_args.min_tokens_to_gen < slot.inference_args.max_tokens_to_gen) {
                        const std::vector terminal_token_bans {llama_vocab_eos(llama_model_get_vocab(model)), llama_vocab_eot(llama_model_get_vocab(model))};
                        slot.presampler.add_eos_ban(model, terminal_token_bans);
                    }
                    }
                    return true;
                case SequenceStream::Continuation::STOP:
                    is_complete = true;
                    break;
                case SequenceStream::Continuation::BUFFER:
                    break;
            }
        }

        if (!is_complete) {
            if (!piece.empty()) {
                readback_write_to_buffer(slot.readback_buffer, piece, token);
            }
            return !is_eos;
        }

        std::string final_piece = piece;
        if (slot.detokenizer->has_incomplete()) {
            if (const std::string remaining = slot.detokenizer->flush(); !remaining.empty())
                final_piece += remaining;
        }

        if (yield_final && !final_piece.empty()) {
            readback_write_to_buffer(slot.readback_buffer, final_piece, token);
        }

        readback_finish(slot.readback_buffer, R"({"status": "Job Completed", "reason": "Normal completion"})");
        return false;
    }

    void update_gen_slots() {
        // Add tokens to the batch for generating slots
        for (auto& slot : slots | std::views::filter(&Slot::is_generating)) {
            if (batch.n_tokens < batch_size) {
                add_to_batch(slot, slot.last_token, true);
            }
        }

        if (batch.n_tokens == 0) {
            return;
        }

        if (llama_decode(ctx, batch) != 0) {
            std::cerr << "Error: Failed to decode batch" << std::endl;
            return;
        }

        // Process each slot that has a valid batch index
        for (auto& slot : slots) {
            if (slot.i_batch < 0 || slot.i_batch >= batch.n_tokens) {
                continue;
            }

            if (slot.is_generating()) {
                llama_token token;

                // If we have a presampler, we append our main sampler to it, otherwise we just use our main sampler.
                if (slot.presampler.should_presample) {
                    const int presampler_tail = llama_sampler_chain_n(slot.presampler.sampler);
                    llama_sampler_chain_add(slot.presampler.sampler, slot.inference_args.sampler);
                    token = llama_sampler_sample(slot.presampler.sampler, ctx, slot.i_batch);
                    llama_sampler_chain_remove(slot.presampler.sampler, presampler_tail);
                } else {
                    token = llama_sampler_sample(slot.inference_args.sampler, ctx, slot.i_batch);
                }
                llama_sampler_accept(slot.inference_args.sampler, token);
                slot.last_token = token;
                slot.i_batch = -1;

                if (!process_token(slot, token)) {
                    slot.end(++current_job_index);
                }
            }
        }
    }

    void update_slots() {
        batch.n_tokens = 0;

        update_prompt_slots();
        update_gen_slots();
    }

    void run() {
        while (!should_exit) {
            process_tasks();
            update_slots();

            const bool all_idle = std::ranges::all_of(slots, [](const auto& slot) {
                return !slot.is_processing();
            });

            if (all_idle) {
                std::unique_lock lock(mutex_tasks);
                if (queue_tasks.empty()) {
                    cv_tasks.wait(lock, [this]() {
                        return !queue_tasks.empty() || should_exit;
                    });
                }
            }
        }
    }

public:
    Processor(llama_model* model, llama_context* ctx, const int num_slots = 4)
        : model(model), ctx(ctx), tokenizer(model, ctx) {

        batch_size = llama_n_batch(ctx);
        batch = llama_batch_init(static_cast<int32_t>(batch_size), 0, 1);

        // Create slots and initialize them
        slots.reserve(num_slots);
        for (int i = 0; i < num_slots; i++) {
            slots.emplace_back(model);
            slots.back().job_index = (++current_job_index);
            slots.back().clear(ctx);
        }

        worker_thread = std::thread(&Processor::run, this);
        auto inference_abort_callback = [](void* data) -> bool {
            // Abort inference and reset the abort toggle.
            if (const auto abort_flag = static_cast<bool*>(data); *abort_flag) {
                *abort_flag = false;
                return true;
            }
            return false;
        };
        llama_set_abort_callback(ctx, inference_abort_callback, &abort_inference);
    }

    ~Processor() {
        should_exit = true;
        cv_tasks.notify_all();
        if (worker_thread.joinable()) {
            worker_thread.join();
        }
        llama_batch_free(batch);
    }

    bool cancel_work(const int request_id_to_cancel) {
        bool found = false;

        // Is our job pending in the request queue? If so, remove it.
        {
            std::lock_guard lock(mutex_tasks);
            if (!queue_tasks.empty()) {
                std::queue<Request> new_queue;

                while (!queue_tasks.empty()) {
                    Request req = queue_tasks.front();
                    queue_tasks.pop();

                    if (req.id != request_id_to_cancel) {
                        new_queue.push(req);
                    } else {
                        if (req.readback_buffer) {
                            readback_finish(req.readback_buffer, R"({"status": "Job Cancelled", "reason": "User requested cancellation"})");
                        }
                        found = true;
                    }
                }

                queue_tasks = std::move(new_queue);
            }
        }

        // Check if any slots are processing
        const bool was_any_processing = std::ranges::any_of(slots, &Slot::is_processing);

        // Check all slots for the job just in case of a race
        for (auto& slot : slots) {
            if (slot.request_id == request_id_to_cancel) {
                if (slot.readback_buffer) {
                    readback_finish(slot.readback_buffer, R"({"status": "Job Cancelled", "reason": "User requested cancellation"})");
                }
                slot.clear(ctx);
                slot.job_index = ++current_job_index;
                found = true;
            }
        }

        // If we have no running slots and no requests in the queue, abort any queued inferences
        if (!was_any_processing) return found;

        const bool all_idle = std::ranges::all_of(slots, [](const auto& slot) {
            return !slot.is_processing();
        });

        if (queue_tasks.empty() && all_idle) {
            // Abort inference is reset via the mechanism in the lambda abort fn
            abort_inference = true;
        }

        return found;
    }

    int submit_work(
        const std::string& prompt,
        const InferenceArgs& args,
        ReadbackBuffer* readback_buffer) {
        const auto tokens_opt = tokenizer.tokenize(prompt, args.max_tokens_to_gen);
        if (!tokens_opt) {
            std::cerr << "Error: Failed to tokenize prompt" << std::endl;
            return -1;
        }

        const std::vector<llama_token>& prompt_tokens = tokens_opt.value();
        static int next_id = 1;
        const int request_id = next_id++;

        {
            const Request request{request_id, prompt_tokens, args, readback_buffer};
            std::lock_guard lock(mutex_tasks);
            queue_tasks.push(request);
        }

        cv_tasks.notify_one();
        return request_id;
    }
};

}