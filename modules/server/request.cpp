module;
#include <vector>
#include "llama.h"
export module LlamaServer.Request;

import LlamaServer.InferenceArgs;
import LlamaServer.ReadbackBuffer;
/*
 * A light abstraction over a request to fill a slot. This pends in a queue until we have free slots to take
 * the next request.
 */

export namespace LlamaServer {

struct Request {
    int id;
    std::vector<llama_token> prompt_tokens;
    InferenceArgs inference_args;
    ReadbackBuffer* readback_buffer;
};

}