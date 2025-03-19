module;
#include <string>
#include <vector>
#include <optional>
#include <string_view>
#include "llama.h"
export module LlamaServer.Tokenization;

/*
 *  This tokenizer extends on the lcpp tokenizer to support utf8 and partial tokens (Utf-8 tokens can be multiple
 *  actual outputs from the model)
 *
 *  Provides:
 *  General tokenization and Utf-8 aware streaming tokenization.
 *
 *  Mechanism:
 *  A buffer to hold incomplete tokens until we obtain the fully formed token.
 */

inline size_t validate_utf8(const std::string_view str) {
    const auto* bytes = reinterpret_cast<const uint8_t*>(str.data());
    const size_t len = str.size();
    size_t i = 0;

    while (i < len) {
        if (bytes[i] <= 0x7F) {
            // Single byte character (0xxxxxxx)
            i++;
        } else if ((bytes[i] & 0xE0) == 0xC0) {
            // (110xxxxx 10xxxxxx)
            if (i + 1 >= len || (bytes[i+1] & 0xC0) != 0x80) {
                return i;
            }
            i += 2;
        } else if ((bytes[i] & 0xF0) == 0xE0) {
            // (1110xxxx 10xxxxxx 10xxxxxx)
            if (i + 2 >= len || (bytes[i+1] & 0xC0) != 0x80 || (bytes[i+2] & 0xC0) != 0x80) {
                return i;
            }
            i += 3;
        } else if ((bytes[i] & 0xF8) == 0xF0) {
            // (11110xxx 10xxxxxx 10xxxxxx 10xxxxxx)
            if (i + 3 >= len || (bytes[i+1] & 0xC0) != 0x80 || (bytes[i+2] & 0xC0) != 0x80 || (bytes[i+3] & 0xC0) != 0x80) {
                return i;
            }
            i += 4;
        } else {
            // Invalid
            return i;
        }
    }

    return len;
}

export namespace LlamaServer {

class TokenStreamDetokenizer {
    std::string utf_buffer;
    const llama_model* model;

    [[nodiscard]] std::optional<std::string> token_to_piece(const llama_token token, const bool special) const {
        if (!model) {
            return std::nullopt;
        }

        const auto* vocab = llama_model_get_vocab(model);

        std::string piece(64, '\0');

        int n_chars = llama_token_to_piece(vocab, token, piece.data(), static_cast<int32_t>(piece.size()), 0, special);

        if (n_chars < 0) {
            piece.resize(-n_chars);
            n_chars = llama_token_to_piece(vocab, token, piece.data(), static_cast<int32_t>(piece.size()), 0, special);
            if (n_chars < 0) {
                return std::nullopt;
            }
        }

        piece.resize(n_chars);

        return piece;
    }

public:
    explicit TokenStreamDetokenizer(const llama_model* model)
        : model(model) {
    }

    std::optional<std::string> process_token(const llama_token token, const bool parse_special) {
        const auto piece = token_to_piece(token, parse_special);
        if (!piece) {
            return std::nullopt;
        }

        utf_buffer += *piece;

        const size_t valid_bytes = validate_utf8(utf_buffer);

        if (valid_bytes == 0) {
            return std::string{};
        }

        if (valid_bytes == utf_buffer.size()) {
            std::string result = std::move(utf_buffer);
            utf_buffer.clear();
            return result;
        }

        std::string result = utf_buffer.substr(0, valid_bytes);
        utf_buffer = utf_buffer.substr(valid_bytes);
        return result;
    }

    std::string flush() {
        std::string result = std::move(utf_buffer);
        utf_buffer.clear();
        return result;
    }

    [[nodiscard]] bool has_incomplete() const {
        return !utf_buffer.empty();
    }

    void reset() {
        utf_buffer.clear();
    }
};

class Tokenizer {
    llama_model* model;
    llama_context* ctx;

public:
    Tokenizer(llama_model* model, llama_context* ctx)
        : model(model), ctx(ctx)
    {
    }

    [[nodiscard]] bool is_eos_token(const llama_token token) const {
        return token == llama_vocab_eos(llama_model_get_vocab(model));
    }

    [[nodiscard]] std::optional<std::vector<llama_token>> tokenize(const std::string_view& text, const bool add_special = true, const bool parse_special = true) const {
        const int n_prompt = -llama_tokenize(
            llama_model_get_vocab(model),
            text.data(),
            static_cast<int32_t>(text.size()),
            nullptr,
            0,
            add_special,
            parse_special);

        std::vector<llama_token> tokens(n_prompt);

        if (llama_tokenize(
            llama_model_get_vocab(model),
            text.data(),
            static_cast<int32_t>(text.size()),
            tokens.data(),
            static_cast<int32_t>(tokens.size()),
            add_special,
            parse_special) < 0) {
            return std::nullopt;
        }

        return tokens;
    }
};

}