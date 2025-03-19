cmake_minimum_required(VERSION 3.28)
project(LlamaMultiUserInference)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN TRUE)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)
set(CMAKE_BUILD_TYPE Release)

include(FetchContent)
FetchContent_Declare(
        llama
        GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
        GIT_TAG master
)

set(GGML_CUDA ON CACHE BOOL "llama.cpp: use CUDA" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "llama.cpp: build examples" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "llama.cpp: build tests" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "llama.cpp: build server" FORCE)

FetchContent_MakeAvailable(llama)

set(SERVER_MODULE_FILES
        modules/server/inference_args.cpp
        modules/server/presampler.cpp
        modules/server/readback_buffer.cpp
        modules/server/request.cpp
        modules/server/slot.cpp
        modules/server/trie.cpp
        modules/server/processor.cpp
        modules/server/samplers.cpp
        modules/server/sequence_stream.cpp
        modules/server/tokenization.cpp
        modules/server/front_end.cpp
)

add_executable(llama_example main.cpp ${SERVER_MODULE_FILES})

target_sources(llama_example
        PRIVATE
        FILE_SET CXX_MODULES
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/modules/server
        FILES ${SERVER_MODULE_FILES}
)

target_compile_features(llama_example PRIVATE cxx_std_23)
target_include_directories(llama_example PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(llama_example PRIVATE llama ggml ggml-cpu ggml-cuda ggml-base)