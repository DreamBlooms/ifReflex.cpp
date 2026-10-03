cmake_minimum_required(VERSION 3.14)
project(ifreflex-llama CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Pinned llama.cpp tree vendored under third_party/llama.cpp (release v0.5.0).
set(LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp" CACHE PATH "llama.cpp checkout")
if(NOT EXISTS "${LLAMA_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "llama.cpp not found at ${LLAMA_DIR}. Run: git submodule update --init --depth 1")
endif()

# The pinned submodule is upstream v0.5.0; the diffusion build is that base plus
# the patches in third_party/patches (DiffusionGemma PR #24423 + ifreflex's
# canvas split). Apply them at configure time so a fresh checkout builds without
# a manual step. The script is safe to rerun.
execute_process(
    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/scripts/apply_patches.sh" "${LLAMA_DIR}"
    RESULT_VARIABLE PATCH_RESULT)
if(NOT PATCH_RESULT EQUAL 0)
    message(FATAL_ERROR "scripts/apply_patches.sh failed (${PATCH_RESULT})")
endif()

set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD OFF CACHE BOOL "" FORCE)

option(IFREFLEX_STATIC "Link libraries statically" OFF)
if(IFREFLEX_STATIC)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
else()
    set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
endif()

option(IFREFLEX_CUDA   "Build with the CUDA backend"   OFF)
option(IFREFLEX_VULKAN "Build with the Vulkan backend" OFF)
option(IFREFLEX_HIP    "Build with the ROCm/HIP backend" OFF)
option(IFREFLEX_METAL  "Build with the Metal backend"  OFF)
if(IFREFLEX_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
endif()
if(IFREFLEX_VULKAN)
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
endif()
if(IFREFLEX_HIP)
    set(GGML_HIP ON CACHE BOOL "" FORCE)
endif()
if(IFREFLEX_METAL)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
endif()

add_subdirectory("${LLAMA_DIR}" "${CMAKE_BINARY_DIR}/llama.cpp")

# nlohmann/json is header-only and vendored under third_party/nlohmann so the
# build stays self-contained and cross-compilable (no host /usr/include leaks).
# The HTTP transport is adapted from lkarlslund/laya.cpp (MIT).
find_package(Threads REQUIRED)

add_executable(ifreflex-cli
    src/main.cpp
    src/engine.cpp
    src/diffusion.cpp
    src/diffusion_core.cpp
    src/diffusion_driver.cpp
    src/diffusion_gemma.cpp
    src/llada_moe.cpp
    src/prompt.cpp
    src/readout.cpp
    src/sys.cpp
    src/protocol.cpp
    src/http.cpp
)
target_include_directories(ifreflex-cli PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
)
target_link_libraries(ifreflex-cli PRIVATE llama Threads::Threads)
if(WIN32)
    target_link_libraries(ifreflex-cli PRIVATE ws2_32)  # Winsock for cpp-httplib
endif()

# Parity tests: numeric readout and prompt layout verified against the reference
# implementations (reflex, SemIf). Built and run via `ctest`.
enable_testing()
add_executable(test_readout scripts/parity/test_readout.cpp src/readout.cpp)
target_include_directories(test_readout PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
)
add_test(NAME readout_parity COMMAND test_readout)

# Offline harness for the diffusion structured readout (DiffusionGemma canvases).
add_executable(smoke_diffusion
    scripts/smoke_diffusion.cpp
    src/diffusion.cpp
    src/diffusion_core.cpp
    src/diffusion_driver.cpp
    src/diffusion_gemma.cpp
    src/llada_moe.cpp
    src/prompt.cpp
    src/readout.cpp
    src/sys.cpp
)
target_include_directories(smoke_diffusion PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
)
target_link_libraries(smoke_diffusion PRIVATE llama Threads::Threads)
if (WIN32)
    target_link_libraries(smoke_diffusion PRIVATE ws2_32)
endif()
