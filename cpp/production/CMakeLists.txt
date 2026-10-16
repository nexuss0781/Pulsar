cmake_minimum_required(VERSION 3.20)
project(PulsarProductionRuntime VERSION 0.1.0 LANGUAGES CXX)

option(PULSAR_BUILD_LLAMA_BACKEND "Build the GGUF-backed llama.cpp executable" ON)
option(PULSAR_FETCH_LLAMA_CPP "Fetch the pinned llama.cpp source when no local source tree is supplied" ON)
set(PULSAR_LLAMA_CPP_SOURCE_DIR "" CACHE PATH "Existing llama.cpp source tree (optional)")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(pulsar_runtime INTERFACE)
target_include_directories(pulsar_runtime INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:include>)
target_compile_features(pulsar_runtime INTERFACE cxx_std_17)
install(TARGETS pulsar_runtime EXPORT PulsarRuntimeTargets)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(pulsar_runtime INTERFACE -Wall -Wextra -Wpedantic)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(pulsar_runtime_tests tests/runtime_tests.cpp)
    target_link_libraries(pulsar_runtime_tests PRIVATE pulsar_runtime)
    add_executable(pulsar_tree_v3_tests "${CMAKE_CURRENT_SOURCE_DIR}/../pulsar_tree_v3_test.cpp")
    target_link_libraries(pulsar_tree_v3_tests PRIVATE pulsar_runtime)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(pulsar_runtime_tests PRIVATE -Werror)
        target_compile_options(pulsar_tree_v3_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(NAME pulsar_runtime_tests COMMAND pulsar_runtime_tests)
    add_test(NAME pulsar_tree_v3_tests COMMAND pulsar_tree_v3_tests)
endif()

if(PULSAR_BUILD_LLAMA_BACKEND)
    if(PULSAR_LLAMA_CPP_SOURCE_DIR)
        if(NOT EXISTS "${PULSAR_LLAMA_CPP_SOURCE_DIR}/CMakeLists.txt")
            message(FATAL_ERROR "PULSAR_LLAMA_CPP_SOURCE_DIR must point to a llama.cpp source tree")
        endif()
        set(_pulsar_llama_source "${PULSAR_LLAMA_CPP_SOURCE_DIR}")
    elseif(PULSAR_FETCH_LLAMA_CPP)
        include(FetchContent)
        FetchContent_Declare(llama_cpp
            GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
            GIT_TAG cd0a5b6a180dff0048e5fd993098b21c0b2c91b8
            GIT_SHALLOW FALSE)
        FetchContent_GetProperties(llama_cpp)
        if(NOT llama_cpp_POPULATED)
            FetchContent_Populate(llama_cpp)
        endif()
        set(_pulsar_llama_source "${llama_cpp_SOURCE_DIR}")
    else()
        message(FATAL_ERROR
            "llama.cpp source is required. Set PULSAR_LLAMA_CPP_SOURCE_DIR or enable PULSAR_FETCH_LLAMA_CPP.")
    endif()

    # Build a portable CPU library. Each release build can then be compiled on
    # x86_64 or ARM64 without embedding the builder's -march=native settings.
    set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_UI OFF CACHE BOOL "" FORCE)
    set(GGML_CCACHE OFF CACHE BOOL "" FORCE)
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    set(GGML_CPU_ALL_VARIANTS OFF CACHE BOOL "" FORCE)
    set(GGML_CUDA OFF CACHE BOOL "" FORCE)
    set(GGML_HIP OFF CACHE BOOL "" FORCE)
    set(GGML_METAL OFF CACHE BOOL "" FORCE)
    set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
    set(GGML_SYCL OFF CACHE BOOL "" FORCE)
    set(GGML_OPENCL OFF CACHE BOOL "" FORCE)
    set(GGML_BLAS OFF CACHE BOOL "" FORCE)
    set(GGML_RPC OFF CACHE BOOL "" FORCE)
    add_subdirectory("${_pulsar_llama_source}" "${CMAKE_CURRENT_BINARY_DIR}/llama.cpp" EXCLUDE_FROM_ALL)

    add_executable(pulsar-generate
        src/main.cpp
        src/llama_model.cpp)
    target_include_directories(pulsar-generate PRIVATE src ${CMAKE_CURRENT_SOURCE_DIR}/..)
    target_link_libraries(pulsar-generate PRIVATE pulsar_runtime llama)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(pulsar-generate PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_executable(pulsar-phase3-bench
        src/phase3_bench.cpp
        src/llama_model.cpp)
    target_include_directories(pulsar-phase3-bench PRIVATE src ${CMAKE_CURRENT_SOURCE_DIR}/..)
    target_link_libraries(pulsar-phase3-bench PRIVATE pulsar_runtime llama)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(pulsar-phase3-bench PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_executable(pulsar-phase4-bench
        src/phase4_bench.cpp
        src/llama_model.cpp)
    target_include_directories(pulsar-phase4-bench PRIVATE src ${CMAKE_CURRENT_SOURCE_DIR}/..)
    target_link_libraries(pulsar-phase4-bench PRIVATE pulsar_runtime llama)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(pulsar-phase4-bench PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
    set_target_properties(pulsar-generate PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
    set_target_properties(pulsar-phase3-bench PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
    install(TARGETS pulsar-generate RUNTIME DESTINATION bin)
    # The upstream PUBLIC_HEADER lists are source-relative and become relative
    # to this parent project in an install(TARGETS ...) call. The CLI package
    # does not expose llama/ggml headers, so install only the runtime libraries.
    set_target_properties(llama ggml PROPERTIES PUBLIC_HEADER "")
    install(TARGETS llama ggml ggml-base ggml-cpu
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)
    set_target_properties(llama ggml ggml-base ggml-cpu PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()

install(DIRECTORY include/pulsar DESTINATION include)
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/../pulsar_core_v1.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../pulsar_speculative_v2.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../pulsar_tree_v3.hpp"
    DESTINATION include)
install(EXPORT PulsarRuntimeTargets
    FILE PulsarRuntimeTargets.cmake
    NAMESPACE Pulsar::
    DESTINATION lib/cmake/PulsarRuntime)
