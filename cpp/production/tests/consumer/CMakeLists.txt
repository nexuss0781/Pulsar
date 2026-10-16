cmake_minimum_required(VERSION 3.20)
project(PulsarRuntimeConsumer LANGUAGES CXX)

find_package(PulsarRuntime 0.1.0 CONFIG REQUIRED)
add_executable(pulsar_runtime_consumer main.cpp)
target_link_libraries(pulsar_runtime_consumer PRIVATE Pulsar::pulsar_runtime)
