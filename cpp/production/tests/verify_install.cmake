foreach(required_var BUILD_DIR PREFIX WORK_DIR SOURCE_DIR)
    if(NOT DEFINED ${required_var})
        message(FATAL_ERROR "${required_var} must be supplied")
    endif()
endforeach()

execute_process(
    COMMAND "${CMAKE_COMMAND}" --install "${BUILD_DIR}" --prefix "${PREFIX}"
    RESULT_VARIABLE install_result OUTPUT_VARIABLE install_stdout ERROR_VARIABLE install_stderr)
if(NOT install_result EQUAL 0)
    message(FATAL_ERROR "Runtime installation failed:\n${install_stdout}\n${install_stderr}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" -S "${SOURCE_DIR}/consumer" -B "${WORK_DIR}"
            "-DCMAKE_PREFIX_PATH=${PREFIX}"
    RESULT_VARIABLE configure_result OUTPUT_VARIABLE configure_stdout ERROR_VARIABLE configure_stderr)
if(NOT configure_result EQUAL 0)
    message(FATAL_ERROR "Downstream consumer configure failed:\n${configure_stdout}\n${configure_stderr}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" --build "${WORK_DIR}" --config Release --parallel 2
    RESULT_VARIABLE build_result OUTPUT_VARIABLE build_stdout ERROR_VARIABLE build_stderr)
if(NOT build_result EQUAL 0)
    message(FATAL_ERROR "Downstream consumer build failed:\n${build_stdout}\n${build_stderr}")
endif()

set(consumer_binary "${WORK_DIR}/pulsar_runtime_consumer")
if(NOT EXISTS "${consumer_binary}")
    set(consumer_binary "${WORK_DIR}/Release/pulsar_runtime_consumer")
endif()
execute_process(
    COMMAND "${consumer_binary}"
    RESULT_VARIABLE consumer_result OUTPUT_VARIABLE consumer_stdout ERROR_VARIABLE consumer_stderr)
if(NOT consumer_result EQUAL 0)
    message(FATAL_ERROR "Downstream consumer failed:\n${consumer_stdout}\n${consumer_stderr}")
endif()

if(EXISTS "${PREFIX}/bin/pulsar-generate")
    execute_process(
        COMMAND "${PREFIX}/bin/pulsar-generate" --help
        RESULT_VARIABLE cli_result OUTPUT_VARIABLE cli_stdout ERROR_VARIABLE cli_stderr)
    if(NOT cli_result EQUAL 0 OR NOT cli_stdout MATCHES "Pulsar native CPU text generation")
        message(FATAL_ERROR "Installed CLI smoke check failed:\n${cli_stdout}\n${cli_stderr}")
    endif()
endif()

message(STATUS "Installed package and downstream consumer: PASS")
