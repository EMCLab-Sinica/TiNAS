cmake_minimum_required(VERSION 2.8.12)

project(LEA_conv_tests)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_C_FLAGS "-fsanitize=address -Werror=undef -g3 -O0")

set(cnn_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/cnn/cnn.c
    ${CMAKE_CURRENT_SOURCE_DIR}/cnn/cnn_add.c
    ${CMAKE_CURRENT_SOURCE_DIR}/cnn/cnn_add_intermittent.c
    ${CMAKE_CURRENT_SOURCE_DIR}/cnn/cnn_batchnorm.c
    ${CMAKE_CURRENT_SOURCE_DIR}/cnn/cnn_batchnorm_intermittent.c
    ${CMAKE_CURRENT_SOURCE_DIR}/cnn/cnn_buffers.c
    ${CMAKE_CURRENT_SOURCE_DIR}/cnn/cnn_conv_tiled_dw.c
    ${CMAKE_CURRENT_SOURCE_DIR}/cnn/cnn_conv_tiled_dw_intermittent.c
    ${CMAKE_CURRENT_SOURCE_DIR}/cnn/cnn_conv_tiled_std.c
    ${CMAKE_CURRENT_SOURCE_DIR}/cnn/cnn_conv_tiled_std_intermittent.c
    ${CMAKE_CURRENT_SOURCE_DIR}/cnn/cnn_matops.c
    ${CMAKE_CURRENT_SOURCE_DIR}/cnn/cnn_pool.c
    ${CMAKE_CURRENT_SOURCE_DIR}/cnn/cnn_pool_intermittent.c
    ${CMAKE_CURRENT_SOURCE_DIR}/cnn/cnn_relu.c
    ${CMAKE_CURRENT_SOURCE_DIR}/cnn/cnn_relu_intermittent.c
    ${CMAKE_CURRENT_SOURCE_DIR}/cnn/cnn_utils.c
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/extfram_simulator.c
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/myuart_simulator.c
    ${CMAKE_CURRENT_SOURCE_DIR}/dsplib/source/matrix/msp_matrix_add_q15.c
    ${CMAKE_CURRENT_SOURCE_DIR}/dsplib/source/matrix/msp_matrix_mpy_q15.c
    ${CMAKE_CURRENT_SOURCE_DIR}/dsplib/source/vector/msp_add_q15.c
    ${CMAKE_CURRENT_SOURCE_DIR}/dsplib/source/vector/msp_offset_q15.c
    ${CMAKE_CURRENT_SOURCE_DIR}/dsplib/source/vector/msp_mac_q15.c
    ${CMAKE_CURRENT_SOURCE_DIR}/dsplib/source/vector/msp_mpy_q15.c
    ${CMAKE_CURRENT_SOURCE_DIR}/dsplib/source/vector/msp_sub_q15.c
    main_simulator.c
)

add_executable(LEA_conv_tests ${cnn_SOURCES})

target_compile_definitions(LEA_conv_tests
    PRIVATE
        SIMULATOR
)

target_include_directories(LEA_conv_tests
    SYSTEM PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/dsplib/include
)
target_include_directories(LEA_conv_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/fake-headers
)

# For sqrtf used in BatchNorm
target_link_libraries(LEA_conv_tests m)
