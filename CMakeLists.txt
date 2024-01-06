cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(BackgroundSubtraction)

set(Torch_CUDA_ENABLED OFF)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(OpenCV REQUIRED)
set(OpenCV_DIR "" CACHE PATH "Path to OpenCV directory")

add_executable(BackgroundSubtraction main.cpp)
target_link_libraries(BackgroundSubtraction "${TORCH_LIBRARIES}" ${OpenCV_LIBS})
set_property(TARGET BackgroundSubtraction PROPERTY CXX_STANDARD 17)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET BackgroundSubtraction
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:BackgroundSubtraction>)
endif (MSVC)