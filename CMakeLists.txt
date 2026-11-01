cmake_minimum_required(VERSION 3.16)

project(autonomous_prediction_pipeline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED)

add_library(prediction_lib INTERFACE)
target_include_directories(prediction_lib INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(prediction_lib INTERFACE Eigen3::Eigen)

# Keep warning flags on project-owned targets so fetched third-party targets
# such as GoogleTest are not compiled under -Wconversion / -Wshadow.
if(MSVC)
  target_compile_options(prediction_lib INTERFACE /W4)
else()
  target_compile_options(prediction_lib INTERFACE
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -Wshadow
  )
endif()

enable_testing()
add_subdirectory(tests)
