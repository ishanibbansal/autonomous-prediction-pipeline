include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Keep clangd from borrowing third-party compile flags for project headers.
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.20")
  set_target_properties(gtest gtest_main gmock gmock_main PROPERTIES
    EXPORT_COMPILE_COMMANDS OFF
  )
endif()

add_executable(test_ingestion test_ingestion.cpp)
target_link_libraries(test_ingestion PRIVATE
  prediction_lib
  GTest::gtest_main
  Eigen3::Eigen
)

include(GoogleTest)
gtest_discover_tests(test_ingestion)

add_executable(test_csv_reader test_csv_reader.cpp)
target_link_libraries(test_csv_reader PRIVATE
  prediction_lib
  GTest::gtest_main
  Eigen3::Eigen
)
gtest_discover_tests(test_csv_reader)

add_executable(test_kalman_filter
  test_kalman_filter.cpp
  kalman_filter_linkage.cpp
)
target_link_libraries(test_kalman_filter PRIVATE
  prediction_lib
  GTest::gtest_main
  Eigen3::Eigen
)
gtest_discover_tests(test_kalman_filter)

add_executable(test_data_association
  test_data_association.cpp
  data_association_linkage.cpp
)
target_link_libraries(test_data_association PRIVATE
  prediction_lib
  GTest::gtest_main
  Eigen3::Eigen
)
gtest_discover_tests(test_data_association)

add_executable(test_tracker
  test_tracker.cpp
  tracker_linkage.cpp
)
target_link_libraries(test_tracker PRIVATE
  prediction_lib
  GTest::gtest_main
  Eigen3::Eigen
)
gtest_discover_tests(test_tracker)
