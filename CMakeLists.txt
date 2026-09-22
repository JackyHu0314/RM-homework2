cmake_minimum_required(VERSION 3.16)

project(rm_homework2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs videoio)

add_library(vision_common INTERFACE)
target_include_directories(vision_common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(vision_common INTERFACE ${OpenCV_LIBS})

add_executable(task1_image src/task1_image/main.cpp)
target_link_libraries(task1_image PRIVATE vision_common)

add_executable(task2_fit src/task2_fit/main.cpp)
target_link_libraries(task2_fit PRIVATE vision_common)

add_executable(task3_windmill src/task3_windmill/main.cpp)
target_link_libraries(task3_windmill PRIVATE vision_common)

include(CTest)
if(BUILD_TESTING)
    add_test(
        NAME task1_generates_all_outputs
        COMMAND task1_image
            ${CMAKE_CURRENT_SOURCE_DIR}/resources/test_image.jpg
            ${CMAKE_CURRENT_BINARY_DIR}/task1_test_output
    )
endif()
