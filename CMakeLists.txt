cmake_minimum_required(VERSION 3.16)
project(hush_bus CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

add_executable(hush_bus_executor
    hush_bus_executor.cpp
    hush_engine_adapters.cpp
)

target_compile_definitions(hush_bus_executor PRIVATE HUSH_USE_REAL_ENGINE=1)

target_include_directories(hush_bus_executor PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/processor
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/nlohmann
)

target_link_libraries(hush_bus_executor PRIVATE
    hush_core
    Threads::Threads
)
