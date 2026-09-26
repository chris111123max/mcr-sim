cmake_minimum_required(VERSION 3.12)
project(MCRBeamFeasibleHook VERSION 0.1 LANGUAGES CXX)

find_package(Sofa.Core REQUIRED)
find_package(Sofa.Simulation.Core REQUIRED)

add_library(MCRBeamFeasibleHook SHARED
    NativePrecommitMappingHook.cpp
)

target_compile_features(MCRBeamFeasibleHook PRIVATE cxx_std_17)
target_link_libraries(MCRBeamFeasibleHook
    PRIVATE
        Sofa.Core
        Sofa.Simulation.Core
)

set_target_properties(MCRBeamFeasibleHook PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)
