cmake_minimum_required(VERSION 3.12)
project(MCRBeamFeasibleHook VERSION 0.1 LANGUAGES CXX)

# The server's SOFA 21.12 tree has usable shared libraries and headers, but its
# generated package Config files refer to uninstalled target files. Import the
# two required libraries directly; this remains test-only and avoids touching
# or reinstalling the shared SOFA environment.
set(SOFA_PREFIX "$ENV{SOFA_ROOT}" CACHE PATH "SOFA 21.12 build/install prefix")
set(SOFA_SOURCE_ROOT "$ENV{SOFA_SRC}/sofa" CACHE PATH "SOFA source tree")

foreach(_required
    "${SOFA_PREFIX}/lib/libSofa.Core.so"
    "${SOFA_PREFIX}/lib/libSofa.SimulationCore.so"
    "${SOFA_SOURCE_ROOT}/SofaKernel/modules/SofaCore/src/sofa/core/VecId.h"
    "${SOFA_SOURCE_ROOT}/SofaKernel/modules/SofaSimulationCore/src/sofa/simulation/mechanicalvisitor/MechanicalPropagateOnlyPositionAndVelocityVisitor.h"
    "${SOFA_PREFIX}/include/Sofa.Config/sofa/config.h")
    if(NOT EXISTS "${_required}")
        message(FATAL_ERROR "Required SOFA 21.12 development file is unavailable: ${_required}")
    endif()
endforeach()

add_library(Sofa.Core SHARED IMPORTED GLOBAL)
set_target_properties(Sofa.Core PROPERTIES
    IMPORTED_LOCATION "${SOFA_PREFIX}/lib/libSofa.Core.so"
)
add_library(Sofa.SimulationCore SHARED IMPORTED GLOBAL)
set_target_properties(Sofa.SimulationCore PROPERTIES
    IMPORTED_LOCATION "${SOFA_PREFIX}/lib/libSofa.SimulationCore.so"
)

file(GLOB _sofa_module_include_dirs LIST_DIRECTORIES true
    "${SOFA_SOURCE_ROOT}/SofaKernel/modules/*/src"
)
file(GLOB _sofa_generated_include_dirs LIST_DIRECTORIES true
    "${SOFA_PREFIX}/include/*"
)

add_library(MCRBeamFeasibleHook SHARED
    NativePrecommitMappingHook.cpp
)
target_compile_features(MCRBeamFeasibleHook PRIVATE cxx_std_17)
target_include_directories(MCRBeamFeasibleHook PRIVATE
    ${_sofa_module_include_dirs}
    ${_sofa_generated_include_dirs}
)
target_link_libraries(MCRBeamFeasibleHook PRIVATE Sofa.Core Sofa.SimulationCore)
target_link_options(MCRBeamFeasibleHook PRIVATE
    "-Wl,-rpath-link,${SOFA_PREFIX}/lib"
)
set_target_properties(MCRBeamFeasibleHook PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    BUILD_RPATH "${SOFA_PREFIX}/lib"
)
