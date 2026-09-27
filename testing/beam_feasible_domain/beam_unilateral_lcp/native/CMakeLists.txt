cmake_minimum_required(VERSION 3.12)
project(MCRBeamLinearizedUnilateral VERSION 0.1 LANGUAGES CXX)

# Test-only SOFA 21.12 plugin. Reuse the same direct-link strategy as the
# validated native precommit hook because the server package config is incomplete.
set(SOFA_PREFIX "$ENV{SOFA_ROOT}" CACHE PATH "SOFA 21.12 build/install prefix")
set(SOFA_SOURCE_ROOT "$ENV{SOFA_SRC}/sofa" CACHE PATH "SOFA source tree")

foreach(_required
    "${SOFA_PREFIX}/lib/libSofa.Core.so"
    "${SOFA_SOURCE_ROOT}/SofaKernel/modules/SofaCore/src/sofa/core/behavior/Constraint.h"
    "${SOFA_SOURCE_ROOT}/SofaKernel/modules/SofaCore/src/sofa/defaulttype/RigidTypes.h"
    "${SOFA_PREFIX}/include/Sofa.Config/sofa/config.h")
    if(NOT EXISTS "${_required}")
        message(FATAL_ERROR "Required SOFA 21.12 development file is unavailable: ${_required}")
    endif()
endforeach()

add_library(Sofa.Core SHARED IMPORTED GLOBAL)
set_target_properties(Sofa.Core PROPERTIES
    IMPORTED_LOCATION "${SOFA_PREFIX}/lib/libSofa.Core.so"
)

file(GLOB _sofa_module_include_dirs LIST_DIRECTORIES true
    "${SOFA_SOURCE_ROOT}/SofaKernel/modules/*/src"
)
file(GLOB _sofa_generated_include_dirs LIST_DIRECTORIES true
    "${SOFA_PREFIX}/include/*"
)

add_library(MCRBeamLinearizedUnilateral SHARED
    BeamLinearizedUnilateralConstraint.cpp
)
target_compile_features(MCRBeamLinearizedUnilateral PRIVATE cxx_std_17)
target_include_directories(MCRBeamLinearizedUnilateral PRIVATE
    ${_sofa_module_include_dirs}
    ${_sofa_generated_include_dirs}
)
target_link_libraries(MCRBeamLinearizedUnilateral PRIVATE Sofa.Core)
target_link_options(MCRBeamLinearizedUnilateral PRIVATE
    "-Wl,-rpath-link,${SOFA_PREFIX}/lib"
)
set_target_properties(MCRBeamLinearizedUnilateral PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    BUILD_RPATH "${SOFA_PREFIX}/lib"
)
