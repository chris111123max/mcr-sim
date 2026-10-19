cmake_minimum_required(VERSION 3.12)
project(MCRNonlinearRelinearizationTest LANGUAGES CXX)

set(SOFA_PREFIX "$ENV{SOFA_ROOT}" CACHE PATH "SOFA 21.12 build/install prefix")
set(SOFA_SOURCE_ROOT "$ENV{SOFA_SRC}" CACHE PATH "SOFA source tree")

foreach(_required
    "${SOFA_PREFIX}/lib/libSofa.Core.so"
    "${SOFA_PREFIX}/lib/libSofa.SimulationCore.so"
    "${SOFA_PREFIX}/lib/libSofaConstraint.so"
    "${SOFA_PREFIX}/include/Sofa.Config/sofa/config.h"
    "${SOFA_SOURCE_ROOT}/SofaKernel/modules/SofaCore/src/sofa/core/ConstraintParams.h"
    "${SOFA_SOURCE_ROOT}/modules/SofaConstraint/src/SofaConstraint/GenericConstraintSolver.h"
    "${SOFA_SOURCE_ROOT}/SofaKernel/modules/SofaSimulationCore/src/sofa/simulation/mechanicalvisitor/MechanicalPropagateOnlyPositionAndVelocityVisitor.h")
    if(NOT EXISTS "${_required}")
        message(FATAL_ERROR "Required SOFA development file unavailable: ${_required}")
    endif()
endforeach()

add_library(Sofa.Core SHARED IMPORTED GLOBAL)
set_target_properties(Sofa.Core PROPERTIES IMPORTED_LOCATION "${SOFA_PREFIX}/lib/libSofa.Core.so")
add_library(Sofa.SimulationCore SHARED IMPORTED GLOBAL)
set_target_properties(Sofa.SimulationCore PROPERTIES IMPORTED_LOCATION "${SOFA_PREFIX}/lib/libSofa.SimulationCore.so")
add_library(Sofa.Constraint SHARED IMPORTED GLOBAL)
set_target_properties(Sofa.Constraint PROPERTIES IMPORTED_LOCATION "${SOFA_PREFIX}/lib/libSofaConstraint.so")

file(GLOB _sofa_module_include_dirs LIST_DIRECTORIES true
    "${SOFA_SOURCE_ROOT}/SofaKernel/modules/*/src"
    "${SOFA_SOURCE_ROOT}/modules/*/src")
file(GLOB _sofa_generated_include_dirs LIST_DIRECTORIES true "${SOFA_PREFIX}/include/*")

add_library(MCRNonlinearRelinearizationTest SHARED
    TestOnlyRelinearizingGenericConstraintSolver.cpp)
target_compile_features(MCRNonlinearRelinearizationTest PRIVATE cxx_std_17)
target_include_directories(MCRNonlinearRelinearizationTest PRIVATE
    ${_sofa_module_include_dirs} ${_sofa_generated_include_dirs})
target_link_libraries(MCRNonlinearRelinearizationTest PRIVATE
    Sofa.Core Sofa.SimulationCore Sofa.Constraint)
target_link_options(MCRNonlinearRelinearizationTest PRIVATE
    "-Wl,-rpath-link,${SOFA_PREFIX}/lib")
set_target_properties(MCRNonlinearRelinearizationTest PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    BUILD_RPATH "${SOFA_PREFIX}/lib")
