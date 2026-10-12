cmake_minimum_required(VERSION 3.12)
project(MCRContactReferenceConsistency LANGUAGES CXX)

set(SOFA_PREFIX "$ENV{SOFA_ROOT}" CACHE PATH "SOFA 21.12 build/install prefix")
set(SOFA_SOURCE_ROOT "$ENV{SOFA_SRC}" CACHE PATH "SOFA source tree")

# The mCR SOFA 21.12 checkout uses SOFA_SRC=<...>/sofa/src with the
# actual C++ sources in SOFA_SRC/sofa/.  Match the already-working
# mcr_sim/native/beam_unilateral/CMakeLists.txt layout. Also accept callers
# that directly provide the nested "sofa" directory.
set(_mcr_source_candidate "${SOFA_SOURCE_ROOT}")
if(NOT EXISTS "${_mcr_source_candidate}/SofaKernel/modules/SofaCore/src/sofa/core/ConstraintParams.h"
   AND EXISTS "${_mcr_source_candidate}/sofa/SofaKernel/modules/SofaCore/src/sofa/core/ConstraintParams.h")
    set(_mcr_source_candidate "${_mcr_source_candidate}/sofa")
endif()
# FORCE is deliberate: V3's first release cached the wrong SOFA_SRC root
# during failed CMake configuration; a subsequent build must repair it.
set(SOFA_SOURCE_ROOT "${_mcr_source_candidate}" CACHE PATH "Resolved SOFA source tree" FORCE)
message(STATUS "Resolved SOFA source root: ${SOFA_SOURCE_ROOT}")

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

add_library(MCRContactReferenceConsistency SHARED
    TestOnlyRelinearizingGenericConstraintSolver.cpp
    FastBeamSDF.cpp)
target_compile_features(MCRContactReferenceConsistency PRIVATE cxx_std_17)
target_include_directories(MCRContactReferenceConsistency PRIVATE
    ${_sofa_module_include_dirs} ${_sofa_generated_include_dirs})
target_link_libraries(MCRContactReferenceConsistency PRIVATE
    Sofa.Core Sofa.SimulationCore Sofa.Constraint)
target_link_options(MCRContactReferenceConsistency PRIVATE
    "-Wl,-rpath-link,${SOFA_PREFIX}/lib")
set_target_properties(MCRContactReferenceConsistency PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    BUILD_RPATH "${SOFA_PREFIX}/lib")

target_link_libraries(MCRContactReferenceConsistency PRIVATE "${SOFA_PREFIX}/lib/libSofaMeshCollision.so" "${SOFA_PREFIX}/lib/libSofaBaseMechanics.so")
find_path(EIGEN3_INCLUDE_DIR Eigen/Core HINTS "$ENV{EIGEN3_INCLUDE_DIR}" "${SOFA_PREFIX}/include/eigen3" PATH_SUFFIXES eigen3)
if(NOT EIGEN3_INCLUDE_DIR)
    message(FATAL_ERROR "Eigen3 headers required; set EIGEN3_INCLUDE_DIR")
endif()
target_include_directories(MCRContactReferenceConsistency PRIVATE "${EIGEN3_INCLUDE_DIR}")
