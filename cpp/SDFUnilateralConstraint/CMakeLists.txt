cmake_minimum_required(VERSION 3.12)
project(SDFUnilateralConstraint VERSION 0.1.0 LANGUAGES CXX)

option(
    MCR_SOFA_BUILD_TREE_FALLBACK
    "Build directly against an existing SOFA build/source tree without requiring an installed CMake package."
    OFF
)

set(SRC_DIR src/SDFUnilateralConstraint)
set(HEADER_FILES
    ${SRC_DIR}/SDFUnilateralConstraint.h
)
set(SOURCE_FILES
    ${SRC_DIR}/SDFUnilateralConstraint.cpp
    ${SRC_DIR}/initPlugin.cpp
)

set(GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${GENERATED_INCLUDE_DIR}/SDFUnilateralConstraint")
configure_file(
    "${SRC_DIR}/config.h.in"
    "${GENERATED_INCLUDE_DIR}/SDFUnilateralConstraint/config.h"
    @ONLY
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
target_compile_definitions(${PROJECT_NAME} PRIVATE SOFA_BUILD_SDFUNILATERALCONSTRAINT)
target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${GENERATED_INCLUDE_DIR}"
)

if(MCR_SOFA_BUILD_TREE_FALLBACK)
    if(NOT MCR_SOFA_INCLUDE_DIRS)
        message(FATAL_ERROR "MCR_SOFA_INCLUDE_DIRS is required in build-tree fallback mode")
    endif()
    if(NOT MCR_SOFA_CONSTRAINT_LIBRARY)
        message(FATAL_ERROR "MCR_SOFA_CONSTRAINT_LIBRARY is required in build-tree fallback mode")
    endif()

    message(STATUS "SDFUnilateralConstraint: using read-only SOFA build-tree fallback")
    message(STATUS "SDFUnilateralConstraint: SofaConstraint=${MCR_SOFA_CONSTRAINT_LIBRARY}")

    target_include_directories(${PROJECT_NAME} PRIVATE ${MCR_SOFA_INCLUDE_DIRS})
    if(Boost_INCLUDE_DIR)
        target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE "${Boost_INCLUDE_DIR}")
    endif()

    # Link only to the already-built SofaConstraint DSO.  The running SOFA
    # process supplies its normal dependency graph; no SOFA files are installed
    # or modified by this project.
    target_link_libraries(${PROJECT_NAME} PRIVATE "${MCR_SOFA_CONSTRAINT_LIBRARY}")

    if(MCR_SOFA_RUNTIME_LIBRARY_DIRS)
        set_target_properties(
            ${PROJECT_NAME}
            PROPERTIES
                BUILD_RPATH "${MCR_SOFA_RUNTIME_LIBRARY_DIRS}"
        )
    endif()
else()
    find_package(SofaConstraint REQUIRED)
    target_link_libraries(${PROJECT_NAME} PUBLIC SofaConstraint)

    sofa_create_package_with_targets(
        PACKAGE_NAME ${PROJECT_NAME}
        PACKAGE_VERSION ${PROJECT_VERSION}
        TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
        INCLUDE_SOURCE_DIR "src"
        INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
        RELOCATABLE "plugins"
    )
endif()
