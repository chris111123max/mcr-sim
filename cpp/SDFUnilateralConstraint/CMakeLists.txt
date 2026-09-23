cmake_minimum_required(VERSION 3.12)
project(SDFUnilateralConstraint VERSION 0.1.0 LANGUAGES CXX)

find_package(SofaConstraint REQUIRED)

set(SRC_DIR src/SDFUnilateralConstraint)
set(HEADER_FILES
    ${SRC_DIR}/config.h.in
    ${SRC_DIR}/SDFUnilateralConstraint.h
)
set(SOURCE_FILES
    ${SRC_DIR}/SDFUnilateralConstraint.cpp
    ${SRC_DIR}/initPlugin.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaConstraint)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_14)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "plugins"
)
