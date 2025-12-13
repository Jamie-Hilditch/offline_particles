cmake_minimum_required(VERSION 3.21)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C VERSION ${SKBUILD_PROJECT_VERSION})

# required packages
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_program(CYTHON_EXECUTABLE cython REQUIRED)

# ------------------------------------------------------------
# Define inputs
# ------------------------------------------------------------

# Declare kernel modules
include(cmake/kernels.cmake)

# Cython compilation arguments
set(CYTHON_ARGS -3 -X boundscheck=False)

# ------------------------------------------------------------
# end inputs
# ------------------------------------------------------------

# ------------------------------------------------------------
# Check required variables
# -----------------------------------------------------------

if(NOT DEFINED KERNEL_MODULES OR NOT DEFINED KERNEL_PYX_FILES)
    message(FATAL_ERROR
        "KERNEL_MODULES and KERNEL_PYX_FILES must be defined"
    )
endif()

list(LENGTH KERNEL_MODULES _n_mods)
list(LENGTH KERNEL_PYX_FILES _n_pyx)

if(NOT _n_mods EQUAL _n_pyx)
    message(FATAL_ERROR
        "KERNEL_MODULES and KERNEL_PYX_FILES must have same length"
    )
endif()

# ------------------------------------------------------------
# Build each kernel module
# ------------------------------------------------------------

foreach(module_name pyx_file IN ZIP_LISTS KERNEL_MODULES KERNEL_PYX_FILES)

    get_filename_component(mod_basename ${pyx_file} NAME_WE)
    set(c_file "${CMAKE_CURRENT_BINARY_DIR}/${mod_basename}.c")

    # Generate C from Cython
    add_custom_command(
        OUTPUT ${c_file}
        COMMAND ${CYTHON_EXECUTABLE}
                ${CYTHON_ARGS}
                -o ${c_file}
                ${pyx_file}
        DEPENDS ${pyx_file}
        COMMENT "Cythonizing ${module_name}"
        VERBATIM
    )

    # Build Python extension module
    python_add_library(${module_name}
        MODULE
        WITH_SOABI
        ${c_file}
    )

endforeach()