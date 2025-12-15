cmake_minimum_required(VERSION 3.21)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C VERSION ${SKBUILD_PROJECT_VERSION})

# required packages
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_program(CYTHON_EXECUTABLE cython REQUIRED)
find_package(OpenMP REQUIRED)

# ------------------------------------------------------------
# Define inputs
# ------------------------------------------------------------

# Declare extension modules
set(EXTENSION_MODULES
    kernels._core
    kernels.status
    kernels.roms.rk2_w_advection
)

# Cython compilation arguments
set(CYTHON_ARGS -3 -X boundscheck=False)

# ------------------------------------------------------------
# end inputs
# ------------------------------------------------------------


# ------------------------------------------------------------
# Build each kernel module
# ------------------------------------------------------------

foreach(module IN LISTS EXTENSION_MODULES)
    # extract module name
    string(REGEX REPLACE ".*\\.([^.]+)$" "\\1" module_name ${module})

    # extract module directory relative to package root
    string(REGEX REPLACE "\\.[^.]+$" "" parent ${module})
    string(REPLACE "." "/" module_dir ${parent})

    # compilation paths
    set(c_dir "${CMAKE_CURRENT_BINARY_DIR}/${module_dir}")
    file(MAKE_DIRECTORY ${c_dir})
    set(c_file "${c_dir}/${module_name}.c")
    set(pyx_file "${CMAKE_CURRENT_SOURCE_DIR}/src/offline_particles/${module_dir}/${module_name}.pyx")

    # Generate C from Cython
    add_custom_command(
        OUTPUT ${c_file}
        COMMAND ${CYTHON_EXECUTABLE}
                ${CYTHON_ARGS}
                -o ${c_file}
                ${pyx_file}
        DEPENDS ${pyx_file}
        COMMENT "Cythonizing ${module}"
        VERBATIM
    )

    # unique CMake target name
    string(REPLACE "." "__" target_name ${module})

    python_add_library(${target_name} MODULE ${c_file})
    set_target_properties(${target_name} PROPERTIES OUTPUT_NAME ${module_name}${Python_SOABI_SUFFIX})

    if(OpenMP_CXX_FOUND)
        target_compile_options(${target_name} PRIVATE ${OpenMP_CXX_FLAGS})
        target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_CXX)
    endif()

    # install
    install(
        TARGETS ${target_name}
        LIBRARY DESTINATION "offline_particles/${module_dir}"
    )

endforeach()