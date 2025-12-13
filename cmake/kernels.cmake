# ============================================================
# Kernel module declarations
# ============================================================

# --------------------------------------------
# Declare kernel modules
# -------------------------------------------

# Kernel module names relative to offline_particles.kernels
set(KERNEL_NAMES
    status
    roms.rk2_w_advection
)

# ----------------------------
# Generate file lists
# ----------------------------

# Base directory for kernel sources
set(KERNELS_DIR
    ${CMAKE_CURRENT_SOURCE_DIR}/src/offline_particles/kernels
)

# Derived lists (exported to parent scope)
set(KERNEL_MODULES "")
set(KERNEL_PYX_FILES "")

foreach(name IN LISTS KERNEL_NAMES)

    # Full Python module name
    list(APPEND KERNEL_MODULES
        offline_particles.kernels.${name}
    )

    # Convert dotted path → filesystem path
    string(REPLACE "." "/" rel_path ${name})

    set(pyx_file ${KERNELS_DIR}/${rel_path}.pyx)

    if(NOT EXISTS ${pyx_file})
        message(FATAL_ERROR
            "Kernel '${name}' not found at ${pyx_file}"
        )
    endif()

    list(APPEND KERNEL_PYX_FILES ${pyx_file})

endforeach()

# # Make variables visible to includer
# set(KERNEL_MODULES ${KERNEL_MODULES} PARENT_SCOPE)
# set(KERNEL_PYX_FILES ${KERNEL_PYX_FILES} PARENT_SCOPE)
