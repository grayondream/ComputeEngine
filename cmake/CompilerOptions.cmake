function(set_ce_compiler_options target)
    target_compile_options(${target} PRIVATE
        -Wall -Wextra -Wpedantic
        -Wno-unused-parameter
        -Wno-missing-field-initializers
    )

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE
            -fvisibility=hidden
            -ffunction-sections
            -fdata-sections
        )
    endif()

    if(CE_BUILD_AVX)
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag("-mavx2 -mfma" CE_HAS_AVX2)
        if(CE_HAS_AVX2)
            target_compile_options(${target} PRIVATE
                -mavx2 -mfma -mbmi -mbmi2 -mpopcnt
            )
            target_compile_definitions(${target} PRIVATE CE_HAS_AVX2=1)
        endif()

        check_cxx_compiler_flag("-mavx512f -mfma -mbmi2" CE_HAS_AVX512)
        if(CE_HAS_AVX512)
            target_compile_options(${target} PRIVATE
                -mavx512f -mavx512bw -mavx512vl -mfma -mbmi2
            )
            target_compile_definitions(${target} PRIVATE CE_HAS_AVX512=1)
        endif()
    endif()

    if(CE_BUILD_NEON)
        target_compile_definitions(${target} PRIVATE CE_HAS_NEON=1)
    endif()

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_definitions(${target} PRIVATE CE_DEBUG=1)
    endif()
endfunction()
