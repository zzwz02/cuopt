# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

# Fetch googletest (static) via plain FetchContent; provides the GTest::gtest,
# GTest::gtest_main, GTest::gmock and GTest::gmock_main targets.
function(find_and_configure_gtest)
    include(FetchContent)
    set(BUILD_SHARED_LIBS OFF)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.16.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(googletest)
endfunction()

find_and_configure_gtest()
