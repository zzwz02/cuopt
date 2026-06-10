#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Install SuiteSparse development files (AMD ordering used by the barrier
# solver's built-in sparse LDL^T factorization).
if command -v dnf &> /dev/null; then
    dnf -y install suitesparse-devel
elif command -v apt-get &> /dev/null; then
    apt-get update
    apt-get install -y libsuitesparse-dev
else
    echo "Neither dnf nor apt-get found. Cannot install SuiteSparse dependencies."
    exit 1
fi
