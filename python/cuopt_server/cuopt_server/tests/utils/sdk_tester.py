# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys

from cuopt import routing

from cuopt_server.utils.utils import build_routing_datamodel_from_json

data_file = sys.argv[1]

data_model, solver_settings = build_routing_datamodel_from_json(data_file)

solution = routing.Solve(data_model, solver_settings)
