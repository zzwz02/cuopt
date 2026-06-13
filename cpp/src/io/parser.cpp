/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/linear_programming/io/parser.hpp>

#include <mps_parser_internal.hpp>

namespace cuopt::linear_programming::io {

template <typename i_t, typename f_t>
mps_data_model_t<i_t, f_t> read_mps(const std::string& mps_file, bool fixed_mps_format)
{
  {
    mps_data_model_t<i_t, f_t> problem;
    try {
      mps_parser_t<i_t, f_t> parser(problem, mps_file, fixed_mps_format);
      return problem;
    } catch (const std::exception&) {
      // Free-format parsing rejects strictly column-aligned files (e.g. row
      // names with embedded blanks, QFORPLAN). Retry in fixed format before
      // giving up; rethrow the original failure if that also fails.
      if (fixed_mps_format) { throw; }
    }
  }
  try {
    mps_data_model_t<i_t, f_t> problem;
    mps_parser_t<i_t, f_t> parser(problem, mps_file, /*fixed_mps_format=*/true);
    return problem;
  } catch (const std::exception&) {
    // Re-run free format so the caller sees the original error.
    mps_data_model_t<i_t, f_t> problem;
    mps_parser_t<i_t, f_t> parser(problem, mps_file, false);
    return problem;
  }
}

template <typename i_t, typename f_t>
mps_data_model_t<i_t, f_t> read_mps_from_string(std::string_view mps_contents,
                                                bool fixed_mps_format)
{
  mps_data_model_t<i_t, f_t> problem;
  mps_parser_t<i_t, f_t> parser(problem, mps_contents, fixed_mps_format);
  return problem;
}

template mps_data_model_t<int, float> read_mps(const std::string& mps_file, bool fixed_mps_format);
template mps_data_model_t<int, double> read_mps(const std::string& mps_file, bool fixed_mps_format);
template mps_data_model_t<int, float> read_mps_from_string(std::string_view mps_contents,
                                                           bool fixed_mps_format);
template mps_data_model_t<int, double> read_mps_from_string(std::string_view mps_contents,
                                                            bool fixed_mps_format);

}  // namespace cuopt::linear_programming::io
