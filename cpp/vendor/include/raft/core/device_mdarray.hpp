/*
 * cuOpt vendored RAFT shim — minimal owning device matrix (make_device_matrix).
 * Backed by the rmm shim's device_uvector (cudaMemPool, no cuda::mr).
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <cstddef>

namespace raft {

/** @brief Owning row-major device matrix exposing data_handle() and a view(). */
template <typename ElementType, typename IndexType = int>
class device_matrix {
 public:
  using element_type = ElementType;
  using extents_type = extents<IndexType, dynamic_extent, dynamic_extent>;
  using view_type    = device_mdspan<ElementType, extents_type>;

  device_matrix(raft::resources const& handle, IndexType n_rows, IndexType n_cols)
    : data_{static_cast<std::size_t>(n_rows) * static_cast<std::size_t>(n_cols),
            handle.get_stream()},
      n_rows_{n_rows},
      n_cols_{n_cols}
  {
  }

  [[nodiscard]] ElementType* data_handle() noexcept { return data_.data(); }
  [[nodiscard]] ElementType const* data_handle() const noexcept { return data_.data(); }

  [[nodiscard]] view_type view() noexcept
  {
    return view_type{data_.data(), extents_type{n_rows_, n_cols_}};
  }

  [[nodiscard]] IndexType extent(int dim) const noexcept { return dim == 0 ? n_rows_ : n_cols_; }

 private:
  rmm::device_uvector<ElementType> data_;
  IndexType n_rows_;
  IndexType n_cols_;
};

/** @brief Allocate an n_rows x n_cols owning device matrix. */
template <typename ElementType, typename IndexType = int>
auto make_device_matrix(raft::resources const& handle, IndexType n_rows, IndexType n_cols)
{
  return device_matrix<ElementType, IndexType>{handle, n_rows, n_cols};
}

}  // namespace raft
