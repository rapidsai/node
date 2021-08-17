// Copyright (c) 2021, NVIDIA CORPORATION.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <raft/mr/device/buffer.hpp>

namespace raft {
namespace sparse {

/** @brief A Container object for sparse coordinate. There are two motivations
 * behind using a container for COO arrays.
 *
 * The first motivation is that it simplifies code, rather than always having
 * to pass three arrays as function arguments.
 *
 * The second is more subtle, but much more important. The size
 * of the resulting COO from a sparse operation is often not known ahead of time,
 * since it depends on the contents of the underlying graph. The COO object can
 * allocate the underlying arrays lazily so that the object can be created by the
 * user and passed as an output argument in a sparse primitive. The sparse primitive
 * would have the responsibility for allocating and populating the output arrays,
 * while the original caller still maintains ownership of the underlying memory.
 *
 * @tparam T: the type of the value array.
 * @tparam Index_Type: the type of index array
 *
 */
template <typename T, typename Index_Type = int>
class COO {
 protected:
  raft::mr::device::buffer<Index_Type> rows_arr;
  raft::mr::device::buffer<Index_Type> cols_arr;
  raft::mr::device::buffer<T> vals_arr;

 public:
  Index_Type nnz;
  Index_Type n_rows;
  Index_Type n_cols;

  /**
   * @param d_alloc: the device allocator to use for the underlying buffers
   * @param stream: CUDA stream to use
   */
  COO(std::shared_ptr<raft::mr::device::allocator> d_alloc, cudaStream_t stream)
    : rows_arr(d_alloc, stream, 0),
      cols_arr(d_alloc, stream, 0),
      vals_arr(d_alloc, stream, 0),
      nnz(0),
      n_rows(0),
      n_cols(0) {}

  /*
   * @brief Returns the rows array
   */
  Index_Type *rows() { return this->rows_arr.data(); }

  /**
   * @brief Returns the cols array
   */
  Index_Type *cols() { return this->cols_arr.data(); }

  /**
   * @brief Returns the vals array
   */
  T *vals() { return this->vals_arr.data(); }
};

};  // namespace sparse
};  // namespace raft
