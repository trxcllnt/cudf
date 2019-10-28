/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_device_view.hpp>

#include <cudf/wrappers/timestamps.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

namespace cudf {
  namespace detail {

#define IS_TIMESTAMP(TYPE)                       \
std::is_same<TYPE, cudf::timestamp_D>::value  || \
std::is_same<TYPE, cudf::timestamp_s>::value  || \
std::is_same<TYPE, cudf::timestamp_ms>::value || \
std::is_same<TYPE, cudf::timestamp_us>::value || \
std::is_same<TYPE, cudf::timestamp_ns>::value


    template <typename FromType>
    struct cast_timestamp {

      static_assert(is_timestamp<FromType>(),"");

      column_device_view input;
      mutable_column_view output;

      template <typename ToType>
      typename std::enable_if_t<IS_TIMESTAMP(ToType), void>
      operator ()(cudaStream_t stream) {
        thrust::tabulate(rmm::exec_policy(stream)->on(stream),
                         output.begin<int16_t>(), output.end<int16_t>(),
                         [=] __device__ (size_type i) {
                           return simt::std::chrono::time_point_cast<ToType>(input.element<FromType>(i));
                         });
      }

      template <typename ToType>
      typename std::enable_if_t<
        std::is_arithmetic<ToType>::value,
      void>
      operator ()(cudaStream_t stream) {
        thrust::tabulate(rmm::exec_policy(stream)->on(stream),
                         output.begin<int16_t>(), output.end<int16_t>(),
                         [=] __device__ (size_type i) {
                           return static_cast<ToType>(input.element<FromType>(i));
                         });
      }
    }

    struct unary_cast_from_launcher {

      column_device_view input;
      mutable_column_view output;

      template <typename FromType>
      typename std::enable_if_t<IS_TIMESTAMP(FromType), void>
      operator ()(cudaStream_t stream) {}
    };

#undef IS_TIMESTAMP
  }
  
  std::unique_ptr<column> cast(column_view const& input, data_type out_type) {
  }  
}