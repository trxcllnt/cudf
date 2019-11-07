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

#include <cudf/unary.hpp>
#include <cudf/null_mask.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_device_view.cuh>

#include <cudf/utilities/traits.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

namespace cudf {
namespace detail {

#define IS_TIMESTAMP(TYPE)                     ( \
std::is_same<TYPE, cudf::timestamp_D>::value  || \
std::is_same<TYPE, cudf::timestamp_s>::value  || \
std::is_same<TYPE, cudf::timestamp_ms>::value || \
std::is_same<TYPE, cudf::timestamp_us>::value || \
std::is_same<TYPE, cudf::timestamp_ns>::value)

  template <typename FromType>
  struct unary_cast {

    column_device_view input;
    mutable_column_view output;

    unary_cast(
      column_device_view inp,
      mutable_column_view out
    ) : input(inp), output(out) {}

    template <typename ToType>
    typename std::enable_if_t<std::is_same<cudf::string_view, ToType>::value ||
                              std::is_same<cudf::string_view, FromType>::value, void>
    operator ()(cudaStream_t stream) {
      CUDF_FAIL("Column type must be numeric");
    }

    template <typename ToType>
    typename std::enable_if_t<IS_TIMESTAMP(FromType) && is_numeric<ToType>(), void>
    operator ()(cudaStream_t stream) {
      thrust::tabulate(rmm::exec_policy(stream)->on(stream),
                        output.begin<ToType>(), output.end<ToType>(),
                        [=] __device__ (size_type i) {
                          return static_cast<ToType>(input.element<FromType>(i).time_since_epoch().count());
                        });
    }

    template <typename ToType>
    typename std::enable_if_t<IS_TIMESTAMP(FromType) && IS_TIMESTAMP(ToType), void>
    operator ()(cudaStream_t stream) {
      thrust::tabulate(rmm::exec_policy(stream)->on(stream),
                        output.begin<ToType>(), output.end<ToType>(),
                        [=] __device__ (size_type i) {
                          return simt::std::chrono::time_point_cast<typename ToType::duration>(input.element<FromType>(i));
                        });
    }

    template <typename ToType>
    typename std::enable_if_t<
      !(IS_TIMESTAMP(FromType) && IS_TIMESTAMP(ToType)) &&
      !(IS_TIMESTAMP(FromType) && is_numeric<ToType>()) &&
      !(std::is_same<cudf::string_view, ToType>::value || std::is_same<cudf::string_view, FromType>::value), void>
    operator ()(cudaStream_t stream) {
      thrust::tabulate(rmm::exec_policy(stream)->on(stream),
                        output.begin<ToType>(), output.end<ToType>(),
                        [=] __device__ (size_type i) {
                          return static_cast<ToType>(input.element<FromType>(i));
                        });
    }
  };

  struct unary_cast_launcher {

    column_device_view input;
    mutable_column_view output;

    unary_cast_launcher(
      column_device_view inp,
      mutable_column_view out
    ) : input(inp), output(out) {}

    template <typename FromType>
    typename std::enable_if_t<IS_TIMESTAMP(FromType) || is_numeric<FromType>(), void>
    operator ()(cudaStream_t stream) {
      experimental::type_dispatcher(input.type(), unary_cast<FromType>{input, output}, stream);
    }

    template <typename FromType>
    typename std::enable_if_t<!IS_TIMESTAMP(FromType) && !is_numeric<FromType>(), void>
    operator ()(cudaStream_t stream) {
      CUDF_FAIL("Column type must be numeric");
    }
  };

#undef IS_TIMESTAMP
}

std::unique_ptr<column> cast(column_view const& input, data_type type, cudaStream_t stream, rmm::mr::device_memory_resource* mr) {
  CUDF_EXPECTS(is_fixed_width(type), "Unary cast-to type must be fixed-width.");
  auto output = std::make_unique<cudf::column>(type, input.size(),
                                               rmm::device_buffer{input.size() * cudf::size_of(type), stream, mr},
                                               copy_bitmask(input, stream, mr), input.null_count());
  experimental::type_dispatcher(type, detail::unary_cast_launcher{*column_device_view::create(input), *output}, stream);
  return output;
}
}
