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

#include <cudf/null_mask.hpp>
#include <cudf/unary.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>

#include <cudf/utilities/traits.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/execution_policy.h>
#include <thrust/tabulate.h>

namespace cudf {
namespace experimental {
namespace detail {

template <typename T, typename R>
struct cast_numeric {
  static_assert(is_numeric<T>() && is_numeric<R>(), "");
  column_device_view column;
  cast_numeric(column_device_view inp) : column(inp) {}
  CUDA_DEVICE_CALLABLE R operator()(size_type const i) {
    return static_cast<R>(column.element<T>(i));
  }
};

template <typename T, typename R>
struct cast_numeric_to_timestamp {
  static_assert(is_numeric<T>() && is_timestamp<R>(), "");
  column_device_view column;
  cast_numeric_to_timestamp(column_device_view inp) : column(inp) {}
  CUDA_DEVICE_CALLABLE R operator()(size_type const i) {
    return static_cast<R>(static_cast<typename R::rep>(column.element<T>(i)));
  }
};

template <typename T, typename R>
struct cast_timestamp_to_numeric {
  static_assert(is_timestamp<T>() && is_numeric<R>(), "");
  column_device_view column;
  cast_timestamp_to_numeric(column_device_view inp) : column(inp) {}
  CUDA_DEVICE_CALLABLE R operator()(size_type const i) {
    return static_cast<R>(column.element<T>(i).time_since_epoch().count());
  }
};

template <typename T, typename R>
struct cast_timestamp_to_timestamp {
  static_assert(is_timestamp<T>() && is_timestamp<R>(), "");
  column_device_view column;
  cast_timestamp_to_timestamp(column_device_view inp) : column(inp) {}
  CUDA_DEVICE_CALLABLE R operator()(size_type const i) {
    using D = typename R::duration;
    using namespace simt::std::chrono;
    return time_point_cast<D>(column.element<T>(i)).time_since_epoch().count();
  }
};

template <typename T>
struct dispatch_numeric_cast {
  static_assert(is_numeric<T>(), "");

  column_device_view input;
  mutable_column_view output;

  dispatch_numeric_cast(column_device_view inp, mutable_column_view out)
      : input(inp), output(out) {}

  template <typename R>
  typename std::enable_if_t<is_numeric<R>(), void> operator()(
      cudaStream_t stream) {
    thrust::tabulate(rmm::exec_policy(stream)->on(stream), output.begin<R>(),
                     output.end<R>(), cast_numeric<T, R>{input});
  }

  template <typename R>
  typename std::enable_if_t<is_timestamp<R>(), void> operator()(
      cudaStream_t stream) {
    thrust::tabulate(rmm::exec_policy(stream)->on(stream), output.begin<R>(),
                     output.end<R>(), cast_numeric_to_timestamp<T, R>{input});
  }

  template <typename R>
  typename std::enable_if_t<!is_numeric<R>() && !is_timestamp<R>(), void>
  operator()(cudaStream_t stream) {
    CUDF_FAIL("Column type must be numeric");
  }
};

template <typename T>
struct dispatch_timestamp_cast {
  static_assert(is_timestamp<T>(), "");

  column_device_view input;
  mutable_column_view output;

  dispatch_timestamp_cast(column_device_view inp, mutable_column_view out)
      : input(inp), output(out) {}

  template <typename R>
  typename std::enable_if_t<is_numeric<R>(), void> operator()(
      cudaStream_t stream) {
    thrust::tabulate(rmm::exec_policy(stream)->on(stream), output.begin<R>(),
                     output.end<R>(), cast_timestamp_to_numeric<T, R>{input});
  }

  template <typename R>
  typename std::enable_if_t<is_timestamp<R>(), void> operator()(
      cudaStream_t stream) {
    thrust::tabulate(rmm::exec_policy(stream)->on(stream), output.begin<R>(),
                     output.end<R>(), cast_timestamp_to_timestamp<T, R>{input});
  }

  template <typename R>
  typename std::enable_if_t<!is_numeric<R>() && !is_timestamp<R>(), void>
  operator()(cudaStream_t stream) {
    CUDF_FAIL("Column type must be numeric");
  }
};

struct dispatch_unary_cast {
  column_device_view input;
  mutable_column_view output;

  dispatch_unary_cast(column_device_view inp, mutable_column_view out)
      : input(inp), output(out) {}

  template <typename T>
  typename std::enable_if_t<is_numeric<T>(), void> operator()(
      cudaStream_t stream) {
    experimental::type_dispatcher(
        output.type(), dispatch_numeric_cast<T>{input, output}, stream);
  }

  template <typename T>
  typename std::enable_if_t<is_timestamp<T>(), void> operator()(
      cudaStream_t stream) {
    experimental::type_dispatcher(
        output.type(), dispatch_timestamp_cast<T>{input, output}, stream);
  }

  template <typename T>
  typename std::enable_if_t<!is_timestamp<T>() && !is_numeric<T>(), void>
  operator()(cudaStream_t stream) {
    CUDF_FAIL("Column type must be numeric");
  }
};
}  // namespace detail

std::unique_ptr<column> cast(column_view const& input,
                             data_type type,
                             cudaStream_t stream,
                             rmm::mr::device_memory_resource* mr) {
  CUDF_EXPECTS(is_fixed_width(type), "Unary cast-to type must be fixed-width.");

  auto size = input.size();
  auto null_mask = copy_bitmask(input, stream, mr);
  auto output = std::make_unique<column>(
      type, size, rmm::device_buffer{size * cudf::size_of(type), stream, mr},
      null_mask, input.null_count(), std::vector<std::unique_ptr<column>>{});

  auto launch_cast =
      detail::dispatch_unary_cast{*column_device_view::create(input),
                                  static_cast<mutable_column_view>(*output)};

  experimental::type_dispatcher(input.type(), launch_cast, stream);

  return output;
}

}  // namespace experimental
}  // namespace cudf
