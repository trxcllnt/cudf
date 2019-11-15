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

#include <rolling/rolling_detail.hpp>
#include <cudf/rolling.hpp>
#include <cudf/types.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/utilities/nvtx_utils.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/copying.hpp>

#include <memory>

namespace cudf {
namespace experimental {

namespace detail {

namespace { // anonymous

/**
 * @brief Computes the rolling window function
 *
 * @tparam ColumnType  Datatype of values pointed to by the pointers
 * @tparam agg_op  A functor that defines the aggregation operation
 * @tparam average Perform average across all valid elements in the window
 * @param nrows[in]  Number of rows in input table
 * @param out_col[out]  Pointers to pre-allocated output column's data
 * @param out_cols_valid[out]  Pointers to the pre-allocated validity mask of
 * 		  the output column
 * @param in_col[in]  Pointers to input column's data
 * @param in_cols_valid[in]  Pointers to the validity mask of the input column
 * @param window[in]  The static rolling window size, accumulates from
 *                in_col[i-window+1] to in_col[i] inclusive
 * @param min_periods[in]  Minimum number of observations in window required to
 *                have a value, otherwise 0 is stored in the valid bit mask
 * @param forward_window[in]  The static rolling window size in the forward
 *                direction, accumulates from in_col[i] to
 *                in_col[i+forward_window] inclusive
 * @param[in] window_col The window size values, window_col[i] specifies window
 *                size for element i. If window_col = NULL, then window is used as
 *                the static window size for all elements
 * @param[in] min_periods_col The minimum number of observation values,
 *                min_periods_col[i] specifies minimum number of observations for
 *                element i. If min_periods_col = NULL, then min_periods is used as
 *                the static value for all elements
 * @param[in] forward_window_col The forward window size values,
 *                forward_window_col[i] specifies forward window size for element i.
 *                If forward_window_col = NULL, then forward_window is used as the
 *                static forward window size for all elements

 */
template <typename T, typename agg_op, bool average, int block_size, typename WindowIterator>
__launch_bounds__(block_size)
__global__
void gpu_rolling(column_device_view input,
                 mutable_column_device_view output,
                 WindowIterator window_begin,
                 WindowIterator forward_window_begin,
                 size_type min_periods)
{
  size_type i = blockIdx.x * block_size + threadIdx.x;
  size_type stride = block_size * gridDim.x;

  agg_op op;

  auto active_threads = __ballot_sync(0xffffffff, i < input.size());
  while(i < input.size())
  {
    T val = agg_op::template identity<T>();
    // declare this as volatile to avoid some compiler optimizations that lead to incorrect results
    // for CUDA 10.0 and below (fixed in CUDA 10.1)
    volatile cudf::size_type count = 0;

    size_type window = window_begin[i];
    size_type forward_window = forward_window_begin[i];

    // compute bounds
    size_type start_index = max(0, i - window + 1);
    size_type end_index = min(input.size(), i + forward_window + 1);       // exclusive

    // aggregate
    // TODO: We should explore using shared memory to avoid redundant loads.
    //       This might require separating the kernel into a special version
    //       for dynamic and static sizes.
    for (size_type j = start_index; j < end_index; j++) {
      if (!input.nullable() || input.is_valid(j)) {
        val = op(input.element<T>(j), val);
        count++;
      }
    }

    // check if we have enough input samples
    bool output_is_valid = (count >= min_periods);

    // set the mask
    cudf::bitmask_type result_mask{__ballot_sync(active_threads, output_is_valid)};

    // only one thread writes the mask
    if (0 == threadIdx.x % cudf::experimental::detail::warp_size)
      output.set_mask_word(cudf::word_index(i), result_mask);

    // store the output value, one per thread
    if (output_is_valid)
      cudf::detail::store_output_functor<T, average>{}(output.element<T>(i), val, count);

    // process next element 
    i += stride;
    active_threads = __ballot_sync(active_threads, i < input.size());
  }
}

struct rolling_window_launcher
{
  template<typename T, typename agg_op, bool average, typename WindowIterator,
      typename std::enable_if_t<cudf::detail::is_supported<T, agg_op>(), std::nullptr_t> = nullptr>
  std::unique_ptr<column> dispatch_aggregation_type(column_view const& input,
                                                    WindowIterator window_begin,
                                                    WindowIterator forward_window_begin,
                                                    size_type min_periods,
                                                    rmm::mr::device_memory_resource *mr,
                                                    cudaStream_t stream)
  {
    cudf::nvtx::range_push("CUDF_ROLLING_WINDOW", cudf::nvtx::color::ORANGE);

    std::unique_ptr<column> output =
        allocate_like(input, cudf::experimental::mask_allocation_policy::RETAIN, mr);

    constexpr cudf::size_type block_size = 256;
    cudf::size_type grid = (input.size() + block_size-1) / block_size;

    auto input_device_view = column_device_view::create(input);
    auto output_device_view = mutable_column_device_view::create(*output);

    gpu_rolling<T, agg_op, average, block_size><<<grid, block_size, 0, stream>>>
      (*input_device_view, *output_device_view, window_begin, forward_window_begin, min_periods);

    // check the stream for debugging
    CHECK_STREAM(stream);

    cudf::nvtx::range_pop();

    return output;
  }

  /**
   * @brief If we cannot perform aggregation on this type then throw an error
   */
  template<typename T, typename agg_op, bool average, typename WindowIterator,
     typename std::enable_if_t<!cudf::detail::is_supported<T, agg_op>(), std::nullptr_t> = nullptr>
  std::unique_ptr<column> dispatch_aggregation_type(column_view const& input,
                                                    WindowIterator window_begin,
                                                    WindowIterator forward_window_begin,
                                                    size_type min_periods,
                                                    rmm::mr::device_memory_resource *mr,
                                                    cudaStream_t stream)
  {
    CUDF_FAIL("Unsupported column type/operation combo. Only `min` and `max` are supported for "
              "non-arithmetic types for aggregations.");
  }

  /**
   * @brief Helper function for gdf_rolling. Deduces the type of the
   * aggregation column and type and calls another function to invoke the
   * rolling window kernel.
   */
  template <typename T, typename WindowIterator>
  std::unique_ptr<column> operator()(column_view const& input,
                                     WindowIterator window_begin,
                                     WindowIterator forward_window_begin,
                                     size_type min_periods,
                                     rolling_operator op,
                                     rmm::mr::device_memory_resource *mr,
                                     cudaStream_t stream)
  {
    switch (op) {
    case rolling_operator::SUM:
      return dispatch_aggregation_type<T, cudf::DeviceSum, false>(input, window_begin,
                                                                  forward_window_begin, min_periods,
                                                                  mr, stream);
    case rolling_operator::MIN:
      return dispatch_aggregation_type<T, cudf::DeviceMin, false>(input, window_begin,
                                                                  forward_window_begin, min_periods,
                                                                  mr, stream);
    case rolling_operator::MAX:
      return dispatch_aggregation_type<T, cudf::DeviceMax, false>(input, window_begin,
                                                                  forward_window_begin, min_periods,
                                                                  mr, stream);
    case rolling_operator::COUNT:
      return dispatch_aggregation_type<T, cudf::DeviceCount, false>(input, window_begin,
                                                                    forward_window_begin, min_periods,
                                                                    mr, stream);
    case rolling_operator::MEAN:
      return dispatch_aggregation_type<T, cudf::DeviceSum, true>(input, window_begin,
                                                                 forward_window_begin, min_periods,
                                                                 mr, stream);
    default:
      // TODO: need a nice way to convert enums to strings, same would be useful for groupby
      CUDF_FAIL("Rolling aggregation function not implemented");
    }
  }
};

} // namespace anonymous

// Applies a rolling window function to the values in a column.
template <typename WindowIterator>
std::unique_ptr<column> rolling_window(column_view const& input,
                                       WindowIterator window_begin,
                                       WindowIterator forward_window_begin,
                                       size_type min_periods,
                                       rolling_operator op,
                                       rmm::mr::device_memory_resource* mr,
                                       cudaStream_t stream = 0)
{
  return cudf::experimental::type_dispatcher(input.type(),
                                             rolling_window_launcher{},
                                             input, window_begin, forward_window_begin,
                                             min_periods, op, mr, stream);
}

// Applies a user-defined rolling window function to the values in a column.
template <typename WindowIterator>
std::unique_ptr<column> rolling_window(column_view const &input,
                                       WindowIterator window_begin,
                                       WindowIterator forward_window_begin,
                                       size_type min_periods,
                                       std::string const& user_defined_aggregator,
                                       rolling_operator agg_op,
                                       data_type output_type,
                                       rmm::mr::device_memory_resource* mr,
                                       cudaStream_t stream = 0)
{
  // TODO
  return cudf::make_numeric_column(data_type{INT32}, 0);
}

} // namespace detail

// Applies a fixed-size rolling window function to the values in a column.
std::unique_ptr<column> rolling_window(column_view const& input,
                                       size_type window,
                                       size_type forward_window,
                                       size_type min_periods,
                                       rolling_operator op,
                                       rmm::mr::device_memory_resource* mr)
{
  auto window_begin = thrust::make_constant_iterator(window);
  auto forward_window_begin = thrust::make_constant_iterator(forward_window);

  return cudf::experimental::detail::rolling_window(input, window_begin, forward_window_begin,
                                                    min_periods, op, mr, 0);
}

// Applies a variable-size rolling window function to the values in a column.
std::unique_ptr<column> rolling_window(column_view const& input,
                                       column_view const& window,
                                       column_view const& forward_window,
                                       size_type min_periods,
                                       rolling_operator op,
                                       rmm::mr::device_memory_resource* mr)
{
  if (input.size() == 0 || window.size() == 0)
    return cudf::make_numeric_column(data_type{INT32}, 0);
  
  CUDF_EXPECTS(window.type().id() == INT32 && forward_window.type().id() == INT32,
               "window/forward_window must have INT32 type");

  CUDF_EXPECTS(window.size() != input.size() && forward_window.size() != input.size(),
               "window/forward_window size must match input size");

  return cudf::experimental::detail::rolling_window(input, window.begin<size_type>(),
                                                    forward_window.begin<size_type>(), min_periods,
                                                    op, mr, 0);
}

// Applies a fixed-size user-defined rolling window function to the values in a column.
std::unique_ptr<column> rolling_window(column_view const &input,
                                       size_type window,
                                       size_type forward_window,
                                       size_type min_periods,
                                       std::string const& user_defined_aggregator,
                                       rolling_operator op,
                                       data_type output_type,
                                       rmm::mr::device_memory_resource* mr)
{
  auto window_begin = thrust::make_constant_iterator(window);
  auto forward_window_begin = thrust::make_constant_iterator(forward_window);

  return cudf::experimental::detail::rolling_window(input, window_begin, forward_window_begin,
                                                    min_periods, user_defined_aggregator, op,
                                                    output_type, mr, 0);
}

// Applies a variable-size user-defined rolling window function to the values in a column.
std::unique_ptr<column> rolling_window(column_view const &input,
                                       column_view const& window,
                                       column_view const& forward_window,
                                       size_type min_periods,
                                       std::string const& user_defined_aggregator,
                                       rolling_operator op,
                                       data_type output_type,
                                       rmm::mr::device_memory_resource* mr)
{
  if (input.size() == 0 || window.size() == 0)
    return cudf::make_numeric_column(data_type{INT32}, 0);
  
  CUDF_EXPECTS(window.type().id() == INT32 && forward_window.type().id() == INT32,
               "window/forward_window must have INT32 type");

  CUDF_EXPECTS(window.size() != input.size() && forward_window.size() != input.size(),
               "window/forward_window size must match input size");

  return cudf::experimental::detail::rolling_window(input, window.begin<size_type>(),
                                                    forward_window.begin<size_type>(), min_periods,
                                                    user_defined_aggregator, op, output_type, mr, 0);
}

} // namespace experimental 
} // namespace cudf
