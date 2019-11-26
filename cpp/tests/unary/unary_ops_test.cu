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

#include <cudf/cudf.h>
#include <cudf/column/column_factories.hpp>
#include <cudf/legacy/interop.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/wrappers/timestamps.hpp>
#include <initializer_list>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/legacy/cudf_test_utils.cuh>
#include <tests/utilities/type_lists.hpp>
#include <vector>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
 
template <typename T>
cudf::test::fixed_width_column_wrapper<T> create_fixed_columns(cudf::size_type start, cudf::size_type size, bool nullable) {
    auto iter = cudf::test::make_counting_transform_iterator(start, [](auto i) { return T(i);});

    if(not nullable) {
        return cudf::test::fixed_width_column_wrapper<T> (iter, iter + size);
    } else {
        auto valids = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i%2==0? true:false; });
        return  cudf::test::fixed_width_column_wrapper<T> (iter, iter + size, valids);
    }

}

template <typename T>
cudf::test::fixed_width_column_wrapper<T> create_expected_columns(cudf::size_type size, bool nullable, bool nulls_to_be) {

    if(not nullable) {
        auto iter = cudf::test::make_counting_transform_iterator(0, [nulls_to_be](auto i) { return not nulls_to_be;});
        return cudf::test::fixed_width_column_wrapper<T> (iter, iter + size);
    } else {
        auto iter = cudf::test::make_counting_transform_iterator(0, [nulls_to_be](auto i) { return i%2==0? not nulls_to_be: nulls_to_be; });
        return cudf::test::fixed_width_column_wrapper<T> (iter, iter + size);
    }
}

template <typename T>
struct IsNull : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(IsNull, cudf::test::NumericTypes);

TYPED_TEST(IsNull, AllValid)
{
    using T = TypeParam;

    cudf::size_type start = 0;
    cudf::size_type size = 10;
    cudf::test::fixed_width_column_wrapper<T> col = create_fixed_columns<T>(start, size, false);
    cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected = create_expected_columns<cudf::experimental::bool8>(size, false, true);

    std::unique_ptr<cudf::column> got = cudf::experimental::is_null(col);

    cudf::test::expect_columns_equal(expected, got->view());
}

TYPED_TEST(IsNull, WithInvalids)
{
    using T = TypeParam;

    cudf::size_type start = 0;
    cudf::size_type size = 10;
    cudf::test::fixed_width_column_wrapper<T> col = create_fixed_columns<T>(start, size, true);
    cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected = create_expected_columns<cudf::experimental::bool8>(size, true, true);

    std::unique_ptr<cudf::column> got = cudf::experimental::is_null(col);

    cudf::test::expect_columns_equal(expected, got->view());
}

TYPED_TEST(IsNull, EmptyColumns)
{
    using T = TypeParam;

    cudf::size_type start = 0;
    cudf::size_type size = 0;
    cudf::test::fixed_width_column_wrapper<T> col = create_fixed_columns<T>(start, size, true);
    cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected = create_expected_columns<cudf::experimental::bool8>(size, true, true);

    std::unique_ptr<cudf::column> got = cudf::experimental::is_null(col);

    cudf::test::expect_columns_equal(expected, got->view());
}

template <typename T>
struct IsNotNull : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(IsNotNull, cudf::test::NumericTypes);

TYPED_TEST(IsNotNull, AllValid)
{
    using T = TypeParam;

    cudf::size_type start = 0;
    cudf::size_type size = 10;
    cudf::test::fixed_width_column_wrapper<T> col = create_fixed_columns<T>(start, size, false);
    cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected = create_expected_columns<cudf::experimental::bool8>(size, false, false);

    std::unique_ptr<cudf::column> got = cudf::experimental::is_valid(col);

    cudf::test::expect_columns_equal(expected, got->view());
}

TYPED_TEST(IsNotNull, WithInvalids)
{
    using T = TypeParam;

    cudf::size_type start = 0;
    cudf::size_type size = 10;
    cudf::test::fixed_width_column_wrapper<T> col = create_fixed_columns<T>(start, size, true);
    cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected = create_expected_columns<cudf::experimental::bool8>(size, true, false);

    std::unique_ptr<cudf::column> got = cudf::experimental::is_valid(col);

    cudf::test::expect_columns_equal(expected, got->view());
}

TYPED_TEST(IsNotNull, EmptyColumns)
{
    using T = TypeParam;

    cudf::size_type start = 0;
    cudf::size_type size = 0;
    cudf::test::fixed_width_column_wrapper<T> col = create_fixed_columns<T>(start, size, true);
    cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected = create_expected_columns<cudf::experimental::bool8>(size, true, false);

    std::unique_ptr<cudf::column> got = cudf::experimental::is_valid(col);

    cudf::test::expect_columns_equal(expected, got->view());
}

static const auto test_timestamps_D = std::vector<int32_t>{
    -1528,  // 1965-10-26
    17716,  // 2018-07-04
    19382,  // 2023-01-25
};

static const auto test_timestamps_s = std::vector<int64_t>{
    -131968728,  // 1965-10-26 14:01:12
    1530705600,  // 2018-07-04 12:00:00
    1674631932,  // 2023-01-25 07:32:12
};

static const auto test_timestamps_ms = std::vector<int64_t>{
    -131968727238,  // 1965-10-26 14:01:12.762
    1530705600000,  // 2018-07-04 12:00:00.000
    1674631932929,  // 2023-01-25 07:32:12.929
};

static const auto test_timestamps_us = std::vector<int64_t>{
    -131968727238000,  // 1965-10-26 14:01:12.762000000
    1530705600000000,  // 2018-07-04 12:00:00.000000000
    1674631932929000,  // 2023-01-25 07:32:12.929000000
};

static const auto test_timestamps_ns = std::vector<int64_t>{
    -131968727238000000,  // 1965-10-26 14:01:12.762000000
    1530705600000000000,  // 2018-07-04 12:00:00.000000000
    1674631932929000000,  // 2023-01-25 07:32:12.929000000
};

template <typename T>
inline auto make_data_type() {
  return cudf::data_type{cudf::experimental::type_to_id<T>()};
}

template <typename T, typename R>
inline auto make_column(std::vector<R> data) {
  return cudf::test::fixed_width_column_wrapper<T>{data.begin(), data.end()};
}

template <typename T, typename R>
inline auto make_column(std::vector<R> data,
                        std::vector<bool> mask) {
  return cudf::test::fixed_width_column_wrapper<T>{data.begin(), data.end(),
                                                   mask.begin()};
}

template <typename T, typename R>
void validate_cast_result(cudf::column_view expected,
                          cudf::column_view actual) {
  using namespace cudf::test;
  // round-trip through the host because sizeof(T) may not equal sizeof(R)
  std::vector<T> h_data;
  std::vector<cudf::bitmask_type> null_mask;
  std::tie(h_data, null_mask) = to_host<T>(expected);
  if (null_mask.size() == 0) {
    expect_columns_equal(make_column<R, T>(h_data), actual);
  } else {
    std::vector<bool> h_null_mask{};
    for (cudf::size_type i = 0; i < expected.size(); ++i) {
      h_null_mask.push_back(cudf::bit_is_set(null_mask.data(), i));
    }
    expect_columns_equal(make_column<R, T>(h_data, h_null_mask), actual);
  }
}

struct CastTimestamps : public cudf::test::BaseFixture {};
TEST_F(CastTimestamps, IsIdempotent) {
  using namespace cudf::test;

  auto timestamps_D = make_column<cudf::timestamp_D>(test_timestamps_D);
  auto timestamps_s = make_column<cudf::timestamp_s>(test_timestamps_s);
  auto timestamps_ms = make_column<cudf::timestamp_ms>(test_timestamps_ms);
  auto timestamps_us = make_column<cudf::timestamp_us>(test_timestamps_us);
  auto timestamps_ns = make_column<cudf::timestamp_ns>(test_timestamps_ns);

  auto timestamps_D_rep = cudf::experimental::cast(
      timestamps_D, make_data_type<cudf::timestamp_D::rep>());
  auto timestamps_s_rep = cudf::experimental::cast(
      timestamps_s, make_data_type<cudf::timestamp_s::rep>());
  auto timestamps_ms_rep = cudf::experimental::cast(
      timestamps_ms, make_data_type<cudf::timestamp_ms::rep>());
  auto timestamps_us_rep = cudf::experimental::cast(
      timestamps_us, make_data_type<cudf::timestamp_us::rep>());
  auto timestamps_ns_rep = cudf::experimental::cast(
      timestamps_ns, make_data_type<cudf::timestamp_ns::rep>());

  auto timestamps_D_got = cudf::experimental::cast(
      *timestamps_D_rep, cudf::data_type{cudf::TIMESTAMP_DAYS});
  auto timestamps_s_got = cudf::experimental::cast(
      *timestamps_s_rep, cudf::data_type{cudf::TIMESTAMP_SECONDS});
  auto timestamps_ms_got = cudf::experimental::cast(
      *timestamps_ms_rep, cudf::data_type{cudf::TIMESTAMP_MILLISECONDS});
  auto timestamps_us_got = cudf::experimental::cast(
      *timestamps_us_rep, cudf::data_type{cudf::TIMESTAMP_MICROSECONDS});
  auto timestamps_ns_got = cudf::experimental::cast(
      *timestamps_ns_rep, cudf::data_type{cudf::TIMESTAMP_NANOSECONDS});

  validate_cast_result<cudf::timestamp_D, cudf::timestamp_D>(timestamps_D,
                                                             *timestamps_D_got);
  validate_cast_result<cudf::timestamp_s, cudf::timestamp_s>(timestamps_s,
                                                             *timestamps_s_got);
  validate_cast_result<cudf::timestamp_ms, cudf::timestamp_ms>(
      timestamps_ms, *timestamps_ms_got);
  validate_cast_result<cudf::timestamp_us, cudf::timestamp_us>(
      timestamps_us, *timestamps_us_got);
  validate_cast_result<cudf::timestamp_ns, cudf::timestamp_ns>(
      timestamps_ns, *timestamps_ns_got);
}

template <typename T>
struct CastToTimestamps : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(CastToTimestamps, cudf::test::NumericTypes);

TYPED_TEST(CastToTimestamps, AllValid) {
  using T = TypeParam;
  using namespace cudf::test;

  auto timestamps_D = make_column<T>(test_timestamps_D);
  auto timestamps_s = make_column<T>(test_timestamps_s);
  auto timestamps_ms = make_column<T>(test_timestamps_ms);
  auto timestamps_us = make_column<T>(test_timestamps_us);
  auto timestamps_ns = make_column<T>(test_timestamps_ns);

  auto timestamps_D_got = cudf::experimental::cast(
      timestamps_D, cudf::data_type{cudf::TIMESTAMP_DAYS});
  auto timestamps_s_got = cudf::experimental::cast(
      timestamps_s, cudf::data_type{cudf::TIMESTAMP_SECONDS});
  auto timestamps_ms_got = cudf::experimental::cast(
      timestamps_ms, cudf::data_type{cudf::TIMESTAMP_MILLISECONDS});
  auto timestamps_us_got = cudf::experimental::cast(
      timestamps_us, cudf::data_type{cudf::TIMESTAMP_MICROSECONDS});
  auto timestamps_ns_got = cudf::experimental::cast(
      timestamps_ns, cudf::data_type{cudf::TIMESTAMP_NANOSECONDS});

  validate_cast_result<T, cudf::timestamp_D>(timestamps_D, *timestamps_D_got);
  validate_cast_result<T, cudf::timestamp_s>(timestamps_s, *timestamps_s_got);
  validate_cast_result<T, cudf::timestamp_ms>(timestamps_ms,
                                              *timestamps_ms_got);
  validate_cast_result<T, cudf::timestamp_us>(timestamps_us,
                                              *timestamps_us_got);
  validate_cast_result<T, cudf::timestamp_ns>(timestamps_ns,
                                              *timestamps_ns_got);
}

template <typename T>
struct CastFromTimestamps : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(CastFromTimestamps, cudf::test::NumericTypes);

TYPED_TEST(CastFromTimestamps, AllValid) {
  using T = TypeParam;
  using namespace cudf::test;

  auto timestamps_D = make_column<cudf::timestamp_D>(test_timestamps_D);
  auto timestamps_s = make_column<cudf::timestamp_s>(test_timestamps_s);
  auto timestamps_ms = make_column<cudf::timestamp_ms>(test_timestamps_ms);
  auto timestamps_us = make_column<cudf::timestamp_us>(test_timestamps_us);
  auto timestamps_ns = make_column<cudf::timestamp_ns>(test_timestamps_ns);

  auto timestamps_D_exp = make_column<T>(test_timestamps_D);
  auto timestamps_s_exp = make_column<T>(test_timestamps_s);
  auto timestamps_ms_exp = make_column<T>(test_timestamps_ms);
  auto timestamps_us_exp = make_column<T>(test_timestamps_us);
  auto timestamps_ns_exp = make_column<T>(test_timestamps_ns);

  auto timestamps_D_got =
      cudf::experimental::cast(timestamps_D, make_data_type<T>());
  auto timestamps_s_got =
      cudf::experimental::cast(timestamps_s, make_data_type<T>());
  auto timestamps_ms_got =
      cudf::experimental::cast(timestamps_ms, make_data_type<T>());
  auto timestamps_us_got =
      cudf::experimental::cast(timestamps_us, make_data_type<T>());
  auto timestamps_ns_got =
      cudf::experimental::cast(timestamps_ns, make_data_type<T>());

  validate_cast_result<T, T>(timestamps_D_exp, *timestamps_D_got);
  validate_cast_result<T, T>(timestamps_s_exp, *timestamps_s_got);
  validate_cast_result<T, T>(timestamps_ms_exp, *timestamps_ms_got);
  validate_cast_result<T, T>(timestamps_us_exp, *timestamps_us_got);
  validate_cast_result<T, T>(timestamps_ns_exp, *timestamps_ns_got);
}

TYPED_TEST(CastFromTimestamps, WithNulls) {
  using T = TypeParam;
  using namespace cudf::test;

  auto timestamps_D =
      make_column<cudf::timestamp_D>(test_timestamps_D, {true, false, true});
  auto timestamps_s =
      make_column<cudf::timestamp_s>(test_timestamps_s, {true, false, true});
  auto timestamps_ms =
      make_column<cudf::timestamp_ms>(test_timestamps_ms, {true, false, true});
  auto timestamps_us =
      make_column<cudf::timestamp_us>(test_timestamps_us, {true, false, true});
  auto timestamps_ns =
      make_column<cudf::timestamp_ns>(test_timestamps_ns, {true, false, true});

  auto timestamps_D_exp =
      make_column<T>(test_timestamps_D, {true, false, true});
  auto timestamps_s_exp =
      make_column<T>(test_timestamps_s, {true, false, true});
  auto timestamps_ms_exp =
      make_column<T>(test_timestamps_ms, {true, false, true});
  auto timestamps_us_exp =
      make_column<T>(test_timestamps_us, {true, false, true});
  auto timestamps_ns_exp =
      make_column<T>(test_timestamps_ns, {true, false, true});

  auto timestamps_D_got =
      cudf::experimental::cast(timestamps_D, make_data_type<T>());
  auto timestamps_s_got =
      cudf::experimental::cast(timestamps_s, make_data_type<T>());
  auto timestamps_ms_got =
      cudf::experimental::cast(timestamps_ms, make_data_type<T>());
  auto timestamps_us_got =
      cudf::experimental::cast(timestamps_us, make_data_type<T>());
  auto timestamps_ns_got =
      cudf::experimental::cast(timestamps_ns, make_data_type<T>());

  validate_cast_result<T, T>(timestamps_D_exp, *timestamps_D_got);
  validate_cast_result<T, T>(timestamps_s_exp, *timestamps_s_got);
  validate_cast_result<T, T>(timestamps_ms_exp, *timestamps_ms_got);
  validate_cast_result<T, T>(timestamps_us_exp, *timestamps_us_got);
  validate_cast_result<T, T>(timestamps_ns_exp, *timestamps_ns_got);
}
