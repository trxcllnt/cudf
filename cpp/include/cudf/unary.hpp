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

#pragma once

#include "types.hpp"

namespace cudf
{

/**
 * @brief Types of unary operations that can be performed on data.
 */
enum unary_op{
  SIN,          ///< Trigonometric sine
  COS,          ///< Trigonometric cosine
  TAN,          ///< Trigonometric tangent
  ARCSIN,       ///< Trigonometric sine inverse
  ARCCOS,       ///< Trigonometric cosine inverse
  ARCTAN,       ///< Trigonometric tangent inverse
  EXP,          ///< Exponential (base e, Euler number)
  LOG,          ///< Natural Logarithm (base e)
  SQRT,         ///< Square-root (x^0.5)
  CEIL,         ///< Smallest integer value not less than arg
  FLOOR,        ///< largest integer value not greater than arg
  ABS,          ///< Absolute value
  BIT_INVERT,   ///< Bitwise Not (~)
  NOT,          ///< Logical Not (!)
};

/**
 * @brief  Performs unary op on all values in column
 * 
 * @param column_view Input column
 * @param unary_op operation to perform
 *
 * @returns unique_ptr<column> Result of the operation
 */
// std::unique_ptr<column> unary_operation(column_view const& input, unary_op op);


/**
 * @brief  Casts data from dtype specified in input to dtype specified in output
 * 
 * @note In case of conversion from GDF_DATE32/GDF_DATE64/GDF_TIMESTAMP to
 *  GDF_TIMESTAMP, the time unit for output should be set in out_info.time_unit
 *
 * @param column_view Input column
 * @param out_type Desired datatype of output column
 * @param out_info Extra info for output column in case of conversion to types
 *  that require extra info
 *
 * @returns unique_ptr<column> Result of the cast operation
 */
std::unique_ptr<column> cast(column_view const& input, data_type out_type);


/**
 * @brief Checks the `input` column for `null` values, and creates a `bool`
 * column of same size with `true` representing `null` values and `false` for
 * other.
 *
 * @param input A column_view as input
 *
 * @returns unique_ptr<column> A column of type GDF_BOOL8 with `true` representing `null` values.
 */
// std::unique_ptr<column> is_null(column_view const& input);

/**
 * @brief Checks the `input` column for `null` values, and creates a `bool`
 * column of same size with `false` representing `null` values and `true` for
 * other.
 *
 * @param input A column_view as input
 *
 * @returns unique_ptr<column> A column of type GDF_BOOL8 with `false` representing `null` values.
 */
// std::unique_ptr<column> is_not_null(column_view const& input);

} // namespace cudf
