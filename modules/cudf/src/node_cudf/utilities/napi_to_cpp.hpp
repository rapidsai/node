// Copyright (c) 2020, NVIDIA CORPORATION.
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

#include "node_cudf/scalar.hpp"
#include "node_cudf/types.hpp"

#include <cudf/types.hpp>
#include <nv_node/utilities/napi_to_cpp.hpp>

#include <napi.h>
#include <memory>
#include <type_traits>

namespace nv {

template <>
inline NapiToCPP::operator cudf::type_id() const {
  if (IsNumber()) { return static_cast<cudf::type_id>(operator int32_t()); }
  if (DataType::is_instance(val)) { return DataType::Unwrap(ToObject())->id(); }
  NAPI_THROW(Napi::Error::New(Env()), "Expected value to be a DataType or typeId");
}

template <>
inline NapiToCPP::operator cudf::duration_D() const {
  return cudf::duration_D{operator cudf::duration_D::rep()};
}

template <>
inline NapiToCPP::operator cudf::duration_s() const {
  return cudf::duration_s{operator cudf::duration_s::rep()};
}

template <>
inline NapiToCPP::operator cudf::duration_ms() const {
  return cudf::duration_ms{operator cudf::duration_ms::rep()};
}

template <>
inline NapiToCPP::operator cudf::duration_us() const {
  return cudf::duration_us{operator cudf::duration_us::rep()};
}

template <>
inline NapiToCPP::operator cudf::duration_ns() const {
  return cudf::duration_ns{operator cudf::duration_ns::rep()};
}

template <>
inline NapiToCPP::operator cudf::timestamp_D() const {
  return cudf::timestamp_D{operator cudf::duration_D()};
}

template <>
inline NapiToCPP::operator cudf::timestamp_s() const {
  return cudf::timestamp_s{operator cudf::duration_s()};
}

template <>
inline NapiToCPP::operator cudf::timestamp_ms() const {
  return cudf::timestamp_ms{operator cudf::duration_ms()};
}

template <>
inline NapiToCPP::operator cudf::timestamp_us() const {
  return cudf::timestamp_us{operator cudf::duration_us()};
}

template <>
inline NapiToCPP::operator cudf::timestamp_ns() const {
  return cudf::timestamp_ns{operator cudf::duration_ns()};
}

// template <>
// inline NapiToCPP::operator BooleanScalar*() const {
//   if (BooleanScalar::is_instance(val)) { return BooleanScalar::Unwrap(ToObject()); }
//   NAPI_THROW(Napi::Error::New(Env()), "Expected value to be a BooleanScalar");
// }

// template <>
// inline NapiToCPP::operator Int8Scalar*() const {
//   if (Int8Scalar::is_instance(val)) { return Int8Scalar::Unwrap(ToObject()); }
//   NAPI_THROW(Napi::Error::New(Env()), "Expected value to be a Int8Scalar");
// }

// template <>
// inline NapiToCPP::operator Int16Scalar*() const {
//   if (Int16Scalar::is_instance(val)) { return Int16Scalar::Unwrap(ToObject()); }
//   NAPI_THROW(Napi::Error::New(Env()), "Expected value to be a Int16Scalar");
// }

// template <>
// inline NapiToCPP::operator Int32Scalar*() const {
//   if (Int32Scalar::is_instance(val)) { return Int32Scalar::Unwrap(ToObject()); }
//   NAPI_THROW(Napi::Error::New(Env()), "Expected value to be a Int32Scalar");
// }

// template <>
// inline NapiToCPP::operator Int64Scalar*() const {
//   if (Int64Scalar::is_instance(val)) { return Int64Scalar::Unwrap(ToObject()); }
//   NAPI_THROW(Napi::Error::New(Env()), "Expected value to be a Int64Scalar");
// }

// template <>
// inline NapiToCPP::operator Uint8Scalar*() const {
//   if (Uint8Scalar::is_instance(val)) { return Uint8Scalar::Unwrap(ToObject()); }
//   NAPI_THROW(Napi::Error::New(Env()), "Expected value to be a Uint8Scalar");
// }

// template <>
// inline NapiToCPP::operator Uint16Scalar*() const {
//   if (Uint16Scalar::is_instance(val)) { return Uint16Scalar::Unwrap(ToObject()); }
//   NAPI_THROW(Napi::Error::New(Env()), "Expected value to be a Uint16Scalar");
// }

// template <>
// inline NapiToCPP::operator Uint32Scalar*() const {
//   if (Uint32Scalar::is_instance(val)) { return Uint32Scalar::Unwrap(ToObject()); }
//   NAPI_THROW(Napi::Error::New(Env()), "Expected value to be a Uint32Scalar");
// }

// template <>
// inline NapiToCPP::operator Uint64Scalar*() const {
//   if (Uint64Scalar::is_instance(val)) { return Uint64Scalar::Unwrap(ToObject()); }
//   NAPI_THROW(Napi::Error::New(Env()), "Expected value to be a Uint64Scalar");
// }

// template <>
// inline NapiToCPP::operator Float32Scalar*() const {
//   if (Float32Scalar::is_instance(val)) { return Float32Scalar::Unwrap(ToObject()); }
//   NAPI_THROW(Napi::Error::New(Env()), "Expected value to be a Float32Scalar");
// }

// template <>
// inline NapiToCPP::operator Float64Scalar*() const {
//   if (Float64Scalar::is_instance(val)) { return Float64Scalar::Unwrap(ToObject()); }
//   NAPI_THROW(Napi::Error::New(Env()), "Expected value to be a Float64Scalar");
// }

// template <>
// inline NapiToCPP::operator StringScalar*() const {
//   if (StringScalar::is_instance(val)) { return StringScalar::Unwrap(ToObject()); }
//   NAPI_THROW(Napi::Error::New(Env()), "Expected value to be a StringScalar");
// }

// template <>
// inline NapiToCPP::operator TimestampDayScalar*() const {
//   if (TimestampDayScalar::is_instance(val)) { return TimestampDayScalar::Unwrap(ToObject()); }
//   NAPI_THROW(Napi::Error::New(Env()), "Expected value to be a TimestampDayScalar");
// }

// template <>
// inline NapiToCPP::operator TimestampSecScalar*() const {
//   if (TimestampSecScalar::is_instance(val)) { return TimestampSecScalar::Unwrap(ToObject()); }
//   NAPI_THROW(Napi::Error::New(Env()), "Expected value to be a TimestampSecScalar");
// }

// template <>
// inline NapiToCPP::operator TimestampMilliScalar*() const {
//   if (TimestampMilliScalar::is_instance(val)) { return TimestampMilliScalar::Unwrap(ToObject());
//   } NAPI_THROW(Napi::Error::New(Env()), "Expected value to be a TimestampMilliScalar");
// }

// template <>
// inline NapiToCPP::operator TimestampMicroScalar*() const {
//   if (TimestampMicroScalar::is_instance(val)) { return TimestampMicroScalar::Unwrap(ToObject());
//   } NAPI_THROW(Napi::Error::New(Env()), "Expected value to be a TimestampMicroScalar");
// }

// template <>
// inline NapiToCPP::operator TimestampNanoScalar*() const {
//   if (TimestampNanoScalar::is_instance(val)) { return TimestampNanoScalar::Unwrap(ToObject()); }
//   NAPI_THROW(Napi::Error::New(Env()), "Expected value to be a TimestampNanoScalar");
// }

// template <>
// inline NapiToCPP::operator DurationDayScalar*() const {
//   if (DurationDayScalar::is_instance(val)) { return DurationDayScalar::Unwrap(ToObject()); }
//   NAPI_THROW(Napi::Error::New(Env()), "Expected value to be a DurationDayScalar");
// }

// template <>
// inline NapiToCPP::operator DurationSecScalar*() const {
//   if (DurationSecScalar::is_instance(val)) { return DurationSecScalar::Unwrap(ToObject()); }
//   NAPI_THROW(Napi::Error::New(Env()), "Expected value to be a DurationSecScalar");
// }

// template <>
// inline NapiToCPP::operator DurationMilliScalar*() const {
//   if (DurationMilliScalar::is_instance(val)) { return DurationMilliScalar::Unwrap(ToObject()); }
//   NAPI_THROW(Napi::Error::New(Env()), "Expected value to be a DurationMilliScalar");
// }

// template <>
// inline NapiToCPP::operator DurationMicroScalar*() const {
//   if (DurationMicroScalar::is_instance(val)) { return DurationMicroScalar::Unwrap(ToObject()); }
//   NAPI_THROW(Napi::Error::New(Env()), "Expected value to be a DurationMicroScalar");
// }

// template <>
// inline NapiToCPP::operator DurationNanoScalar*() const {
//   if (DurationNanoScalar::is_instance(val)) { return DurationNanoScalar::Unwrap(ToObject()); }
//   NAPI_THROW(Napi::Error::New(Env()), "Expected value to be a DurationNanoScalar");
// }

// template <>
// inline NapiToCPP::operator Decimal32Scalar*() const {
//   if (Decimal32Scalar::is_instance(val)) { return Decimal32Scalar::Unwrap(ToObject()); }
//   NAPI_THROW(Napi::Error::New(Env()), "Expected value to be a Decimal32Scalar");
// }

// template <>
// inline NapiToCPP::operator Decimal64Scalar*() const {
//   if (Decimal64Scalar::is_instance(val)) { return Decimal64Scalar::Unwrap(ToObject()); }
//   NAPI_THROW(Napi::Error::New(Env()), "Expected value to be a Decimal64Scalar");
// }

}  // namespace nv
