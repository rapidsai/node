// // Copyright (c) 2020, NVIDIA CORPORATION.
// //
// // Licensed under the Apache License, Version 2.0 (the "License");
// // you may not use this file except in compliance with the License.
// // You may obtain a copy of the License at
// //
// //     http://www.apache.org/licenses/LICENSE-2.0
// //
// // Unless required by applicable law or agreed to in writing, software
// // distributed under the License is distributed on an "AS IS" BASIS,
// // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// // See the License for the specific language governing permissions and
// // limitations under the License.

// #pragma once

// #include "types.hpp"
// #include "visit_struct/visit_struct.hpp"

// #include <cuda_runtime_api.h>
// #include <nv_node/utilities/args.hpp>
// #include "utilities/napi_to_cpp.hpp"
// #include "utilities/cpp_to_napi.hpp"

// #include <type_traits>

// namespace nv {

// // template <>
// // Napi::Value inline CPPToNapi::operator()(cudaDeviceProp const& props) const {
// //   auto cast_t = *this;
// //   auto obj    = Napi::Object::New(env);
// //   visit_struct::for_each(props, [&](char const* name, auto const& val) {  //
// //     using T = typename std::decay<decltype(val)>::type;
// //     if (name == std::string{"name"}) {
// //       obj.Set(name, cast_t(std::string{reinterpret_cast<char const*>(&val)}));
// //     } else if (std::is_pointer<T>()) {
// //       using P = typename std::remove_pointer<T>::type;
// //       obj.Set(name, cast_t(reinterpret_cast<P*>(val), sizeof(val)));
// //     } else {
// //       obj.Set(name, cast_t(val));
// //     }
// //   });
// //   return obj;
// // }

// template <>
// inline NapiToCPP::operator cudaDeviceProp() const {
//   cudaDeviceProp props{};
//   if (val.IsObject()) {
//     auto obj = val.As<Napi::Object>();
//     visit_struct::for_each(props, [&](char const* key, auto& val) {
//       if (obj.Has(key) && !obj.Get(key).IsUndefined()) {
//         using T                     = typename std::decay<decltype(val)>::type;
//         *reinterpret_cast<T*>(&val) = NapiToCPP(obj.Get(key)).operator T();
//       }
//     });
//     // visit_struct::for_each(props, [&](const char* key, auto& val) {
//     //   using T = typename std::decay<decltype(val)>::type;
//     //   if (!obj.Has(key) || obj.Get(key).IsUndefined()) { return; }
//     //   if (!std::is_pointer<T>()) {
//     //     T* dst = reinterpret_cast<T*>(&val);
//     //     *dst   = NapiToCPP(obj.Get(key)).operator T();
//     //   } else {
//     //     using T_ = typename std::remove_pointer<T>::type;
//     //     T_* src  = reinterpret_cast<T_*>(NapiToCPP(obj.Get(key)).operator T());
//     //     memcpy(reinterpret_cast<T_*>(val), src, sizeof(val));
//     //     free(src);
//     //   }
//     // });
//   }
//   return props;
// }

// }  // namespace nv
