// Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include "webgl.hpp"

#include <nv_node/addon.hpp>
#include <nv_node/utilities/napi_to_cpp.hpp>

#include <napi.h>

std::ostream& operator<<(std::ostream& os, const nv::NapiToCPP& self) {
  return os << self.operator std::string();
};

struct rapidsai_webgl : public nv::EnvLocalAddon, public Napi::Addon<rapidsai_webgl> {
  rapidsai_webgl(Napi::Env const& env, Napi::Object exports) : nv::EnvLocalAddon(env, exports) {
    DefineAddon(
      exports,
      {
        InstanceValue("_cpp_exports", _cpp_exports.Value()),
        InstanceMethod("init", &rapidsai_webgl::InitAddon),
        InstanceValue("WebGL2RenderingContext",
                      InitClass<nv::WebGL2RenderingContext>(env, exports)),
        InstanceValue("WebGLActiveInfo", InitClass<nv::WebGLActiveInfo>(env, exports)),
        InstanceValue("WebGLShaderPrecisionFormat",
                      InitClass<nv::WebGLShaderPrecisionFormat>(env, exports)),
        InstanceValue("WebGLBuffer", InitClass<nv::WebGLBuffer>(env, exports)),
        InstanceValue("WebGLContextEvent", InitClass<nv::WebGLContextEvent>(env, exports)),
        InstanceValue("WebGLFramebuffer", InitClass<nv::WebGLFramebuffer>(env, exports)),
        InstanceValue("WebGLProgram", InitClass<nv::WebGLProgram>(env, exports)),
        InstanceValue("WebGLQuery", InitClass<nv::WebGLQuery>(env, exports)),
        InstanceValue("WebGLRenderbuffer", InitClass<nv::WebGLRenderbuffer>(env, exports)),
        InstanceValue("WebGLSampler", InitClass<nv::WebGLSampler>(env, exports)),
        InstanceValue("WebGLShader", InitClass<nv::WebGLShader>(env, exports)),
        InstanceValue("WebGLSync", InitClass<nv::WebGLSync>(env, exports)),
        InstanceValue("WebGLTexture", InitClass<nv::WebGLTexture>(env, exports)),
        InstanceValue("WebGLTransformFeedback",
                      InitClass<nv::WebGLTransformFeedback>(env, exports)),
        InstanceValue("WebGLUniformLocation", InitClass<nv::WebGLUniformLocation>(env, exports)),
        InstanceValue("WebGLVertexArrayObject",
                      InitClass<nv::WebGLVertexArrayObject>(env, exports)),
      });
  }
};

NODE_API_ADDON(rapidsai_webgl);
