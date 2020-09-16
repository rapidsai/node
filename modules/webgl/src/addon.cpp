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

#include "webgl.hpp"

#include <nv_node/utilities/napi_to_cpp.hpp>

std::ostream& operator<<(std::ostream& os, const nv::NapiToCPP& self) {
  return os << self.operator std::string();
};

Napi::Object initModule(Napi::Env env, Napi::Object exports) {
  nv::WebGL2RenderingContext::Init(env, exports);
  nv::WebGLActiveInfo::Init(env, exports);
  nv::WebGLShaderPrecisionFormat::Init(env, exports);
  nv::WebGLBuffer::Init(env, exports);
  nv::WebGLContextEvent::Init(env, exports);
  nv::WebGLFramebuffer::Init(env, exports);
  nv::WebGLProgram::Init(env, exports);
  nv::WebGLQuery::Init(env, exports);
  nv::WebGLRenderbuffer::Init(env, exports);
  nv::WebGLSampler::Init(env, exports);
  nv::WebGLShader::Init(env, exports);
  nv::WebGLSync::Init(env, exports);
  nv::WebGLTexture::Init(env, exports);
  nv::WebGLTransformFeedback::Init(env, exports);
  nv::WebGLUniformLocation::Init(env, exports);
  nv::WebGLVertexArrayObject::Init(env, exports);

  return exports;
}

NODE_API_MODULE(node_webgl, initModule);
