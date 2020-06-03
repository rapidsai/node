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

#include <node_webgl/casting.hpp>
#include <node_webgl/context.hpp>
#include <node_webgl/webgl.hpp>

std::ostream& operator<<(std::ostream& os, const node_webgl::FromJS& self) {
  return os << self.operator std::string();
};

Napi::Object initModule(Napi::Env env, Napi::Object exports) {
  node_webgl::WebGL2RenderingContext::Init(env, exports);
  node_webgl::WebGLActiveInfo::Init(env, exports);
  node_webgl::WebGLShaderPrecisionFormat::Init(env, exports);
  node_webgl::WebGLBuffer::Init(env, exports);
  node_webgl::WebGLContextEvent::Init(env, exports);
  node_webgl::WebGLFramebuffer::Init(env, exports);
  node_webgl::WebGLProgram::Init(env, exports);
  node_webgl::WebGLQuery::Init(env, exports);
  node_webgl::WebGLRenderbuffer::Init(env, exports);
  node_webgl::WebGLSampler::Init(env, exports);
  node_webgl::WebGLShader::Init(env, exports);
  node_webgl::WebGLSync::Init(env, exports);
  node_webgl::WebGLTexture::Init(env, exports);
  node_webgl::WebGLTransformFeedback::Init(env, exports);
  node_webgl::WebGLUniformLocation::Init(env, exports);
  node_webgl::WebGLVertexArrayObject::Init(env, exports);

  return exports;
}

NODE_API_MODULE(node_webgl, initModule);
