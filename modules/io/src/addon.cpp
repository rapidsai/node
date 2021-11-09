// Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <nv_node/utilities/args.hpp>

#include <laz.h>

#include <fstream>
#include <iterator>
#include <vector>

struct rapidsai_io : public nv::EnvLocalAddon, public Napi::Addon<rapidsai_io> {
  rapidsai_io(Napi::Env env, Napi::Object exports) : nv::EnvLocalAddon(env, exports) {
    DefineAddon(exports,
                {InstanceMethod("init", &rapidsai_io::InitAddon),
                 InstanceValue("_cpp_exports", _cpp_exports.Value()),
                 InstanceMethod<&rapidsai_io::read_laz>("readLaz")});
  }

 private:
  void read_laz(Napi::CallbackInfo const& info) {
    nv::CallbackArgs args{info};
    std::string path = args[0];

    std::ifstream input(path, std::ios::binary);
    std::vector<char> buffer(std::istreambuf_iterator<char>(input), {});

    // File Signature
    // 4 bytes
    std::string file_signature;
    file_signature += buffer[0];
    file_signature += buffer[1];
    file_signature += buffer[2];
    file_signature += buffer[3];
    std::cout << "file signature: " << file_signature << std::endl;

    // File Source ID
    // 2 bytes
    uint16_t file_source_id = buffer[4] + buffer[5];
    std::cout << "file source id: " << file_source_id << std::endl;

    // Global Encoding
    // 2 bytes
    uint16_t encoding = buffer[6] + buffer[7];
    std::cout << "global encoding: " << encoding << std::endl;

    // Project ID
    // 16 bytes

    // Major version
    // 1 byte
    int major_version = (int)buffer[24];
    std::cout << "major version: " << major_version << std::endl;

    // Minor version
    // 1 byte
    int minor_version = (int)buffer[25];
    std::cout << "minor version: " << minor_version << std::endl;

    // System Identifer
    // 32 bytes
    std::string system_identifer;
    for (int i = 0; i < 32; ++i) { system_identifer += buffer[i + 26]; }
    std::cout << "system identifer: " << system_identifer << std::endl;

    // Generating Software
    // 32 bytes
    std::string generating_software;
    for (int i = 0; i < 32; ++i) { generating_software += buffer[i + 58]; }
    std::cout << "generating software: " << generating_software << std::endl;

    // File Creation Day of Year
    // 2 bytes
    unsigned short day = buffer[90] + buffer[91];
    std::cout << "day: " << day << std::endl;

    // File Creation Year
    // 2 bytes
    unsigned short year = buffer[92] + buffer[93];
    std::cout << "year: " << year << std::endl;

    // Header Size
    // 2 bytes
    unsigned short header_size = buffer[94] + buffer[95];
    std::cout << "header size: " << header_size << std::endl;

    // Offset to point data
    // 4 bytes
    unsigned long offset_data = buffer[96] + buffer[97] + buffer[98] + buffer[99];
    std::cout << "offset data: " << offset_data << std::endl;

    // Number of Variable Length Records
    // 4 bytes
    unsigned long variable_length = buffer[100] + buffer[101] + buffer[102] + buffer[103];
    std::cout << "variable length: " << variable_length << std::endl;

    // Point Data Format ID
    // 1 byte
    unsigned short point_data_format = buffer[104];
    std::cout << "point data format id: " << point_data_format << std::endl;

    // Point Data Record Length
    // 2 bytes
    unsigned short point_data_length = buffer[105] + buffer[106];
    std::cout << "point data length: " << point_data_length << std::endl;

    // Number of point records
    // 4 bytes
    unsigned long number_of_point_records = buffer[107] + buffer[108] + buffer[109] + buffer[110];
    std::cout << "point data records: " << number_of_point_records << std::endl;

    // Number of points by return
    // 20 bytes
    unsigned long points_by_return = 0;
    for (int i = 0; i < 20; ++i) { points_by_return += buffer[i + 111]; }
    std::cout << "points by return: " << points_by_return << std::endl;

    // X scale factor
    // 8 bytes

    // Y scale factor
    // 8 bytes

    // Z scale factor
    // 8 bytes

    // X offset
    // 8 bytes

    // Y offset
    // 8 bytes

    // Z offset
    // 8 bytes

    // Max X
    // 8 bytes

    // Min X
    // 8 bytes

    // Max Y
    // 8 bytes

    // Min Y
    // 8 bytes

    // Max Z
    // 8 bytes

    // Min Z
    // 8 bytes

    throw std::invalid_argument("received negative value");

    io::read_laz();
  }
};

NODE_API_ADDON(rapidsai_io);
