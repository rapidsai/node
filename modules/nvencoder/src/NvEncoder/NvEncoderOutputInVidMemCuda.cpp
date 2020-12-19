/*
 * Copyright 2019-2020 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include "NvEncoder/NvEncoderOutputInVidMemCuda.h"

NvEncoderOutputInVidMemCuda::NvEncoderOutputInVidMemCuda(CUcontext cuContext,
                                                         uint32_t nWidth,
                                                         uint32_t nHeight,
                                                         NV_ENC_BUFFER_FORMAT eBufferFormat,
                                                         bool bMotionEstimationOnly)
  : NvEncoderCuda(cuContext, nWidth, nHeight, eBufferFormat, 0, bMotionEstimationOnly, true) {}

NvEncoderOutputInVidMemCuda::~NvEncoderOutputInVidMemCuda() {
  try {
    FlushEncoder();
    ReleaseOutputBuffers();
  } catch (...) {}
}

uint32_t NvEncoderOutputInVidMemCuda::GetOutputBufferSize() {
  uint32_t bufferSize = 0;

  if (m_bMotionEstimationOnly) {
    uint32_t encodeWidthInMbs  = (GetEncodeWidth() + 15) >> 4;
    uint32_t encodeHeightInMbs = (GetEncodeHeight() + 15) >> 4;

    bufferSize = encodeWidthInMbs * encodeHeightInMbs * sizeof(NV_ENC_H264_MV_DATA);
  } else {
    // 2-times the input size
    bufferSize = GetFrameSize() * 2;

    bufferSize += sizeof(NV_ENC_ENCODE_OUT_PARAMS);
  }

  bufferSize = ALIGN_UP(bufferSize, 4);

  return bufferSize;
}

void NvEncoderOutputInVidMemCuda::AllocateOutputBuffers(uint32_t numOutputBuffers) {
  uint32_t size = GetOutputBufferSize();

  CUDA_DRVAPI_CALL(cuCtxPushCurrent(m_cuContext));

  for (uint32_t i = 0; i < numOutputBuffers; i++) {
    CUdeviceptr pDeviceFrame;

    CUresult cuResult = cuMemAlloc(&pDeviceFrame, size);
    if (cuResult != CUDA_SUCCESS) {
      NVENC_THROW_ERROR("cuMemAlloc Failed", NV_ENC_ERR_OUT_OF_MEMORY);
    }

    m_pOutputBuffers.push_back((NV_ENC_OUTPUT_PTR)pDeviceFrame);
  }

  CUDA_DRVAPI_CALL(cuCtxPopCurrent(NULL));

  RegisterOutputResources(size);
}

void NvEncoderOutputInVidMemCuda::ReleaseOutputBuffers() {
  if (!m_hEncoder) { return; }

  UnregisterOutputResources();

  for (uint32_t i = 0; i < m_pOutputBuffers.size(); ++i) {
    cuMemFree(reinterpret_cast<CUdeviceptr>(m_pOutputBuffers[i]));
  }

  m_pOutputBuffers.clear();
}

void NvEncoderOutputInVidMemCuda::RegisterOutputResources(uint32_t bfrSize) {
  NV_ENC_BUFFER_USAGE bufferUsage =
    m_bMotionEstimationOnly ? NV_ENC_OUTPUT_MOTION_VECTOR : NV_ENC_OUTPUT_BITSTREAM;

  for (uint32_t i = 0; i < m_pOutputBuffers.size(); ++i) {
    if (m_pOutputBuffers[i]) {
      NV_ENC_REGISTERED_PTR registeredPtr =
        RegisterResource((void *)m_pOutputBuffers[i],
                         NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR,
                         bfrSize,
                         1,
                         bfrSize,
                         NV_ENC_BUFFER_FORMAT_U8,
                         bufferUsage);

      m_vRegisteredResourcesOutputBuffer.push_back(registeredPtr);
    }
  }
}

void NvEncoderOutputInVidMemCuda::UnregisterOutputResources() {
  for (uint32_t i = 0; i < m_vMappedOutputBuffers.size(); ++i) {
    if (m_vMappedOutputBuffers[i]) {
      m_nvenc.nvEncUnmapInputResource(m_hEncoder, m_vMappedOutputBuffers[i]);
    }
  }
  m_vMappedOutputBuffers.clear();

  for (uint32_t i = 0; i < m_vRegisteredResourcesOutputBuffer.size(); ++i) {
    if (m_vRegisteredResourcesOutputBuffer[i]) {
      m_nvenc.nvEncUnregisterResource(m_hEncoder, m_vRegisteredResourcesOutputBuffer[i]);
    }
  }

  m_vRegisteredResourcesOutputBuffer.clear();
}

void NvEncoderOutputInVidMemCuda::CreateEncoder(const NV_ENC_INITIALIZE_PARAMS *pEncoderParams) {
  NvEncoder::CreateEncoder(pEncoderParams);

  AllocateOutputBuffers(m_nEncoderBuffer);

  m_vMappedOutputBuffers.resize(m_nEncoderBuffer, nullptr);
}

void NvEncoderOutputInVidMemCuda::MapResources(uint32_t bfrIdx) {
  NvEncoder::MapResources(bfrIdx);

  // map output surface
  NV_ENC_MAP_INPUT_RESOURCE mapInputResourceBitstreamBuffer = {NV_ENC_MAP_INPUT_RESOURCE_VER};
  mapInputResourceBitstreamBuffer.registeredResource = m_vRegisteredResourcesOutputBuffer[bfrIdx];
  NVENC_API_CALL(m_nvenc.nvEncMapInputResource(m_hEncoder, &mapInputResourceBitstreamBuffer));
  m_vMappedOutputBuffers[bfrIdx] = mapInputResourceBitstreamBuffer.mappedResource;
}

void NvEncoderOutputInVidMemCuda::EncodeFrame(std::vector<NV_ENC_OUTPUT_PTR> &pOutputBuffer,
                                              NV_ENC_PIC_PARAMS *pPicParams) {
  pOutputBuffer.clear();
  if (!IsHWEncoderInitialized()) {
    NVENC_THROW_ERROR("Encoder device not found", NV_ENC_ERR_NO_ENCODE_DEVICE);
  }

  int bfrIdx = m_iToSend % m_nEncoderBuffer;

  MapResources(bfrIdx);

  NVENCSTATUS nvStatus =
    DoEncode(m_vMappedInputBuffers[bfrIdx], m_vMappedOutputBuffers[bfrIdx], pPicParams);

  if (nvStatus == NV_ENC_SUCCESS || nvStatus == NV_ENC_ERR_NEED_MORE_INPUT) {
    m_iToSend++;
    GetEncodedPacket(pOutputBuffer, true);
  } else {
    NVENC_THROW_ERROR("nvEncEncodePicture API failed", nvStatus);
  }
}

void NvEncoderOutputInVidMemCuda::EndEncode(std::vector<NV_ENC_OUTPUT_PTR> &pOutputBuffer) {
  if (!IsHWEncoderInitialized()) {
    NVENC_THROW_ERROR("Encoder device not initialized", NV_ENC_ERR_ENCODER_NOT_INITIALIZED);
  }

  SendEOS();

  GetEncodedPacket(pOutputBuffer, false);
}

void NvEncoderOutputInVidMemCuda::RunMotionEstimation(
  std::vector<NV_ENC_OUTPUT_PTR> &pOutputBuffer) {
  pOutputBuffer.clear();

  if (!m_hEncoder) {
    NVENC_THROW_ERROR("Encoder Initialization failed", NV_ENC_ERR_NO_ENCODE_DEVICE);
    return;
  }

  const uint32_t bfrIdx = m_iToSend % m_nEncoderBuffer;

  MapResources(bfrIdx);

  NVENCSTATUS nvStatus = DoMotionEstimation(
    m_vMappedInputBuffers[bfrIdx], m_vMappedRefBuffers[bfrIdx], m_vMappedOutputBuffers[bfrIdx]);

  if (nvStatus == NV_ENC_SUCCESS) {
    m_iToSend++;
    GetEncodedPacket(pOutputBuffer, true);
  } else {
    NVENC_THROW_ERROR("nvEncRunMotionEstimationOnly API failed", nvStatus);
  }
}

void NvEncoderOutputInVidMemCuda::GetEncodedPacket(std::vector<NV_ENC_OUTPUT_PTR> &pOutputBuffer,
                                                   bool bOutputDelay) {
  unsigned i = 0;
  int iEnd   = bOutputDelay ? m_iToSend - m_nOutputDelay : m_iToSend;

  for (; m_iGot < iEnd; m_iGot++) {
    if (m_vMappedOutputBuffers[m_iGot % m_nEncoderBuffer]) {
      NVENC_API_CALL(m_nvenc.nvEncUnmapInputResource(
        m_hEncoder, m_vMappedOutputBuffers[m_iGot % m_nEncoderBuffer]));
      m_vMappedOutputBuffers[m_iGot % m_nEncoderBuffer] = nullptr;
    }

    if (m_vMappedInputBuffers[m_iGot % m_nEncoderBuffer]) {
      NVENC_API_CALL(m_nvenc.nvEncUnmapInputResource(
        m_hEncoder, m_vMappedInputBuffers[m_iGot % m_nEncoderBuffer]));
      m_vMappedInputBuffers[m_iGot % m_nEncoderBuffer] = nullptr;
    }

    if (m_bMotionEstimationOnly && m_vMappedRefBuffers[m_iGot % m_nEncoderBuffer]) {
      NVENC_API_CALL(m_nvenc.nvEncUnmapInputResource(
        m_hEncoder, m_vMappedRefBuffers[m_iGot % m_nEncoderBuffer]));
      m_vMappedRefBuffers[m_iGot % m_nEncoderBuffer] = nullptr;
    }

    pOutputBuffer.push_back(m_pOutputBuffers[(m_iGot % m_nEncoderBuffer)]);

    i++;
  }
}

void NvEncoderOutputInVidMemCuda::FlushEncoder() {
  if (!m_hEncoder) { return; }

  if (!m_bMotionEstimationOnly) {
    std::vector<NV_ENC_OUTPUT_PTR> pOutputBuffer;
    EndEncode(pOutputBuffer);
  }
}

void NvEncoderOutputInVidMemCuda::DestroyEncoder() {
  if (!m_hEncoder) { return; }

  // Incase of error it is possible for buffers still mapped to encoder.
  // flush the encoder queue and then unmapped it if any surface is still mapped
  FlushEncoder();

  ReleaseOutputBuffers();

  NvEncoder::DestroyEncoder();
}