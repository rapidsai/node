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

#pragma once

#include <stdint.h>
#include <string.h>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>
#include "NvEncoder/NvEncoderCuda.h"
#include "nvEncodeAPI.h"

#define ALIGN_UP(s, a) (((s) + (a)-1) & ~((a)-1))

/**
 * @brief Class for encode or ME only output in video memory feature for Cuda interfaces.
 */
class NvEncoderOutputInVidMemCuda : public NvEncoderCuda {
 public:
  /**
   *  @brief  NvEncoderOutputInVidMem class constructor.
   */
  NvEncoderOutputInVidMemCuda(CUcontext cuContext,
                              uint32_t nWidth,
                              uint32_t nHeight,
                              NV_ENC_BUFFER_FORMAT eBufferFormat,
                              bool bMotionEstimationOnly = false);

  /**
   *  @brief  NvEncoder class virtual destructor.
   */
  virtual ~NvEncoderOutputInVidMemCuda();

  /**
   *  @brief This function is used to initialize the encoder session.
   *  Application must call this function to initialize the encoder, before
   *  starting to encode or motion estimate any frames.
   */
  void CreateEncoder(const NV_ENC_INITIALIZE_PARAMS *pEncoderParams);

  /**
   *  @brief  This function is used to encode a frame.
   *  Applications must call EncodeFrame() function to encode the uncompressed
   *  data, which has been copied to an input buffer obtained from the
   *  GetNextInputFrame() function.
   *  This function returns video memory buffer pointers containing compressed data
   *  in pOutputBuffer. If there is buffering enabled, this may return without
   *  any data in pOutputBuffer.
   */
  void EncodeFrame(std::vector<NV_ENC_OUTPUT_PTR> &pOutputBuffer,
                   NV_ENC_PIC_PARAMS *pPicParams = nullptr);

  /**
   *  @brief  This function to flush the encoder queue.
   *  The encoder might be queuing frames for B picture encoding or lookahead;
   *  the application must call EndEncode() to get all the queued encoded frames
   *  from the encoder. The application must call this function before destroying
   *  an encoder session. Video memory buffer pointer containing compressed data
   *  is returned in pOutputBuffer.
   */
  void EndEncode(std::vector<NV_ENC_OUTPUT_PTR> &pOutputBuffer);

  /**
   *  @brief  This function is used to run motion estimation.
   *  This is used to run motion estimation on a a pair of frames. The
   *  application must copy the reference frame data to the buffer obtained
   *  by calling GetNextReferenceFrame(), and copy the input frame data to
   *  the buffer obtained by calling GetNextInputFrame() before calling the
   *  RunMotionEstimation() function.
   *  This function returns video memory buffer pointers containing
   *  motion vector data in pOutputBuffer.
   */
  void RunMotionEstimation(std::vector<NV_ENC_OUTPUT_PTR> &pOutputBuffer);

  /**
   *  @brief  This function is used to destroy the encoder session.
   *  Application must call this function to destroy the encoder session and
   *  clean up any allocated resources. The application must call EndEncode()
   *  function to get any queued encoded frames before calling DestroyEncoder().
   */
  void DestroyEncoder();

  /**
   *  @brief This function is used to get the size of output buffer required to be
   *  allocated in order to store the output.
   */
  uint32_t GetOutputBufferSize();

 private:
  /**
   *  @brief This function is used to allocate output buffers in video memory for storing
   *  encode or motion estimation output.
   */
  void AllocateOutputBuffers(uint32_t numOutputBuffers);

  /**
   *  @brief This function is used to release output buffers.
   */
  void ReleaseOutputBuffers();

  /**
   *  @brief This function is used to register output buffers with NvEncodeAPI.
   */
  void RegisterOutputResources(uint32_t bfrSize);

  /**
   *  @brief This function is used to unregister output resources which had been previously
   * registered for encoding using RegisterOutputResources() function.
   */
  void UnregisterOutputResources();

  /**
   *  @brief This function is used to map the input and output buffers to NvEncodeAPI.
   */
  void MapResources(uint32_t bfrIdx);

  /**
   *  @brief This is a private function which is used to get video memory buffer pointer containing
   * compressed data or motion estimation output from the encoder HW. This is called by
   * EncodeFrame() function. If there is buffering enabled, this may return without any output data.
   */
  void GetEncodedPacket(std::vector<NV_ENC_OUTPUT_PTR> &pOutputBuffer, bool bOutputDelay);

  /**
   *  @brief This function is used to flush the encoder queue.
   */
  void FlushEncoder();

 private:
  std::vector<NV_ENC_OUTPUT_PTR> m_vMappedOutputBuffers;
  std::vector<NV_ENC_OUTPUT_PTR> m_pOutputBuffers;
  std::vector<NV_ENC_REGISTERED_PTR> m_vRegisteredResourcesOutputBuffer;
};
