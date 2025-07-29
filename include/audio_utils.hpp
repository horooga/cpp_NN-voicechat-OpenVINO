// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <portaudio.h>
#include <string>
#include <vector>

constexpr int SAMPLE_RATE = 16000;
constexpr int CHANNELS = 1;
constexpr int SECONDS = 5;

#include "openvino/runtime/tensor.hpp"

namespace utils {
namespace audio {
/**
 * This function saves an audio waveform, provided as an array of floating-point
 * samples, to a WAV file.
 *
 * @param waveform_ptr Pointer to the array of float samples representing the
 * audio waveform
 * @param waveform_size The number of samples in the waveform array
 * @param file_path The name (and path) of the WAV file to be created
 * @param bits_per_sample The bit depth used to store each sample in the WAV
 * file
 */
std::vector<float> record_audio(int sample_rate, int channels, int seconds);
void save_to_wav(const float *waveform_ptr, size_t waveform_size,
				 const std::filesystem::path &file_path,
				 uint32_t bits_per_sample);

/**
 * This function reads a binary file containing speaker embedding or 32-bit
 * floating-point values and returns ov::Tensor
 *
 * @param file_path The path to the binary file to be read
 * @returns a std::vector<float> containing all float values read from the
 * binary file
 */
} // namespace audio
} // namespace utils
