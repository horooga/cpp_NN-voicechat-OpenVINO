// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "audio_utils.hpp"

#include <fstream>
#include <iostream>
#include <vector>

#include "openvino/core/except.hpp"

#define DR_WAV_IMPLEMENTATION
#include <dr_wav.h>

namespace utils {
namespace audio {

std::vector<float> record_audio(int sample_rate, int channels, int seconds) {
	PaError err = Pa_Initialize();
	if (err != paNoError) {
		throw std::runtime_error("Failed to initialize PortAudio");
	}

	PaStream *stream;
	PaStreamParameters inputParams;
	inputParams.device = Pa_GetDefaultInputDevice();
	if (inputParams.device == paNoDevice) {
		Pa_Terminate();
		throw std::runtime_error("No default input device");
	}
	inputParams.channelCount = channels;
	inputParams.sampleFormat = paFloat32; // 32-bit floating point
	inputParams.suggestedLatency =
		Pa_GetDeviceInfo(inputParams.device)->defaultLowInputLatency;
	inputParams.hostApiSpecificStreamInfo = nullptr;

	err = Pa_OpenStream(&stream, &inputParams,
						nullptr, // no output
						sample_rate, paFramesPerBufferUnspecified, paClipOff,
						nullptr, nullptr);
	if (err != paNoError) {
		Pa_Terminate();
		throw std::runtime_error("Failed to open PortAudio stream");
	}

	err = Pa_StartStream(stream);
	if (err != paNoError) {
		Pa_CloseStream(stream);
		Pa_Terminate();
		throw std::runtime_error("Failed to start PortAudio stream");
	}

	std::cout << "Recording for " << seconds << " seconds..." << std::endl;

	size_t total_frames = sample_rate * seconds;
	std::vector<float> buffer(total_frames * channels);

	err = Pa_ReadStream(stream, buffer.data(), total_frames);
	if (err != paNoError) {
		Pa_StopStream(stream);
		Pa_CloseStream(stream);
		Pa_Terminate();
		throw std::runtime_error("Failed to read audio stream");
	}

	err = Pa_StopStream(stream);
	Pa_CloseStream(stream);
	Pa_Terminate();

	std::cout << "Recording finished." << std::endl;

	return buffer;
}

void save_to_wav(const float *waveform_ptr, size_t waveform_size,
				 const std::filesystem::path &file_path,
				 uint32_t bits_per_sample) {
	drwav_data_format format;
	format.container = drwav_container_riff;
	format.format = DR_WAVE_FORMAT_IEEE_FLOAT;
	format.channels = 1;
	format.sampleRate = 16000;
	format.bitsPerSample = bits_per_sample;

	drwav wav;
	OPENVINO_ASSERT(drwav_init_file_write(&wav, file_path.string().c_str(),
										  &format, nullptr),
					"Failed to initialize WAV writer");

	size_t total_samples = waveform_size * format.channels;

	drwav_uint64 frames_written =
		drwav_write_pcm_frames(&wav, total_samples, waveform_ptr);
	OPENVINO_ASSERT(frames_written == total_samples,
					"Failed to write not all frames");

	drwav_uninit(&wav);
}
} // namespace audio
} // namespace utils
