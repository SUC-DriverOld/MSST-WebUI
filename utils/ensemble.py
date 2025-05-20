# coding: utf-8
__author__ = "Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/"

import os
import librosa
import soundfile as sf
import numpy as np

from utils.logger import get_logger

logger = get_logger()


def stft(wave, nfft, hl):
	wave_left = np.asfortranarray(wave[0])
	wave_right = np.asfortranarray(wave[1])
	spec_left = librosa.stft(wave_left, n_fft=nfft, hop_length=hl)
	spec_right = librosa.stft(wave_right, n_fft=nfft, hop_length=hl)
	spec = np.asfortranarray([spec_left, spec_right])
	return spec


def istft(spec, hl, length):
	spec_left = np.asfortranarray(spec[0])
	spec_right = np.asfortranarray(spec[1])
	wave_left = librosa.istft(spec_left, hop_length=hl, length=length)
	wave_right = librosa.istft(spec_right, hop_length=hl, length=length)
	wave = np.asfortranarray([wave_left, wave_right])
	return wave


def absmax(a, *, axis):
	dims = list(a.shape)
	dims.pop(axis)
	indices = np.ogrid[tuple(slice(0, d) for d in dims)]
	argmax = np.abs(a).argmax(axis=axis)
	indices.insert((len(a.shape) + axis) % len(a.shape), argmax)
	return a[tuple(indices)]


def absmin(a, *, axis):
	dims = list(a.shape)
	dims.pop(axis)
	indices = np.ogrid[tuple(slice(0, d) for d in dims)]
	argmax = np.abs(a).argmin(axis=axis)
	indices.insert((len(a.shape) + axis) % len(a.shape), argmax)
	return a[tuple(indices)]


def lambda_max(arr, axis=None, key=None, keepdims=False):
	idxs = np.argmax(key(arr), axis)
	if axis is not None:
		idxs = np.expand_dims(idxs, axis)
		result = np.take_along_axis(arr, idxs, axis)
		if not keepdims:
			result = np.squeeze(result, axis=axis)
		return result
	else:
		return arr.flatten()[idxs]


def lambda_min(arr, axis=None, key=None, keepdims=False):
	idxs = np.argmin(key(arr), axis)
	if axis is not None:
		idxs = np.expand_dims(idxs, axis)
		result = np.take_along_axis(arr, idxs, axis)
		if not keepdims:
			result = np.squeeze(result, axis=axis)
		return result
	else:
		return arr.flatten()[idxs]


def average_waveforms(pred_track, weights, algorithm):
	"""
	:param pred_track: shape = (num, channels, length)
	:param weights: shape = (num, )
	:param algorithm: One of avg_wave, median_wave, min_wave, max_wave, avg_fft, median_fft, min_fft, max_fft
	:return: averaged waveform in shape (channels, length)
	"""

	pred_track = np.array(pred_track)
	final_length = pred_track.shape[-1]

	mod_track = []
	for i in range(pred_track.shape[0]):
		if algorithm == "avg_wave":
			mod_track.append(pred_track[i] * weights[i])
		elif algorithm in ["median_wave", "min_wave", "max_wave"]:
			mod_track.append(pred_track[i])
		elif algorithm in ["avg_fft", "min_fft", "max_fft", "median_fft"]:
			spec = stft(pred_track[i], nfft=2048, hl=1024)
			if algorithm in ["avg_fft"]:
				mod_track.append(spec * weights[i])
			else:
				mod_track.append(spec)
	pred_track = np.array(mod_track)

	if algorithm in ["avg_wave"]:
		pred_track = pred_track.sum(axis=0)
		pred_track /= np.array(weights).sum().T
	elif algorithm in ["median_wave"]:
		pred_track = np.median(pred_track, axis=0)
	elif algorithm in ["min_wave"]:
		pred_track = np.array(pred_track)
		pred_track = lambda_min(pred_track, axis=0, key=np.abs)
	elif algorithm in ["max_wave"]:
		pred_track = np.array(pred_track)
		pred_track = lambda_max(pred_track, axis=0, key=np.abs)
	elif algorithm in ["avg_fft"]:
		pred_track = pred_track.sum(axis=0)
		pred_track /= np.array(weights).sum()
		pred_track = istft(pred_track, 1024, final_length)
	elif algorithm in ["min_fft"]:
		pred_track = np.array(pred_track)
		pred_track = lambda_min(pred_track, axis=0, key=np.abs)
		pred_track = istft(pred_track, 1024, final_length)
	elif algorithm in ["max_fft"]:
		pred_track = np.array(pred_track)
		pred_track = absmax(pred_track, axis=0)
		pred_track = istft(pred_track, 1024, final_length)
	elif algorithm in ["median_fft"]:
		pred_track = np.array(pred_track)
		pred_track = np.median(pred_track, axis=0)
		pred_track = istft(pred_track, 1024, final_length)
	return pred_track


def ensemble_audios(files, type, weights):
	logger.info(f"Ensemble type: {type}, Number of input files: {len(files)}, Weights: {weights}")
	if weights is None:
		weights = np.ones(len(files))
	data = []
	sr = 44100
	for f in files:
		if not os.path.isfile(f):
			logger.error(f"Can't find file: {f}. Check paths.")
			return None
		wav, sr = librosa.load(f, sr=None, mono=False)
		logger.debug(f"Reading file: {f}, waveform shape: {wav.shape}, sample rate: {sr}")
		data.append(wav)

	lengths = [d.shape[-1] for d in data]
	min_length = min(lengths)
	if len(set(lengths)) > 1:
		logger.warning("Input audio files have different lengths. Truncating all to the shortest length.")
		data = [d[..., :min_length] for d in data]

	data = np.array(data)
	res = average_waveforms(data, weights, type)
	logger.debug("Result shape: {}".format(res.shape))
	return res.T, sr
