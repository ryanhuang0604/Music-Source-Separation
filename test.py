import numpy as np
import torch
import librosa
import tqdm


def predict_slice_2(args, x_part, model_arc, model_en0, model_en1, hop_length = 1024):
	stft = librosa.stft(x_part, hop_length = hop_length)
	stft_mag = np.log1p(np.abs(stft))
	phase = np.angle(stft)

	out = model_arc(torch.from_numpy(stft_mag[None,:,:]).cuda())
	if (args.mode == "arc"):			# test ARC
		out_vocal = out[:, :1025, :]
		out_other = out[:, 1025:, :]
	elif (args.mode == "enhancement"):	# test enhancement
		out_vocal = model_en0(out[:, :1025, :])
		out_other = model_en1(out[:, 1025:, :])

	# Mag spectrogram for voice
	out_vocal = out_vocal.cpu().detach().numpy()[0]
	out_vocal  = np.exp(out_vocal) - 1

	# Mag spectrogram for others
	out_other = out_other.cpu().detach().numpy()[0]
	out_other  = np.exp(out_other) - 1

	# Get phase from original mixture
	phase = np.angle(librosa.stft(x_part, hop_length = hop_length))

	# ISTFT for voice
	out_voice = out_vocal * np.exp(1j*phase)
	y_vocal = librosa.istft(out_voice , hop_length = hop_length)

	# ISTFT for others
	out_other = out_other * np.exp(1j*phase)
	y_other = librosa.istft(out_other, hop_length = hop_length)

	return out_vocal, out_other, y_vocal, y_other, stft


def predict_song_2(args, x, model_arc, model_en0, model_en1, hop_length = 1024):
	#TODO: overlap prediction
	GPU_avail = True
	win_len = 1024*255
	pad = 1024*111 # 112 + 32 + 112
	hop = 1024*31
	out_vocal_total = np.zeros((1025, 1))
	out_other_total = np.zeros((1025, 1))
	stft_original = np.zeros((1025, 1))
	l = len(x)
	x_pad = np.pad(x, (0, win_len), mode = "constant")

	for i in tqdm.tqdm(range(0, l, win_len)):
		part = x_pad[i:i + win_len]
		o_v, o_o, y_v, y_o, stft_ori = predict_slice_2(args, part, model_arc, model_en0, model_en1)
	
		out_vocal_total = np.concatenate((out_vocal_total, o_v[:, :]), axis = 1)
		out_other_total = np.concatenate((out_other_total, o_o[:, :]), axis = 1)
		stft_original = np.concatenate((stft_original, stft_ori), axis = 1)
	
	out_vocal_total = out_vocal_total[:, 1:]
	out_other_total = out_other_total[:, 1:]
	stft_original = stft_original[:, 1:]
	
	est = MWF_2(out_vocal_total, out_other_total, stft_original)
	y_vocal = librosa.istft(est[0] , hop_length =  hop_length)
	y_other = librosa.istft(est[1] , hop_length =  hop_length)
							
	return est, y_vocal, y_other


def MWF_2(source0, source1, stft_original, iter_n = 3, M = 256):
	#  Multi-channel Wiener filtering
	
	# Initilization
	stack = np.stack((source0, source1))
	original_stack = np.stack((stft_original, stft_original))
	v = 0.5 * np.power(np.abs(stack), 2)

	est = np.array(stack)
	est_new = np.zeros(shape = est.shape, dtype = 'complex128')
	l = stft_original.shape[-1]

	# Estimate
	for i in range(iter_n):
		m = M
		for ii in range(0, l, m):
			if (ii + m  >= l):
				m = l - ii
			# Compute spatial covarience matrix
			R = np.sum(est[:, :, ii:ii + m]*est[:, :, ii:ii + m].conj(), axis = 2)
			
			# Second term to prevent dividing by zero
			v_sum = np.sum(v[:, :, ii:ii + m], axis = 2) + 0.0000001*np.ones(shape = (2, v.shape[1]))
			R /= v_sum
	
			# Estimate stft
			for iii in range(m):
				d = (np.sum(v[:, :, ii + iii]*R, axis = 0) + 0.0000001*np.ones(shape = (R.shape[-1])))
				est_new[:, :, ii + iii] = v[:, :, ii + iii]*R/d * original_stack[:, :, ii + iii]
	 
		# Update
		est = np.array(est_new)
		v = 0.5 * np.power(np.abs(est), 2)
		
	return est


def predict_slice_4(args, x_part, model_arc, model_en0, model_en1, model_en2, model_en3, hop_length = 1024):
	stft = librosa.stft(x_part, hop_length = hop_length)
	stft_mag = np.log1p(np.abs(stft))
	phase = np.angle(stft)

	out = model_arc(torch.from_numpy(stft_mag[None,:,:]).cuda())
	if (args.mode == "arc"):			# test ARC
		out_vocal = out[:, :1025, :]
		out_drum = out[:, 1025:2050, :]
		out_bass = out[:, 2050:3075, :]
		out_other = out[:, 3075:, :]
	elif (args.mode == "enhancement"):	# test enhancement
		out_vocal = model_en0(out[:, :1025, :])
		out_drum = model_en1(out[:, 1025:2050, :])
		out_bass = model_en2(out[:, 2050:3075, :])
		out_other = model_en3(out[:, 3075:, :])
	
	# Mag spectrogram for voice
	out_vocal = out_vocal.cpu().detach().numpy()[0]
	out_vocal = np.exp(out_vocal) - 1

	# Mag spectrogram for drum
	out_drum = out_drum.cpu().detach().numpy()[0]
	out_drum = np.exp(out_drum) - 1

	# Mag spectrogram for bass
	out_bass = out_bass.cpu().detach().numpy()[0]
	out_bass = np.exp(out_bass) - 1

	# Mag spectrogram for other
	out_other = out_other.cpu().detach().numpy()[0]
	out_other = np.exp(out_other) - 1


	# Get phase from original mixture
	phase = np.angle(librosa.stft(x_part, hop_length = hop_length))

	# ISTFT for voice
	out_voice = out_vocal * np.exp(1j*phase)
	y_vocal = librosa.istft(out_voice , hop_length = hop_length)

	# ISTFT for drum
	out_drum = out_drum * np.exp(1j*phase)
	y_drum = librosa.istft(out_drum , hop_length = hop_length)

	# ISTFT for bass
	out_bass = out_bass * np.exp(1j*phase)
	y_bass = librosa.istft(out_bass , hop_length = hop_length)

	# ISTFT for others
	out_other = out_other * np.exp(1j*phase)
	y_other = librosa.istft(out_other, hop_length = hop_length)

	return out_vocal, out_drum, out_bass, out_other, y_vocal, y_drum, y_bass, y_other, stft


def predict_song_4(args, x, model_arc, model_en0, model_en1, model_en2, model_en3, hop_length = 1024):
	#TODO: overlap prediction
	GPU_avail = True
	win_len = 1024*255
	pad = 1024*111 # 112 + 32 + 112
	hop = 1024*31
	out_vocal_total = np.zeros((1025, 1))
	out_drum_total = np.zeros((1025, 1))
	out_bass_total = np.zeros((1025, 1))
	out_other_total = np.zeros((1025, 1))
	stft_original = np.zeros((1025, 1))
	l = len(x)
	x_pad = np.pad(x, (0, win_len), mode = "constant")

	for i in tqdm.tqdm(range(0, l, win_len)):
		part = x_pad[i:i + win_len]
		o_v, o_d, o_b, o_o, y_v, y_d, y_b, y_o, stft_ori = predict_slice_4(args, part, model_arc, model_en0, model_en1, model_en2, model_en3)
	
		out_vocal_total = np.concatenate((out_vocal_total, o_v[:, :]), axis = 1)
		out_drum_total = np.concatenate((out_drum_total, o_d[:, :]), axis = 1)
		out_bass_total = np.concatenate((out_bass_total, o_b[:, :]), axis = 1)
		out_other_total = np.concatenate((out_other_total, o_o[:, :]), axis = 1)
		stft_original = np.concatenate((stft_original, stft_ori), axis = 1)
	
	out_vocal_total = out_vocal_total[:, 1:]
	out_drum_total = out_drum_total[:, 1:]
	out_bass_total = out_bass_total[:, 1:]
	out_other_total = out_other_total[:, 1:]
	stft_original = stft_original[:, 1:]
	
	est = MWF_4(out_vocal_total, out_drum_total, out_bass_total, out_other_total, stft_original)
	y_vocal = librosa.istft(est[0] , hop_length =  hop_length)
	y_drum = librosa.istft(est[1] , hop_length =  hop_length)
	y_bass = librosa.istft(est[2] , hop_length =  hop_length)
	y_other = librosa.istft(est[3] , hop_length =  hop_length)
							
	return est, y_vocal, y_drum, y_bass, y_other


def MWF_4(source0, source1, source2, source3, stft_original, iter_n = 3, M = 256):
	#  Multi-channel Wiener filtering
	
	# Initilization
	stack = np.stack((source0, source1,source2, source3))
	original_stack = np.stack((stft_original, stft_original, stft_original, stft_original))
	v = 0.5 * np.power(np.abs(stack), 2)

	est = np.array(stack)
	est_new = np.zeros(shape = est.shape, dtype = 'complex128')
	l = stft_original.shape[-1]

	# Estimate
	for i in range(iter_n):
		m = M
		for ii in range(0, l, m):
			if (ii + m  >= l):
				m = l - ii
			# Compute spatial covarience matrix
			R = np.sum(est[:, :, ii:ii + m]*est[:, :, ii:ii + m].conj(), axis = 2)
			
			# Second term to prevent dividing by zero
			v_sum = np.sum(v[:, :, ii:ii + m], axis = 2) + 0.0000001*np.ones(shape = (4, v.shape[1]))
			R /= v_sum
	
			# Estimate stft
			for iii in range(m):
				d = (np.sum(v[:, :, ii + iii]*R, axis = 0) + 0.0000001*np.ones(shape = (R.shape[-1])))
				est_new[:, :, ii + iii] = v[:, :, ii + iii]*R/d * original_stack[:, :, ii + iii]
	 
		# Update
		est = np.array(est_new)
		v = 0.5 * np.power(np.abs(est), 2)
		
	return est