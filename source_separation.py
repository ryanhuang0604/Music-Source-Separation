import torch
import librosa
import pdb
import soundfile as sf
import librosa.display
import matplotlib.pyplot as plt
from model import *
from test import *


def SourceSeparation(args, song_path, sr):
	if (args.source == 4):
		if (args.normalization == "weight"):
			arc = ARC_weightNorm(sources=args.source)
			en_vocal = Enhancement_weightNorm()
			en_drums = Enhancement_weightNorm()
			en_bass = Enhancement_weightNorm()
			en_other = Enhancement_weightNorm()
		elif (args.normalization == "batch"):
			arc = ARC_batchNorm(sources=args.source)
			en_vocal = Enhancement_batchNorm()
			en_drums = Enhancement_batchNorm()
			en_bass = Enhancement_batchNorm()
			en_other = Enhancement_batchNorm()

		arc.cuda()
		arc.load_state_dict(torch.load("model/{}_{}_arc.pt".format(args.source, args.normalization)))
		arc.eval()

		en_vocal.cuda()
		en_vocal.load_state_dict(torch.load("model/{}_{}_vocal.pt".format(args.source, args.normalization)))
		en_vocal.eval()

		en_drums.cuda()
		en_drums.load_state_dict(torch.load("model/{}_{}_drums.pt".format(args.source, args.normalization)))
		en_drums.eval()

		en_bass.cuda()
		en_bass.load_state_dict(torch.load("model/{}_{}_bass.pt".format(args.source, args.normalization)))
		en_bass.eval()

		en_other.cuda()
		en_other.load_state_dict(torch.load("model/{}_{}_other.pt".format(args.source, args.normalization)))
		en_other.eval()

		x, sr = librosa.load(song_path, sr=sr)

		stft, y_vocal, y_drums, y_bass, y_other = predict_song_4(args, x, arc, en_vocal, en_drums, en_bass, en_other)
	
		return stft, y_vocal, y_drums, y_bass, y_other
	
	elif (args.source == 2):
		if (args.normalization == "weight"):
			arc = ARC_weightNorm(sources=args.source)
			en_vocal = Enhancement_weightNorm()
			en_other = Enhancement_weightNorm()
		elif (args.normalization == "batch"):
			arc = ARC_batchNorm(sources=args.source)
			en_vocal = Enhancement_batchNorm()
			en_other = Enhancement_batchNorm()

		arc.cuda()
		arc.load_state_dict(torch.load("model/{}_{}_arc.pt".format(args.source, args.normalization)))
		arc.eval()

		en_vocal.cuda()
		en_vocal.load_state_dict(torch.load("model/{}_{}_vocal.pt".format(args.source, args.normalization)))
		en_vocal.eval()

		en_other.cuda()
		en_other.load_state_dict(torch.load("model/{}_{}_other.pt".format(args.source, args.normalization)))
		en_other.eval()
	
		x, sr = librosa.load(song_path, sr = sr)
	
		stft, y_vocal, y_other = predict_song_2(args, x, arc, en_vocal, en_other)
	
		return stft, y_vocal, y_other