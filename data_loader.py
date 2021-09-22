import numpy as np
import torch
import glob
import librosa
from torch.utils.data.dataset import Dataset
from utils import STFT, datasets_importer


type = ['vocals', 'drums', 'bass', 'other', 'mixture']

class CustomDataset(Dataset):
	def __init__(self, path, source, samples=3322, sample_rate=44100, hop_size=1024, transforms=None, mode="train"):
		'''
		path: path = train/
		'''
		path2 = glob.glob1(path, "*")
		if (source == 4):
			mixture = []
			vocals = []
			drums = []
			bass = []
			other = []
			for f in path2:
				mix = path+f+"/mixture"
				voc = path+f+"/vocals"
				dru = path+f+"/drums"
				bas = path+f+"/bass"
				oth =  path+f+"/other"
				file = glob.glob1(mix, "*.npy")
				for k in file:
					mixture.append(mix+"/{}".format(k))
					vocals.append(voc+"/{}".format(k))
					drums.append(dru+"/{}".format(k))
					bass.append(bas+"/{}".format(k))
					other.append(oth+"/{}".format(k)) 					
			if (mode == "train"):
				mixture = mixture[:int(samples*0.8)]
				vocals = vocals[:int(samples*0.8)]
				drums = drums[:int(samples*0.8)]
				bass = bass[:int(samples*0.8)]
				other = other[:int(samples*0.8)]
			elif (mode == "val"):
				mixture = mixture[int(samples*0.8):]
				vocals = vocals[int(samples*0.8):]
				drums = drums[int(samples*0.8):]
				bass = bass[int(samples*0.8):]
				other = other[int(samples*0.8):]

			self.mixture = mixture
			self.vocals = vocals
			self.drums  = drums
			self.bass = bass
			self.other = other
			self.sample_rate = 44100
			self.hop_size = hop_size
			self.source = source

		elif (source == 2):
			mixture = []
			vocals = []
			accom = []
			for f in path2:
				mix = path+f+"/mixture"
				voc = path+f+"/vocals"
				acc = path+f+"/accompaniment"
				file = glob.glob1(mix, "*.npy")
				for k in file:
					mixture.append(mix+"/{}".format(k))
					vocals.append(voc+"/{}".format(k))
					accom.append(acc+"/{}".format(k))				
			if (mode == "train"):
				mixture = mixture[:int(samples*0.8)]
				vocals = vocals[:int(samples*0.8)]
				accom = accom[:int(samples*0.8)]
			elif (mode == "val"):
				mixture = mixture[int(samples*0.8):]
				vocals = vocals[int(samples*0.8):]
				accom = accom[int(samples*0.8):]

			self.mixture = mixture
			self.vocals = vocals
			self.accom = accom
			self.sample_rate = 44100
			self.hop_size = hop_size
			self.source = source


	def __getitem__(self, id):
		if (self.source == 4): 
			input = np.load(self.mixture[id])
			vocals= np.load(self.vocals[id])
			drums = np.load(self.drums[id])
			bass = np.load(self.bass[id])
			other = np.load(self.other[id])
			output = np.concatenate((vocals, drums, bass, other), axis=0)
			
			input = torch.from_numpy(np.log1p(np.abs(input)))
			output = torch.from_numpy(np.log1p(np.abs(output)))
		
		elif (self.source == 2):
			input = np.load(self.mixture[id])
			vocals = np.load(self.vocals[id])
			accom = np.load(self.accom[id])
			output = np.concatenate((vocals, accom), axis=0)
			
			input = torch.from_numpy(np.log1p(np.abs(input)))
			output = torch.from_numpy(np.log1p(np.abs(output)))
			
		return input, output


	def __len__(self):
		return len(self.mixture)