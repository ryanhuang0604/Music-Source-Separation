import torch
from torch.utils.data import DataLoader
import librosa
import pdb
import argparse
from data_loader import *
from train import *
from test import *
from source_separation import *


def str2bool(v):
	if v.lower() in ('yes','true','t','y','1'):
		return True
	elif v.lower() in ('no','false','f','n','0'):
		return False
	else:
		return argparse.ArgumentTypeError('Boolean value expected')


def parse_args():
	parser = argparse.ArgumentParser(description='DLP final')
	
	parser.add_argument('-workers', type=int, default=0)
	parser.add_argument('-manualSeed', type=int, help='manual seed')
	parser.add_argument('-pretrain', type=str2bool, nargs='?', default=True)
	parser.add_argument('-train_more', type=str2bool, nargs='?', default=False)
	parser.add_argument('-type', type=str, default="demo", help='demo or exp')
	parser.add_argument('-early_stopping', type=int, default=6)
	parser.add_argument('-batch_size', type=int, default=10, help='number of batchs size')
	parser.add_argument('-lr', type=float, default=0.001)

	parser.add_argument('-epochs', type=int, default=30, help='number of training epochs')
	parser.add_argument('-source', type=int, default=4, help='number of voice to separate')
	parser.add_argument('-state', type=str, default="train", help='train, test')
	parser.add_argument('-mode', type=str, default="arc", help='arc, enhancement')
	parser.add_argument('-normalization', type=str, default="weight", help='weight, batch')
	parser.add_argument('-en_type', type=str, default="vocal", help='vocal, drums, bass, other')
	
	parser.add_argument('-datapath', type=str, default="train/")
	parser.add_argument("-songpath",default='dataset/wav/test/Lyndsey Ollard - Catching Up/mixture.wav', type=str)

	args = parser.parse_args()

	return args


if __name__== '__main__':
	args = parse_args()
	print(args)

	if (args.type == "exp"):
		wandb.init(project='Music',name='Music_%d' % (args.source))

	if (args.state == "train"):
		# load data
		train_dataset = CustomDataset(args.datapath, source=args.source, mode="train")
		val_dataset = CustomDataset(args.datapath, source=args.source, mode="val")

		train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
		val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
		print("dataloader finish")
		train(args, train_loader, val_loader, en_type=args.en_type)

	elif (args.state == "test"):
		if (args.source == 4):
			stft, y_vocal, y_drums, y_bass, y_other = SourceSeparation(args, args.songpath, sr=44100)
			sf.write("test/{}_{}_{}_vocal.wav".format(args.source, args.normalization, args.mode), y_vocal, 44100)
			sf.write("test/{}_{}_{}_drums.wav".format(args.source, args.normalization, args.mode), y_drums, 44100)
			sf.write("test/{}_{}_{}_bass.wav".format(args.source, args.normalization, args.mode), y_bass, 44100)
			sf.write("test/{}_{}_{}_other.wav".format(args.source, args.normalization, args.mode), y_other, 44100)
		elif (args.source == 2):  
			stft, y_vocal, y_other = SourceSeparation(args, args.songpath, sr=44100) 
			sf.write("test/{}_{}_{}_vocal.wav".format(args.source, args.normalization, args.mode), y_vocal, 44100)
			sf.write("test/{}_{}_{}_other.wav".format(args.source, args.normalization, args.mode), y_other, 44100)