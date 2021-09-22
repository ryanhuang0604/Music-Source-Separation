import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import tqdm
import pdb
import wandb
from model import *


def train(args, train_loader, validate_loader, en_type = None):
	'''
	model should always be the ARC models and model2 is for the enhancement models.
	'''
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if (args.mode == "arc"):
		if (args.normalization == "weight"):
			model = ARC_weightNorm(sources = args.source).to(device)
		elif (args.normalization == "batch"):
			model = ARC_batchNorm(sources = args.source).to(device)
		model2 = None
		optimizer = optim.Adam(model.parameters(), args.lr)
		if (args.train_more == True):
			checkpoint = torch.load("model/{}_{}_arc.pt".format(args.source, args.normalization))
			model.load_state_dict(checkpoint)
	elif (args.mode == "enhancement"):
		if (args.normalization == "weight"):
			model = ARC_weightNorm(sources = args.source).to(device)
			model2 = Enhancement_weightNorm().to(device)
		elif (args.normalization == "batch"):
			model = ARC_batchNorm(sources = args.source).to(device)
			model2 = Enhancement_batchNorm().to(device)
		optimizer = optim.Adam(model2.parameters(), args.lr)
		if (args.train_more == True):
			checkpoint = torch.load("model/{}_{}_{}.pt".format(args.source, args.normalization, args.en_type))
			model2.load_state_dict(checkpoint)

	loss = nn.MSELoss()

	counter = 0
	#l = 0
	l_pre = 1000
	
	for i in range(args.epochs):
		if (args.mode == "arc"):
			model.train()
		elif (args.mode == "enhancement"):
			model.eval()
			model2.train()
		train_epoch(args, i, model, train_loader, optimizer, loss, device, model2, en_type)
		
		if (args.mode == "arc"):
			model.eval()
		elif (args.mode == "enhancement"):
			model2.eval()
		l = validate_epoch(args, i, model, validate_loader, loss, device, model2, en_type)
	
		if (np.abs(l - l_pre) <= 0.00001):
			counter += 1
		if (l < l_pre):
			if (args.mode == "arc"):
				torch.save(model.state_dict(), "model/{}_{}_arc.pt".format(args.source, args.normalization))
			elif (args.mode == "enhancement"):
				torch.save(model2.state_dict(), "model/{}_{}_{}.pt".format(args.source, args.normalization, args.en_type))
			l_pre = l
		if (counter >= args.early_stopping):
			break


def train_epoch(args, epoch, model, dataset_loader, optimizer, loss, device, model2 = None, en_type = None):
	loss_sum = []
	loss_avg = 0
	
	if (args.mode == "arc"):
		model.train()
	elif (args.mode == "enhancement"):
		model.eval()
		model2.train()
	
	for i, (features, labels) in enumerate(tqdm.tqdm(dataset_loader)):
		features = features.to(device, dtype=torch.float)
		labels = labels.to(device, dtype=torch.float)
		
		if (args.mode == "arc"):			# train ARC
			optimizer.zero_grad()
			out = model(features)
			l = loss(out, labels)
			l.backward()
			optimizer.step()
		elif (args.mode == "enhancement"):	# train Enhancement
			optimizer.zero_grad()
			with torch.no_grad():    
				out_arc = model(features)
			if (args.source == 4):    
				if (en_type == "vocal"):
					out_en = model2(out_arc[:,:1025,:])
					l = loss(out_en, labels[:,:1025,:])
				elif (en_type == "drums"):
					out_en = model2(out_arc[:,1025:2050,:])
					l = loss(out_en, labels[:,1025:2050,:])
				elif (en_type == "bass"):
					out_en = model2(out_arc[:,2050:3075,:])
					l = loss(out_en, labels[:,2050:3075,:])
				elif (en_type == "other"):
					out_en = model2(out_arc[:,3075:,:])
					l = loss(out_en, labels[:,3075:,:])
			elif (args.source == 2):
				if (en_type == "vocal"):
					out_en = model2(out_arc[:,:1025,:])
					l = loss(out_en, labels[:,:1025,:])
				elif (en_type == "other"): 					# accompaniment
					out_en = model2(out_arc[:,1025:,:])
					l = loss(out_en, labels[:,1025:,:])
		
			l.backward()
			optimizer.step()
		
		loss_sum.append(l.item())

		if args.type == "exp":
			wandb.log({"train loss":l.item()})
			wandb.log({"train loss(avg)":np.mean(loss_sum)})

	loss_avg = np.mean(loss_sum)
	print("epoch {} | {} loss: {}".format(epoch, args.mode, loss_avg))


def validate_epoch(args, epoch, model, dataset_loader, loss, device, model2 = None, en_type = None):
	# TODO: parallelize
	loss_sum = 0
	loss_avg = 0
	counter = 0
	hop = 256
	
	model.eval()
	if (model2 is not None):
		model2.eval()

	l = 0		
	with torch.no_grad():
		for i, (features, labels) in enumerate(tqdm.tqdm(dataset_loader)):
			features = features.to(device, dtype=torch.float)
			labels = labels.to(device, dtype=torch.float)
			# Predict based on the original seting of input size
			for ii in range(0, features.shape[2], hop):
				if (ii + hop >= features.shape[2]):
					out_arc = model(features[:, :, -hop:])
					label = labels[:, :, -hop:]
				else:
					out_arc = model(features[:, :, ii:ii + hop])
					label = labels[:, :, ii:ii + hop]
					
				if (args.mode == "arc"):
					l = loss(out_arc, label)
					
				elif (args.mode == "enhancement"):
					if (args.source == 4):
						if (en_type == "vocal"):
							out_en = model2(out_arc[:,:1025,:])
							l = loss(out_en, labels[:,:1025,:])
						elif (en_type == "drums"):
							out_en = model2(out_arc[:,1025:2050,:])
							l = loss(out_en, labels[:,1025:2050,:])
						elif (en_type == "bass"):
							out_en = model2(out_arc[:,2050:3075,:])
							l = loss(out_en, labels[:,2050:3075,:])
						elif (en_type == "other"):
							out_en = model2(out_arc[:,3075:,:])
							l = loss(out_en, labels[:,3075:,:])
					elif (args.source == 2):
						if (en_type == "vocal"):
							out_en = model2(out_arc[:,:1025,:])
							l = loss(out_en, label[:,:1025,:])
						elif (en_type == "other"):
							out_en = model2(out_arc[:,1025:,:])
							l = loss(out_en, label[:,1025:,:])
	
				loss_sum += l.item()
				counter += 1
			
	loss_avg = loss_sum/counter
	print("epoch {} | validation loss: {}".format(epoch, loss_avg))
	if (args.type == "exp"):
		wandb.log({"val loss":loss_avg})
		
	return loss_avg