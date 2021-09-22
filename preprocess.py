import numpy as np
import glob 
from librosa import stft 
import pdb 
import os
import scipy
import soundfile as sf


path = glob.glob1("dataset/wav/train/","*") 
type = ['accompaniment', 'bass', 'drums', 'mixture', 'other', 'vocals']
count = 0
for f in path:
	random = []
	ran = False
	for j in type:
		data, sr = sf.read("dataset/wav/train/{}/{}.wav".format(f,j))

		# split
		split = []
		noSections = int(np.ceil(len(data) / sr))
		if (ran == False):
			for i in range(15):
				random.append(np.random.randint(0,noSections-6))
				ran = True 
		
		for i in random:
		    temp = data[i*sr:(i+5)*sr , :] # this is for stereo audio; uncomment and comment line above
		    # stereo to mono, therefore can apply to librosa
		    temp2 = (temp.T[0]+temp.T[1])/2
		    # add to list
		    split.append(temp2)
		
		
		for i in range(len(random)):
		    X_libs = stft(split[i], n_fft=2048, hop_length=1024)
		    dir = "train/"
		    filename = dir+'{}/{}.npz'.format(f,j)
		    # write to file
		    if not os.path.exists(dir+'{}'.format(j)):
		    	os.mkdir(dir+'{}'.format(j))
		    #if not os.path.exists(dir+'{}/{}'.format(f,j)):
		    #	os.mkdir(dir+'{}/{}'.format(f,j))
		    
		    #sf.write(dir+'{}/{}/{}.npy'.format(f,j,i), X_libs, sr)
		    with open(dir+"{}/{}.npy".format(j,i+15*count), 'wb') as w:
		    	np.save(w, X_libs)
	print(count)
	count += 1