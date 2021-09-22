import musdb
import museval
import argparse
from test import *
import pdb
import statistics
from model import *
SDR = []
SIR = []
ISR = []
SAR = []
def str2bool(v):
    if v.lower() in ('yes','true','t','y','1'):
        return True
    elif v.lower() in ('no','false','f','n','0'):
        return False
    else:
        return argparse.ArgumentTypeError('Boolean value expected')

def parse_args():
    parser = argparse.ArgumentParser(description='DLP final')
    
    parser.add_argument('-source', type=int, default=2, help='number of voice to separate')
    parser.add_argument("-ARCpath",default="model/arc_2.pt", type=str)
    parser.add_argument("-vocalpath",default="model/vocal_2.pt", type=str)
    parser.add_argument("-drumspath",default="model/drums_4.pt", type=str)
    parser.add_argument("-basspath",default="model/bass_4.pt", type=str)
    parser.add_argument("-otherpath",default="model/other_2.pt", type=str)
    parser.add_argument("-outputdir",default='output/batch/2', type=str, help = 'path to json')
    parser.add_argument("-norm", default='weight', type=str, help = 'weight or batch normalize')
    parser.add_argument("-mode", default='arc', type=str, help = 'arc or enhancement')
    args = parser.parse_args()

    return args


def estimate_and_evaluate(track, estimates, output):
    
    # Evaluate using museval
    scores = museval.eval_mus_track(
        track, estimates, output_dir=output
    )

    # print nicely formatted and aggregated scores
    print(scores)
    #pdb.set_trace()

def SourceSeparation(args, song_path, sr):
    if args.norm == 'weight':
      model = ARC_weightNorm(sources = args.source)
      model2 = Enhancement_weightNorm()
    elif args.norm == 'batch':
      model = ARC_batchNorm(sources = args.source) 
      model2 = Enhancement_batchNorm()   
           
    if args.source == 4:
        arc = model
        arc.cuda()
        arc.load_state_dict(torch.load(args.ARCpath))
        arc.eval()

        en_vocal = model2
        en_vocal.cuda()
        en_vocal.load_state_dict(torch.load(args.vocalpath))
        en_vocal.eval()

        en_drums = model2
        en_drums.cuda()
        en_drums.load_state_dict(torch.load(args.drumspath))
        en_drums.eval()

        en_bass = model2
        en_bass.cuda()
        en_bass.load_state_dict(torch.load(args.basspath))
        en_bass.eval()

        en_other = model2
        en_other.cuda()
        en_other.load_state_dict(torch.load(args.otherpath))
        en_other.eval()

        x, sr = librosa.load(song_path, sr = sr)

        stft, y_vocal, y_drums, y_bass, y_other = predict_song_4(args, x, arc, en_vocal, en_drums, en_bass, en_other)
    
        return stft, y_vocal, y_drums, y_bass, y_other
    else:
        arc = model
        arc.cuda()
        arc.load_state_dict(torch.load(args.ARCpath))
        arc.eval()

        en_voice = model2
        en_voice.cuda()
        en_voice.load_state_dict(torch.load(args.vocalpath))
        en_voice.eval()

        en_others = model2
        en_others.cuda()
        en_others.load_state_dict(torch.load(args.otherpath))
        en_others.eval()
    
        x, sr = librosa.load(song_path, sr = sr)
    
        stft, y_voice, y_others = predict_song_2(args, x, arc, en_voice, en_others)
    
        return stft, y_voice, y_others



if __name__=='__main__':
    args = parse_args()
    args.ARCpath = 'model/{}_{}_arc.pt'.format(args.source,args.norm)
    args.vocalpath = 'model/{}_{}_vocal.pt'.format(args.source,args.norm)
    args.drumspath = 'model/{}_{}_drums.pt'.format(args.source,args.norm)
    args.basspath = 'model/{}_{}_bass.pt'.format(args.source,args.norm)
    args.otherpath = 'model/{}_{}_other.pt'.format(args.source,args.norm)
    print(args)

    mus = musdb.DB(root='dataset/wav/', is_wav=True, subsets='test')
    for track in mus:
        songpath = 'dataset/wav/test/{}/mixture.wav'.format(track.name) 
        if args.source ==4:
            stft, y_vocal, y_drums, y_bass, y_other = SourceSeparation(args, songpath , sr=44100) 
            estimates = {
                'vocals': np.stack((y_vocal,y_vocal), -1),#np.expand_dims(y_vocal, axis=1),
                'drums': np.stack((y_drums,y_drums), -1),#np.expand_dims(y_drums, axis=1),
                'bass': np.stack((y_bass,y_bass), -1),#np.expand_dims(y_bass, axis=1),
                'other': np.stack((y_other,y_other), -1)#np.expand_dims(y_other, axis=1)
            }
        else:    
            stft, y_vocal, y_other = SourceSeparation(args, songpath , sr=44100) 
            estimates = {
                'vocals': np.stack((y_vocal,y_vocal), -1),#np.expand_dims(y_vocal, axis=1),
                'accompaniment': np.stack((y_other,y_other), -1)#np.expand_dims(y_other, axis=1)
            }

        #pdb.set_trace()
        estimate_and_evaluate(track, estimates, args.outputdir)