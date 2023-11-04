import torch
import os
import sndload as SL
from torch.utils.data import Dataset


class BirdSet(Dataset):
    def __init__(self, datafolder="birdsplit", max_ms = 1000, srate = 44100, basefolder = os.path.split(__file__)[0]):
        curdir = os.path.split(__file__)[0]
        curpath = os.path.join(basefolder, datafolder)
        folders = [x for x in os.listdir(curpath) if os.path.isdir(os.path.join(curpath, x)) == True]
        sndpaths = []
        sndlabels = []
        for folder in folders:

            sndfolder = os.path.join(curpath, folder)
            cur_snds = [os.path.join(sndfolder,x) for x in os.listdir(sndfolder) if ".mp3" in x]
            cur_len = len(cur_snds)
            cur_labels = [folder] * cur_len
            sndpaths += cur_snds
            sndlabels += cur_labels

        max_samp = max_ms * 0.001 * srate
        self.paths = sndpaths
        self.labels = sndlabels
        self.max_ms = max_ms
        self.srate = srate
        self.max_samp = max_samp

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # return sound, label
        curpath = self.paths[idx]
        curlabel = self.labels[idx]
        retsnd = SL.sndloader(curpath, want_sr=None, want_bits=None, to_mono=True)
        return retsnd, curlabel

