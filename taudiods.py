import numpy as np
import torch
import torchaudio as TA
import torchaudio.functional as TAF
import torchaudio.transforms as TAT
import os
import sys

def audioload_helper(want_file:str, fpath:str = "data", want_sr:int = None, want_bits:int = None, max_samp: float = np.inf, to_mono:bool = True) -> any:
    cur_fp = None
    if len(fpath) > 0:
        cur_fp = os.path.join(fpath, want_file)
    else:
        cur_fp = want_file
    read_frames = int(max_samp) if np.isfinite(max_samp) == True else -1
    cur_wf, cur_sr = TA.load(cur_fp, num_frames = read_frames)
    cur_nc, cur_nf = cur_wf.shape
    ret_wf = None
    ret_sr = None

    cur_wf2 = None
    if to_mono == True and cur_nc > 1:
        cur_mult = 1./float(cur_nc)
        cur_wf2 = (cur_wf * cur_mult).sum(dim=0).unsqueeze(0)
    else:
        cur_wf2 = cur_wf
    if want_sr != cur_sr:
        ret_wf = resamp_helper(cur_wf2, cur_sr, want_sr, want_bits=want_bits)
        ret_sr = want_sr
    elif want_bits != None:
        cur_dtype = cur_wf.dtype
        cur_bits = 16
        if cur_dtype == torch.float64:
            cur_bits = 64
        elif cur_dtype == torch.float32:
            cur_bits = 32
        elif cur_dtype == torch.float16:
            cur_bits = 16
        if want_bits != cur_bits:

            want_dtype = None
            if want_bits != None:
                if want_bits == 64:
                    want_dtype = torch.float64
                elif want_bits == 32:
                    want_dtype = torch.float32
                elif want_bits == 16:
                    want_dtype = torch.float16
                else:
                    want_dtype = torch.float16
            ret_wf = cur_wf2.to(want_dtype)
            ret_sr = cur_sr
    else:
        ret_wf = cur_wf2
        ret_sr = cur_sr

    if ret_wf.shape[1] < read_frames:
        ret_wf = torch.nn.functional.pad(ret_wf, (0, read_frames - ret_wf.shape[1]))

    return ret_wf, ret_sr

def resamp_helper(cur_wf:any, cur_sr:int, want_sr:int, want_bits:int = None) -> any:
    cur_dtype = cur_wf.dtype
    resamp_f = TAT.Resample(cur_sr, want_sr)
    want_dtype = None
    resamped = None
    if want_bits != None:
        if want_bits == 64:
            want_dtype = torch.float64
        elif want_bits == 32:
            want_dtype = torch.float32
        elif want_bits == 16:
            want_dtype = torch.float16
    if want_dtype == None:
        resamped = resamp_f(cur_wf)
    else:
        resamped = resamp_f(cur_wf).to(want_dtype)
    return resamped

def chunk_wf(cur_wf:any, frame_size:int, overlap:int = 0):
    cur_nc, cur_samp = cur_wf.shape
    hop_size = frame_size - overlap
    end_nf = max(1,int(float(cur_samp-overlap)/float(hop_size)))
    tot_samp = (end_nf *  hop_size) + overlap # will undercount
    # for example if cur_samp > end_nf * hop_size + overlap but cur_samp < (end_nf + 1) * hop_size
    need_samp = cur_samp - tot_samp #if needs extra samples
    extra_samp = 0
    if need_samp > 0:
        end_nf += 1
        tot_samp = (end_nf *  hop_size) + overlap # now should overshoot if anything
        extra_samp = tot_samp - cur_samp
    want_shape = (end_nf, cur_nc, frame_size)
    ret_wf = torch.zeros(want_shape)
    for i in range(end_nf):
        start_samp = i * hop_size
        end_samp = start_samp + frame_size
        ret_end = frame_size
        if i == (end_nf - 1):
            if end_samp > cur_samp:
                ret_end = cur_samp - start_samp
                end_samp = cur_samp
            
        ret_wf[i,:,:ret_end] = cur_wf[:,start_samp:end_samp]
    return ret_wf



rel_datafolder = "data"
curdir = os.path.split(__file__)[:-1]
test_f = "255814__derekxkwan__bach-marimba.wav"
test_fp = os.path.join(*curdir, rel_datafolder)

want_sr = 16000
want_bits = 16
rsed, rsed_sr = audioload_helper(test_f, test_fp, want_sr=want_sr, want_bits=want_bits)
test_np = rsed.numpy()
test_nc, test_samp = rsed.shape
test_dtype = rsed.dtype
print(f"File: {test_f}\nSample Rate:{rsed_sr}, Channels:{test_nc}, Samples:{test_samp}, dtype: {test_dtype}")
fsize = 1024
overlap = 256
reshaped = chunk_wf(rsed, fsize, overlap = overlap)
cur_nf, cur_nc, cur_fs = reshaped.shape

print("=================")
print(f"Reshaping using Frame Size:{fsize} and Overlap:{overlap}...\nNum Frames:{cur_nf}")


