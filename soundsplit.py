import os
import re
import subprocess

def make_folder(fpath):
    if os.path.exists(fpath) == False:
        os.makedirs(fpath)
#wantfolder = ['test folder']
basefolder = 'bird'
outbase = 'birdsplit'
curdir = os.path.split(__file__)[0]
datafolder = os.path.join(curdir, basefolder)
outpath = os.path.join(curdir, outbase)
make_folder(outpath)
#print(os.listdir(datafolder))
wantfolder = [x for x in os.listdir(datafolder) if os.path.isdir(os.path.join(datafolder, x)) == True]


for folder in wantfolder:
    fixfolder = re.sub(r'\s', '_', folder, flags=re.IGNORECASE)
    curpath = os.path.join(datafolder, folder)
    cur_contents = os.listdir(curpath)
    cur_out = os.path.join(outpath,fixfolder)
    print(cur_out)
    make_folder(cur_out)
    outstr = str(cur_out)
    for cur_cont in cur_contents:
        if "mp3" in cur_cont:
            infile = os.path.join(curpath, cur_cont)
            infstr = str(infile)
            cmdarr = ["mp3splt", "-f", "-t", "0.10", "-d", outstr, infstr]
            subprocess.run(cmdarr)
            #print(curcmd)
        

    print(cur_contents)

