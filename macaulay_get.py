import requests
import os
import time
import csv
import re

#https://macaulaylibrary.org/asset/NUMBER is the webpage with the sound player for a bird
#https://cdn.download.ams.birds.cornell.edu/api/v1/asset/NUMBER/audio is the mp3 file corresponding to it

cur_lim = 5
cur_sleep = 10
num = 250500261
curdir = os.path.split(__file__)[0]
datafolder = os.path.join(curdir, 'macaulay', 'csv')
file_outfolder = os.path.join(curdir, 'macaulay_out')

#outf = os.path.join(curdir, 'test')
wantfiles = [x for x in os.listdir(datafolder) if os.path.isdir(os.path.join(datafolder, x)) == False]

def make_folder(fpath):
    if os.path.exists(fpath) == False:
        os.makedirs(fpath)

def fix_name(cur_name):
    ret_name = re.sub(r'\s', '_', cur_name.strip(), flags=re.IGNORECASE)
    ret_name2 = re.sub(r'\[', '-', ret_name, flags=re.IGNORECASE)
    ret_name3 = re.sub(r'\]', '-', ret_name2, flags=re.IGNORECASE)
    return ret_name3

def get_mp3(cnum, cur_outf):
    want_url = f"https://cdn.download.ams.birds.cornell.edu/api/v1/asset/{cnum}/audio"
    resp = requests.get(want_url)
    if resp.ok == True:
        status_str = f"Downloading {cnum} into {cur_outf}"
        print(status_str)
        make_folder(cur_outf)
        save_name = f"ML{cnum}.mp3"
        save_path = os.path.join(cur_outf, save_name)
        with open(save_path, "wb") as writeout:
            writeout.write(resp.content)

def csv_iter(sleep_time = 10, file_lim=float('inf')):
    file_count = 0
    make_folder(file_outfolder)
    am_done = False
    for x in wantfiles:
        if am_done == True:
            break
        curpath = os.path.join(datafolder, x)
        with open(curpath) as csvf:
            reader = csv.DictReader(csvf)
            for row in reader:
                #print(row.keys())
                cur_cname = fix_name(row['Common Name'])
                cur_sname = fix_name(row['Scientific Name'])
                cur_catnum = row['\ufeffML Catalog Number'].strip()
                outf = os.path.join(file_outfolder, cur_sname)
                get_mp3(cur_catnum, outf)
                time.sleep(sleep_time)
                file_count += 1
                if file_count >= file_lim:
                    am_done = True
                    break
                #print(cur_cname, cur_sname, cur_catnum)
                
csv_iter(sleep_time=cur_sleep,file_lim=cur_lim)
