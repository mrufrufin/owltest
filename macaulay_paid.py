from bs4 import BeautifulSoup
import urllib.request
import csv

want_site = 'https://www.macaulaylibrary.org/guide-to-bird-sounds/species-list/'
s_outfile = 'macaulay_sci.txt'
foundfile = 'macaulal_sci_found.txt'

birdnames = []
birdfound = {}
with open(s_outfile, 'r') as curf:
    arr = [x.strip() for x in curf.readlines()]
    birdfound = {x:False for x in arr}
    birdnames += arr


def pg_read():
    with urllib.request.urlopen(want_site) as response:
        html_doc = response.read()
        soup = BeautifulSoup(html_doc, 'html.parser')
        links = soup.find_all("a")
        for l in links:
            curtext = l.text.lower().strip()
            for x in birdnames:
                if x in curtext:
                    birdfound[x] = True

pg_read()

with open(foundfile, 'w') as curf:
    curf.write("scientific_name,found\n")
    for k,v in birdfound.items():
        num_to_write = 1 if v == True else 0
        str_write = f"{k},{num_to_write}\n"
        curf.write(str_write)


