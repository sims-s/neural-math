from urllib.request import urlretrieve 
import os
from tqdm.auto import tqdm
from pyunpack import Archive
import numpy as np
from shutil import rmtree

url_target = 'http://www.primos.mat.br/dados/2T_part%d.7z'
n_parts = 200
n_primes = 2_000_000_000
primes_per_row = 10
primes = np.ones(n_primes, dtype='int64')
prime_counter = 0
prime_save_path = 'data/primes_2B.npy'

os.makedirs('data/interim/', exist_ok=True)

def cleanup(fnames):
    for n in fnames:
        os.remove(n)

def parse_file(fname):
    global prime_counter
    with open(fname, 'r') as f:
        pbar = tqdm(total = n_primes//n_parts//primes_per_row)
        for line in f:
            line = line.split('\t')
            line = np.array([int(e.strip()) for e in line], dtype='int64')
            primes[prime_counter:prime_counter+len(line)] = line
            prime_counter += len(line)
            pbar.update(1)
    pbar.close()



for i in tqdm(range(1, n_parts+1), total=n_parts):
    zip_save_fname = 'data/interim/part_%d.7z'%i
    txt_file_name = 'data/interim/2T_part%d.txt'%i

    urlretrieve(url_target%i, zip_save_fname)
    Archive(zip_save_fname).extractall(txt_file_name[:txt_file_name.rfind('/')])
    parse_file(txt_file_name)
    cleanup([zip_save_fname, txt_file_name])

rmtree('data/interim')


np.save(prime_save_path, primes)