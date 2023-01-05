import os
import pandas as pd
from tqdm import tqdm
DATA_DIR = 'source_wise_splits/created'
folders = os.listdir(DATA_DIR)

print(folders)
data_source = []
data_target = []
for folder in tqdm(folders):
	try:
		data_path = DATA_DIR+"/"+folder+"/en-hi/hi_sents.tsv"
		data = pd.read_csv(data_path, sep="\t")

	except:
		print(data_path, "not found")
		continue
	data_source += list(data['src'])
	data_target += list(data['tgt'])
	assert len(data_source) == len(data_target)

DATA_DIR = 'source_wise_splits/existing'
folders = os.listdir(DATA_DIR)

print(folders)
for folder in tqdm(folders):
	try:
		data_path = DATA_DIR+"/"+folder+"/en-hi/hi_sents.tsv"
		data = pd.read_csv(data_path, sep="\t")

	except:
		print(data_path, "not found")
		continue
	data_source += list(data['src'])
	data_target += list(data['tgt'])
	assert len(data_source) == len(data_target)

with open('data/samanantar/train.en', 'w') as f:
    for item in data_source:
        f.write("%s\n" % item)

with open('data/samanantar/train.hi', 'w') as f:
    for item in data_target:
        f.write("%s\n" % item)