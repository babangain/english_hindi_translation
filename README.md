# English Hindi Translation

## Downloading data for pre-training
```
pip install pandas tqdm fairseq
wget https://storage.googleapis.com/samanantar-public/V0.3/source_wise_splits.zip
unzip source_wise_splits.zip
mkdir -p data/samanantar
python combine_data_v2.py
```

## pre-process the data
```
git clone https://github.com/moses-smt/mosesdecoder.git
MOSES_DIR=mosesdecoder
DATA_FOLDER_NAME=samanantar
DATA_DIR=data/$DATA_FOLDER_NAME
cat $DATA_DIR/train.en | $MOSES_DIR/scripts/tokenizer/lowercase.perl> $DATA_DIR/train.lc.en
cat $DATA_DIR/train.hi | $MOSES_DIR/scripts/tokenizer/lowercase.perl> $DATA_DIR/train.lc.hi

```
If you use the dataset, cite the paper that introduced it
```
@misc{ramesh2021samanantar,
      title={Samanantar: The Largest Publicly Available Parallel Corpora Collection for 11 Indic Languages}, 
      author={Gowtham Ramesh and Sumanth Doddapaneni and Aravinth Bheemaraj and Mayank Jobanputra and Raghavan AK and Ajitesh Sharma and Sujit Sahoo and Harshita Diddee and Mahalakshmi J and Divyanshu Kakwani and Navneet Kumar and Aswin Pradeep and Srihari Nagaraj and Kumar Deepak and Vivek Raghavan and Anoop Kunchukuttan and Pratyush Kumar and Mitesh Shantadevi Khapra},
      year={2021},
      eprint={2104.05596},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
