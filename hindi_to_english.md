## Install Dependencies 
```
git clone https://github.com/babangain/english_hindi_translation
cd english_hindi_translation
pip install pandas tqdm fairseq
git clone https://github.com/moses-smt/mosesdecoder.git
MOSES_DIR=mosesdecoder
git clone https://github.com/glample/fastBPE.git
g++ -std=c++11 -pthread -O3 fastBPE/fastBPE/main.cc -IfastBPE -o fastBPE/fast
FASTBPE_DIR=fastBPE
```
## Downloading data for pre-training
```
wget https://storage.googleapis.com/samanantar-public/V0.3/source_wise_splits.zip
unzip source_wise_splits.zip
mkdir -p data/samanantar
python combine_data_v2.py
```

## Pre-process the data
Not lowercasing the data as True case need to be generated at target side (as target is English)
```

OLD_DATA_FOLDER_NAME=samanantar
DATA_FOLDER_NAME=samanantar_hi_en
DATA_DIR=data/$DATA_FOLDER_NAME
cp data/$OLD_DATA_FOLDER_NAME/train.en $DATA_DIR/train.en
cp data/$OLD_DATA_FOLDER_NAME/train.hi $DATA_DIR/train.hi
```

## Learn Byte-pair Encoding (optional)
This can take long. Instead use the following command to use the bpecode_hi_en provided by us
```
cp bpecode_hi_en $DATA_DIR/bpecode
co vocab_hi_en.en $DATA_DIR/vocab.en
```
Otherwise, to learn bpecode, run the following
```
cat $DATA_DIR/train.en $DATA_DIR/train.hi > all.en_hi
$FASTBPE_DIR/fast learnbpe all.en_hi > $DATA_DIR/bpecode
$FASTBPE_DIR/fast applybpe $DATA_DIR/train.bpe.en $DATA_DIR/train.en $DATA_DIR/bpecode
$FASTBPE_DIR/fast applybpe $DATA_DIR/train.bpe.hi $DATA_DIR/train.hi $DATA_DIR/bpecode
$FASTBPE_DIR/fast getvocab $DATA_DIR/train.bpe.en $DATA_DIR/train.bpe.hi > $DATA_DIR/vocab.en
```

