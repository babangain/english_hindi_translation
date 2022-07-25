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
mkdir -p $DATA_DIR
cp data/$OLD_DATA_FOLDER_NAME/train.en $DATA_DIR/train.en
cp data/$OLD_DATA_FOLDER_NAME/train.hi $DATA_DIR/train.hi
```

## Learn Byte-pair Encoding (optional)
This can take long. Instead use the following command to use the bpecode_hi_en provided by us
```
cp hi_en/bpecode $DATA_DIR/bpecode
cp hi_en/vocab.en $DATA_DIR/vocab.en
```
Otherwise, to learn bpecode, run the following
```
cat $DATA_DIR/train.en $DATA_DIR/train.hi > $DATA_DIR/all.en_hi
$FASTBPE_DIR/fast learnbpe 50000 $DATA_DIR/all.en_hi > $DATA_DIR/bpecode
$FASTBPE_DIR/fast applybpe $DATA_DIR/train.bpe.en $DATA_DIR/train.en $DATA_DIR/bpecode
$FASTBPE_DIR/fast applybpe $DATA_DIR/train.bpe.hi $DATA_DIR/train.hi $DATA_DIR/bpecode
$FASTBPE_DIR/fast getvocab $DATA_DIR/train.bpe.en $DATA_DIR/train.bpe.hi > $DATA_DIR/vocab.en
```
## Binarize the data for faster training
```
BINARY_DATA_DIR=data_bin/$DATA_FOLDER_NAME
mkdir -p $BINARY_DATA_DIR
fairseq-preprocess \
    --source-lang hi --target-lang en \
    --joined-dictionary \
    --srcdict $DATA_DIR/vocab.en \
    --trainpref $DATA_DIR/train.bpe \
    --destdir $BINARY_DATA_DIR \
    --workers 20
```

## Training
```
MODEL_DIR=models/$DATA_FOLDER_NAME
mkdir -p $MODEL_DIR
export CUDA_VISIBLE_DEVICES=5,6
nohup fairseq-train --fp16 \
    $BINARY_DATA_DIR \
    --source-lang hi --target-lang en \
    --arch transformer --log-interval  1  --log-format simple \
    --dropout 0.2 --weight-decay 0.0 \
    --share-all-embeddings \
    --ddp-backend=no_c10d \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --seed 42 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 5000 --disable-validation --valid-subset train \
    --max-tokens 4000 --update-freq 64  \
    --max-epoch 30 \
    --save-interval 10\
    --save-dir $MODEL_DIR &
```
