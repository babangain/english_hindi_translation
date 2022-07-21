# English Hindi Translation
## Clone and Navigate to this repo
```
git clone https://github.com/babangain/english_hindi_translation
cd english_hindi_translation
```
## Install Dependencies
```
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
```

DATA_FOLDER_NAME=samanantar
DATA_DIR=data/$DATA_FOLDER_NAME
cat $DATA_DIR/train.en | $MOSES_DIR/scripts/tokenizer/lowercase.perl> $DATA_DIR/train.lc.en
cat $DATA_DIR/train.hi | $MOSES_DIR/scripts/tokenizer/lowercase.perl> $DATA_DIR/train.lc.hi
cat $DATA_DIR/qna.en | $MOSES_DIR/scripts/tokenizer/lowercase.perl> qna.lc.en

```
## Learn Byte-pair Encoding (optional)
This can take long. Instead use the following command to use the bpecode provided by us
```
cp bpecode $DATA_DIR/bpecode
```
Otherwise, to learn bpecode, run the following
```

cat $DATA_DIR/train.lc.en $DATA_DIR/train.lc.hi qna.lc.en > all.lc
$FASTBPE_DIR/fast learnbpe 50000 all.lc > $DATA_DIR/bpecode
$FASTBPE_DIR/fast applybpe $DATA_DIR/train.bpe.en $DATA_DIR/train.lc.en $DATA_DIR/bpecode
$FASTBPE_DIR/fast applybpe $DATA_DIR/train.bpe.hi $DATA_DIR/train.lc.hi $DATA_DIR/bpecode
$FASTBPE_DIR/fast getvocab $DATA_DIR/train.bpe.en $DATA_DIR/train.bpe.hi > $DATA_DIR/vocab.en
```

## Binarize the data for faster training
```
BINARY_DATA_DIR=data_bin/$DATA_FOLDER_NAME
mkdir -p $BINARY_DATA_DIR
fairseq-preprocess \
    --source-lang en --target-lang hi \
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
    --source-lang en --target-lang hi \
    --arch transformer --log-interval  1  --log-format simple \
    --dropout 0.2 --weight-decay 0.0 \
    --share-all-embeddings \
    --ddp-backend=no_c10d \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 5000 \
    --max-tokens 4000 --update-freq 64  \
    --max-epoch 30 \
    --save-interval 10\
    --save-dir $MODEL_DIR &
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
