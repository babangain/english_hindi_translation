
## Pre-process the data
```
cd data_check/public_data/en-hi
python sep.py
python rem_lang.py
cd ../../combined
cat ../public_data/en-hi/train.clean.{en,hi} ../{reviews,PD}/en-hi/{src-train.txt,tgt-train.txt} > all.txt
cat ../public_data/en-hi/train.clean.en ../{reviews,PD}/en-hi/src-train.txt > all.en
cat ../public_data/en-hi/train.clean.hi ../{reviews,PD}/en-hi/tgt-train.txt > all.hi
```
## Learn Byte-pair Encoding (optional)
This can take long. Instead use the following command to use the bpecode provided by us
```
cp en_hi/bpecode $DATA_DIR/bpecode_fk
cp en_hi/vocab.en $DATA_DIR/vocab_fk.en
```
Otherwise, to learn bpecode, run the following
```
DATA_DIR=.
$FASTBPE_DIR/fast learnbpe 20000 all.txt > $DATA_DIR/bpecode_fk
$FASTBPE_DIR/fast applybpe $DATA_DIR/all.bpe.en $DATA_DIR/all.en $DATA_DIR/bpecode_fk
$FASTBPE_DIR/fast applybpe $DATA_DIR/all.bpe.hi $DATA_DIR/all.hi $DATA_DIR/bpecode_fk
$FASTBPE_DIR/fast getvocab $DATA_DIR/all.bpe.en $DATA_DIR/all.bpe.hi > $DATA_DIR/vocab_fk.en
cd ../..
```
### Pre-process other parts
DATA_FOLDER_NAME=combined_fk
DATA_DIR=data/$DATA_FOLDER_NAME
for SUBSET in test valid
do
  for LANG in en hi
  do
    $FASTBPE_DIR/fast applybpe $DATA_DIR/$SUBSET.bpe.$LANG $DATA_DIR/$SUBSET.$LANG $DATA_DIR/bpecode_fk
  done
done

## Binarize the data for faster training
```
BINARY_DATA_DIR=data_bin/$DATA_FOLDER_NAME
mkdir -p $BINARY_DATA_DIR
fairseq-preprocess \
    --source-lang en --target-lang hi \
    --joined-dictionary \
    --srcdict $DATA_DIR/vocab_fk.en \
    --trainpref $DATA_DIR/all.bpe \
    --validpref $DATA_DIR/valid.bpe \
    --testpref $DATA_DIR/test.bpe \
    --destdir $BINARY_DATA_DIR \
    --workers 20
```

## Training
```
MODEL_DIR=models/$DATA_FOLDER_NAME
mkdir -p $MODEL_DIR
export CUDA_VISIBLE_DEVICES=5
nohup fairseq-train --fp16 \
    $BINARY_DATA_DIR \
    --source-lang en --target-lang hi \
    --arch transformer --log-interval  1  --log-format simple \
    --dropout 0.2 --weight-decay 0.0 \
    --share-all-embeddings \
    --ddp-backend=no_c10d \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --seed 42 \
    --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07  \
    --max-tokens 4000 --update-freq 6  \
    --max-update 100000 \
    --save-interval 5 --save-interval-updates  1000 --keep-interval-updates 20 --skip-invalid-size-inputs-valid-test \
    --save-dir $MODEL_DIR > $DATA_DIR/train_log.log &
```
## Generate 
```
BINARY_DATA_DIR=~/scripts/chat_en_hi/data/data_bin/data/wmt20_chat
OUTFILENAME=wmt20_baseline
fairseq-generate $BINARY_DATA_DIR --batch-size 32 --path models/samanantar/checkpoint_last.pt  --remove-bpe \
--beam 5 --source-lang en --target-lang hi --task translation >  $OUTFILENAME.txt

cat $OUTFILENAME.txt |grep ^H | sort -nr -k1.2 | cut -f3- | $MOSES_DIR/scripts/tokenizer/detokenizer.perl > $OUTFILENAME.hi 

cat $OUTFILENAME.hi | sacrebleu ~/scripts/chat_en_hi/data/data/wmt20_chat/test.hi  -m bleu ter

