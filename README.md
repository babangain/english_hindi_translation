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
    --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 5000 --disable-validation --valid-subset train \
    --max-tokens 4000 --update-freq 64  \
    --max-epoch 30 \
    --save-interval 10\
    --save-dir $MODEL_DIR &
```
# References
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
```
@inproceedings{sennrich-etal-2016-neural,
    title = "Neural Machine Translation of Rare Words with Subword Units",
    author = "Sennrich, Rico  and
      Haddow, Barry  and
      Birch, Alexandra",
    booktitle = "Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2016",
    address = "Berlin, Germany",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P16-1162",
    doi = "10.18653/v1/P16-1162",
    pages = "1715--1725",
}
```
```
@inproceedings{koehn-etal-2007-moses,
    title = "{M}oses: Open Source Toolkit for Statistical Machine Translation",
    author = "Koehn, Philipp  and
      Hoang, Hieu  and
      Birch, Alexandra  and
      Callison-Burch, Chris  and
      Federico, Marcello  and
      Bertoldi, Nicola  and
      Cowan, Brooke  and
      Shen, Wade  and
      Moran, Christine  and
      Zens, Richard  and
      Dyer, Chris  and
      Bojar, Ond{\v{r}}ej  and
      Constantin, Alexandra  and
      Herbst, Evan",
    booktitle = "Proceedings of the 45th Annual Meeting of the Association for Computational Linguistics Companion Volume Proceedings of the Demo and Poster Sessions",
    month = jun,
    year = "2007",
    address = "Prague, Czech Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P07-2045",
    pages = "177--180",
}
```
```
@inproceedings{ott-etal-2019-fairseq,
    title = "fairseq: A Fast, Extensible Toolkit for Sequence Modeling",
    author = "Ott, Myle  and
      Edunov, Sergey  and
      Baevski, Alexei  and
      Fan, Angela  and
      Gross, Sam  and
      Ng, Nathan  and
      Grangier, David  and
      Auli, Michael",
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics (Demonstrations)",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/N19-4009",
    doi = "10.18653/v1/N19-4009",
    pages = "48--53",
    abstract = "fairseq is an open-source sequence modeling toolkit that allows researchers and developers to train custom models for translation, summarization, language modeling, and other text generation tasks. The toolkit is based on PyTorch and supports distributed training across multiple GPUs and machines. We also support fast mixed-precision training and inference on modern GPUs. A demo video can be found at https://www.youtube.com/watch?v=OtgDdWtHvto",
}
```
