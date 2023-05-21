# $subset=dev_other
# python examples/speech_recognition/infer.py /checkpoint/abaevski/data/speech/libri/10h/wav2vec/raw --task audio_finetuning \
# --nbest 1 --path /path/to/model --gen-subset $subset --results-path /path/to/save/results/for/sclite --w2l-decoder kenlm \
# --lm-model /path/to/kenlm.bin --lm-weight 2 --word-score -1 --sil-weight 0 --criterion ctc --labels ltr --max-tokens 4000000 \
# --post-process letter

DIR_FOR_PREPROCESSED_DATA="/home/raghuIITM/DDP_NER/ddp_slue_dataset/manifest/slue-voxpopuli"
CHECKPOINT_PATH="/home/raghuIITM/DDP_NER/fairseq_speecht5/v2/checkpoint_best.pt"
SUBSET="dev"
KENLM_MODEL_PATH="/home/raghuIITM/DDP_NER/kenlm/4gram.bin"
LEXICON_PATH="/home/raghuIITM/DDP_NER/kenlm/lexicon.lst"
RES_DIR="/home/raghuIITM/DDP_NER/ddp/E2E/SpeechT5/SpeechT5/results/v12"
USER_DIR="/home/raghuIITM/DDP_NER/ddp/E2E/SpeechT5/SpeechT5/speecht5"


python examples/speech_recognition/infer.py $DIR_FOR_PREPROCESSED_DATA \
    --task speecht5 \
    --seed 1 \
    --nbest 1 \
    --path $CHECKPOINT_PATH \
    --gen-subset $SUBSET \
    --results-path $RES_DIR \
    --w2l-decoder kenlm \
    --kenlm-model $KENLM_MODEL_PATH \
    --lexicon $LEXICON_PATH \
    --beam 200 \
    --beam-threshold 15 \
    --lm-weight 1.5 \
    --word-score 1.5 \
    --sil-weight -0.3 \
    --criterion asg_loss \
    --max-replabel 2 \
    --user-dir $USER_DIR
