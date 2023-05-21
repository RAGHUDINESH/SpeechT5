DATA_ROOT="/home/raghuIITM/DDP_NER/ddp_slue_dataset/manifest/slue-voxpopuli/e2e_ner/fairseq-preprocess"
SAVE_DIR="/home/raghuIITM/DDP_NER/checkpoints/transformer_lm/v2"
LABEL_DIR="/home/raghuIITM/DDP_NER/ddp_slue_dataset/hubert_labels"
BPE_TOKENIZER="/home/raghuIITM/DDP_NER/ddp/E2E/SpeechT5/SpeechT5/data/spm_char.model"
USER_DIR="/home/raghuIITM/DDP_NER/ddp/E2E/SpeechT5/SpeechT5/speecht5"
PT_CHECKPOINT_PATH="/home/raghuIITM/DDP_NER/checkpoints/SpeechT5/t5_LM2.pt"
USER_DIR="/home/raghuIITM/DDP_NER/ddp/E2E/SpeechT5/SpeechT5/speecht5"

mkdir -p ${SAVE_DIR}
fairseq-train ${DATA_ROOT} \
    --task language_modeling \
    --save-dir ${SAVE_DIR} \
    --user-dir ${USER_DIR} \
    --arch transformer_lm_t5 \
    --finetune-from-model ${PT_CHECKPOINT_PATH} \
    --share-decoder-input-output-embed \
    --dropout 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
    --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --tokens-per-sample 512 --sample-break-mode none \
    --max-tokens 2048 --update-freq 2 \
    --fp16 \
    --max-update 50000 \