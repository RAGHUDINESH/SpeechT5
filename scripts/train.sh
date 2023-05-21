DATA_ROOT="/home/raghuIITM/DDP_NER/ddp_slue_dataset/manifest/slue-voxpopuli"
SAVE_DIR="/home/raghuIITM/DDP_NER/fairseq_speecht5/v2"
TRAIN_SET="fine-tune"
VALID_SET="dev"
LABEL_DIR="/home/raghuIITM/DDP_NER/ddp_slue_dataset/hubert_labels"
BPE_TOKENIZER="/home/raghuIITM/DDP_NER/ddp/E2E/SpeechT5/SpeechT5/data/spm_char.model"
USER_DIR="/home/raghuIITM/DDP_NER/ddp/E2E/SpeechT5/SpeechT5/speecht5"
PT_CHECKPOINT_PATH="/home/raghuIITM/DDP_NER/fairseq_speecht5/v1/checkpoint_best.pt"

mkdir -p ${SAVE_DIR}
fairseq-train ${DATA_ROOT} \
  --save-dir ${SAVE_DIR} \
  --tensorboard-logdir ${SAVE_DIR} \
  --train-subset ${TRAIN_SET} \
  --valid-subset ${VALID_SET} \
  --hubert-label-dir ${LABEL_DIR} \
  --distributed-world-size 4 \
  --distributed-port 0 \
  --ddp-backend legacy_ddp \
  --user-dir ${USER_DIR} \
  --log-format json \
  --seed 1 \
  \
  --task speecht5 \
  --t5-task s2t \
  --sample-rate 16000 \
  --num-workers 0 \
  --max-tokens 1600000 \
  --update-freq 2 \
  --bpe-tokenizer ${BPE_TOKENIZER} \
  \
  --criterion speecht5 \
  --report-accuracy \
  --zero-infinity \
  --ce-weight 0.5 \
  --ctc-weight 0.5 \
  --sentence-avg \
  \
  --optimizer adam \
  --adam-betas "(0.9, 0.98)" \
  --adam-eps 1e-08 \
  --weight-decay 0.1 \
  --clip-norm 25.0 \
  --lr 0.00006 \
  --lr-scheduler tri_stage \
  --phase-ratio "[0.1, 0.4, 0.5]" \
  --final-lr-scale 0.05 \
  \
  --max-update 80000 \
  --max-text-positions 600 \
  --required-batch-size-multiple 1 \
  --save-interval-updates 3000 \
  --skip-invalid-size-inputs-valid-test \
  \
  --arch t5_transformer_base_asr \
  --share-input-output-embed \
  --find-unused-parameters \
  --bert-init \
  --relative-position-embedding \
  --freeze-encoder-updates 13000 \
  \
  --keep-last-epochs 5 \
  --feature-grad-mult 1.0 \
  --best-checkpoint-metric s2t_accuracy \
  --maximize-best-checkpoint-metric \
  --finetune-from-model ${PT_CHECKPOINT_PATH}


# --fp16 \