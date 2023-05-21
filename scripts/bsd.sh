#CHECKPOINT_PATH="/home/raghuIITM/DDP_NER/checkpoints/SpeechT5/base_asr.pt"
#CHECKPOINT_PATH="/home/raghuIITM/DDP_NER/checkpoints/SpeechT5/custom_ckpt_1.pt"
CHECKPOINT_PATH="/home/raghuIITM/DDP_NER/fairseq_speecht5/v2/checkpoint_best.pt"
DATA_ROOT="/home/raghuIITM/DDP_NER/ddp_slue_dataset/manifest/slue-voxpopuli"
SUBSET="dev"
BPE_TOKENIZER="/home/raghuIITM/DDP_NER/ddp/E2E/SpeechT5/SpeechT5/data/spm_char.model"
LABEL_DIR="/home/raghuIITM/DDP_NER/ddp_slue_dataset/hubert_labels"
USER_DIR="/home/raghuIITM/DDP_NER/ddp/E2E/SpeechT5/SpeechT5/speecht5"
BEAM=15
MAX_TOKENS=1600000
CTC_WEIGHT=1
LM_WEIGHT=0.5
LM_PATH="/home/raghuIITM/DDP_NER/checkpoints/transformer_lm/v2/checkpoint_best.pt"
#LM_PATH="/home/raghuIITM/DDP_NER/kenlm/4gram.bin"

fairseq-generate ${DATA_ROOT} \
  --gen-subset ${SUBSET} \
  --bpe-tokenizer ${BPE_TOKENIZER} \
  --user-dir ${USER_DIR} \
  --hubert-label-dir ${LABEL_DIR} \
  --path ${CHECKPOINT_PATH} \
  --beam ${BEAM} \
  --task speecht5 \
  --t5-task s2t \
  --max-tokens ${MAX_TOKENS} \
  --scoring wer \
  --max-len-a 0 \
  --max-len-b 700 \
  --sample-rate 16000 \
  --results-path "/home/raghuIITM/DDP_NER/ddp/E2E/SpeechT5/SpeechT5/results/v21" \
  --lm-weight ${LM_WEIGHT} \
  --lm-path ${LM_PATH} \

#  --bpe "byte_bpe"
#  --ctc-weight ${CTC_WEIGHT} \
#  --lm-weight ${LM_WEIGHT} \
#  --lm-path ${LM_PATH} \
#    --max-text-positions 700 \
#   

  # --device-id 2 \
  # --local_rank 2 \
  # --distributed-world-size 1\
  # --distributed-num-procs 1\
  # --distributed-rank \
#    


# mask-length commented in tasks
# commented hubert_dir in s2t_dataset
# --max-text-positions is 450
# --max-len-b 620