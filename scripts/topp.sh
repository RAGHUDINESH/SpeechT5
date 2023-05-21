CHECKPOINT_PATH="/home/raghuIITM/DDP_NER/fairseq_speecht5/v1/checkpoint_best.pt"
DATA_ROOT="/home/raghuIITM/DDP_NER/ddp_slue_dataset/manifest/slue-voxpopuli"
SUBSET="dev"
BPE_TOKENIZER="/home/raghuIITM/DDP_NER/ddp/E2E/SpeechT5/SpeechT5/data/spm_char.model"
LABEL_DIR="/home/raghuIITM/DDP_NER/ddp_slue_dataset/ne_as_constraints"
USER_DIR="/home/raghuIITM/DDP_NER/ddp/E2E/SpeechT5/SpeechT5/speecht5"
BEAM=1
MAX_TOKENS=1600000
#CTC_WEIGHT=1
# LM_WEIGHT=1
# LM_PATH="/home/raghuIITM/DDP_NER/checkpoints/SpeechT5/t5_LM.pt"

fairseq-generate ${DATA_ROOT} \
  --gen-subset ${SUBSET} \
  --bpe-tokenizer ${BPE_TOKENIZER} \
  --user-dir ${USER_DIR} \
  --hubert-label-dir ${LABEL_DIR} \
  --path ${CHECKPOINT_PATH} \
  --task speecht5 \
  --beam ${BEAM} \
  --sampling \
  --sampling-topk 3 \
  --sampling-topp 0.75 \
  --t5-task s2t \
  --max-tokens ${MAX_TOKENS} \
  --scoring wer \
  --max-len-a 0 \
  --max-len-b 700 \
  --sample-rate 16000 \
  --results-path "/home/raghuIITM/DDP_NER/ddp/E2E/SpeechT5/SpeechT5/results/v8" 

#   --constraints ordered\NLM/