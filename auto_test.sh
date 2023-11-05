# Script for auto testing CHAIR, given the generated answers file

## The following variables are taken from eval_cfg.sh
#### Model ####
MODEL_SIZE=7b
MODEL_VERSION=llava-llama-2-7b-chat
BASE_BRANCH=lightning-preview
CFG_BRANCH=DETR-v2-pad-sl1-th07-pretrain-1-tune-1

#### Prompt ####
PROMPT_VERSION="llava_llama_2"

#### QA data ####
QUESTION_FILE_PREFIX=I3_sub120
QUESTION_DIR=../POPE/llava_qa/question
ANSWERS_DIR=../POPE/llava_performance_cfg/test

#### Additional Setting ####
CFG_SUFFIX=cfg
SAVE_CHAIR_DIR=../POPE/llava_eval_results_cfg

########## Run ##########
python ../POPE/answers2seperate_files.py \
--answers_file $QUESTION_FILE_PREFIX-$MODEL_SIZE-$BASE_BRANCH-$CFG_BRANCH-$CFG_SUFFIX.jsonl \
--answers_dir $ANSWERS_DIR \
--save_dir $ANSWERS_DIR \

python ../POPE/CHAIR_evaluate.py \
--question_file $QUESTION_FILE_PREFIX.json \
--save_path  $SAVE_CHAIR_DIR \
--answer_path $ANSWERS_DIR/$QUESTION_FILE_PREFIX-$MODEL_SIZE-$BASE_BRANCH-$CFG_BRANCH/ \
--answers_file $QUESTION_FILE_PREFIX-$MODEL_SIZE-$BASE_BRANCH-$CFG_BRANCH-$CFG_SUFFIX.jsonl \
