# ../POPE/llava_performance_cfg/test/I3_sub240_7b_lightning-preview-DETR-v2-pad-sl1-th07-pretrain-1-tune-1_cfg0.0.jsonl
# load
import json
import os
import argparse
import numpy as np

def get_parser():
    parser = argparse.ArgumentParser()
    # answers_file
    # answers_dir
    # save_dir

    parser.add_argument("--answers_file", type=str, default="I3_sub240_7b_lightning-preview-DETR-v2-pad-sl1-th07-pretrain-1-tune-1_cfg0.0.jsonl")
    parser.add_argument("--answers_dir", type=str, default="../POPE/llava_performance_cfg/test")
    parser.add_argument("--save_dir", type=str, default="../POPE/llava_performance_cfg/test")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # load file
    args = get_parser()
    answers_file = os.path.join(args.answers_dir, args.answers_file)
    with open(answers_file) as f:
        answers = [json.loads(line) for line in f]

    results = {}
    cfg_ls = []
    for answer in answers:
        if answer["cfg"] not in cfg_ls:
            cfg_ls.append(answer["cfg"])

    answers_dir = os.path.join(args.save_dir, args.answers_file.split("-cfg")[0])
    os.makedirs(answers_dir, exist_ok=True)
    for cfg in cfg_ls:
        results[cfg] = []
        cfg_file = os.path.join(answers_dir, f"{args.answers_file.split('-cfg')[0]}-cfg{cfg}.jsonl")
        with open(cfg_file, 'a') as f:
            for answer in answers:
                if answer["cfg"] == cfg:
                    f.write(json.dumps(answer) + "\n")
                    results[cfg].append(answer)
        print(f"saved {len(results[cfg])} answers to {cfg_file}")
        
