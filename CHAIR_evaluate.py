from hallucination_detection import hallucination_detection, revise_obj_ls, get_all_obj_ls, get_gt_obj_ls_ls
import argparse
import os
import numpy as np
import json
from hallucination_loss import ObjectLoss

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./checkpoints/llava-llama-2-13b-chat-lightning-preview")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="../POPE/data/minival2014/minival2014")
    
    parser.add_argument("--save_path", type=str, default="../POPE/llava_eval_results/loss")
    
    parser.add_argument("--question_path", type=str, default="../POPE/llava_qa/question")
    parser.add_argument("--question_file", type=str, default="I1_sub240_control.json")
    parser.add_argument("--answer_path", type=str, default="../POPE/llava_qa/answer")
    parser.add_argument("--answers_file", type=str, default="I1_sub240_control_cfg1.0")
    parser.add_argument("--cfg_ls", nargs='+', type=float, default=[1.0])

    args = parser.parse_args()
    return args


def eval(args, results):
    TP_counter = 0
    FP_counter = 0
    FN_counter = 0
    TN_counter = 0
    hallu_counter = 0

    for result in results:
        FP_counter += len(result["FP_obj_ls"])
        FN_counter += len(result["FN_obj_ls"])
        TN_counter += len(result["TN_obj_ls"])
        TP_counter += len(result["TP_obj_ls"])

        if len(result["FP_obj_ls"]) > 0 or len(result["FN_obj_ls"]) > 0:
            hallu_counter += 1

    if not os.path.exists(os.path.join(args.save_path, "detailed")):
        os.makedirs(os.path.join(args.save_path, "detailed"))
    save_file_path = os.path.join(args.save_path, "detailed", args.answers_file.replace(".jsonl", ".json"))

    sum_loss_ls = np.sum(np.array([result["loss_ls"] for result in results]), axis=0)

    eval_results = {
        "answer_file": args.answers_file,
        "detailed_results_file": save_file_path,

        "captions_w_hallu_obj": hallu_counter,
        "all_captions": len(results),
        "captions_hallu_ratio(CHAIR_S)": hallu_counter / len(results) if len(results) > 0 else 0,
        "hallu_obj": FN_counter + FP_counter,
        "all_mentioned_obj": TP_counter + FP_counter + FN_counter + TN_counter,
        "obj_hallu_ratio(CHAIR_I)": (FN_counter + FP_counter) / (TP_counter + FP_counter + FN_counter + TN_counter) if TP_counter + FP_counter + FN_counter + TN_counter > 0 else 0,
        "length_response": sum([len(result["response"].split()) for result in results]) / len(results) if len(results) > 0 else 0,

        "Accuracy": (TP_counter + TN_counter) / (TP_counter + TN_counter + FP_counter + FN_counter) if TP_counter + TN_counter + FP_counter + FN_counter > 0 else 0,
        "Precision": TP_counter / (TP_counter + FP_counter) if TP_counter + FP_counter > 0 else 0,
        "Recall": TP_counter / (TP_counter + FN_counter) if TP_counter + FN_counter > 0 else 0,
        "Specificity": TN_counter / (TN_counter + FP_counter) if TN_counter + FP_counter > 0 else 0,

        "TP": TP_counter,
        "FP": FP_counter,
        "FN": FN_counter,
        "TN": TN_counter,

        "total_loss": sum([result["total_loss"] for result in results]),
        "loss_ls": list(sum_loss_ls),

    }

    print(eval_results)

    with open(save_file_path, 'w') as f:
        print(f"save detailed results successfully at {save_file_path}")
        json.dump(results, f)

    file_name = args.answers_file[:-13] +"_eval.json"
    save_file_path = os.path.join(args.save_path, file_name)
    with open(save_file_path, 'a') as f:
        json.dump(eval_results, f)
        f.write('\n')
        print(f"save eval results successfully at {save_file_path}")


def run(args):
    answer_file = os.path.join(args.answer_path, args.answers_file)
    answers = [json.loads(q) for q in open(answer_file, 'r')]

    question_file = os.path.join(args.question_path, args.question_file)    
    questions = [json.loads(q) for q in open(question_file, 'r')][0]
    
    gt_obj_ls_ls = get_gt_obj_ls_ls()
    all_obj_ls, all_obj_prob = get_all_obj_ls()
    # save the results in a json file
    results = []
    for answer,question in zip(answers,questions):
        response = answer['text']
        assert answer['question_id'] == question['id']
        for i in range(len(gt_obj_ls_ls)):
            if gt_obj_ls_ls[i]["image"] == question["image"]:
                gt_obj_ls = revise_obj_ls(gt_obj_ls_ls[i]["objects"])
                break

        TP_obj_ls, FP_obj_ls, FN_obj_ls, TN_obj_ls = hallucination_detection(response, gt_obj_ls, all_obj_ls)

        loss_ls = ObjectLoss(gt_obj_ls, TP_obj_ls, FP_obj_ls, FN_obj_ls, TN_obj_ls)

        if len(FP_obj_ls) > 0 or len(FN_obj_ls) > 0:
            mistake = True
        else:
            mistake = False
        result = {"id": answer['question_id'], "image": question["image"], "mistake": mistake, "total_loss": sum(loss_ls), "loss_ls": loss_ls, "response": response, "FP_obj_ls": FP_obj_ls, "FN_obj_ls": FN_obj_ls, "TP_obj_ls": TP_obj_ls, "TN_obj_ls": TN_obj_ls}
        results.append(result)
    
    eval(args, results)
    


if __name__ == "__main__":
    args = get_parser()
    if args.cfg_ls is not None:
        for cfg in args.cfg_ls:
            args.answers_file = args.answers_file[:-13] + f"_cfg{cfg}.jsonl"
            run(args)
    else:
        run(args)