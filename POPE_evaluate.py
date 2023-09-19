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
    
    parser.add_argument("--save_path", type=str, default="../POPE/llava_eval_results/pope")
    
    parser.add_argument("--question_path", type=str, default="../POPE/llava_qa/question")
    parser.add_argument("--question_file", type=str, default="I1_sub240_control.json")
    parser.add_argument("--answer_path", type=str, default="../POPE/llava_qa/answer")
    parser.add_argument("--label_path", type=str, default="../POPE/llava_qa/label")
    parser.add_argument("--answers_file", type=str, default="I1_sub240_control_cfg1.0.jsonl")
    parser.add_argument("--labels_file", type=str, default='pope_control_label.json')
    parser.add_argument("--cfg_ls", nargs='+', type=float, default=[1.0])

    args = parser.parse_args()
    return args


def eval(args, results):
    answer_file = os.path.join(args.answer_path, args.answers_file)
    label_file = os.path.join(args.label_path, args.labels_file)
    answers = [json.loads(q) for q in open(answer_file, 'r')]
    file = [json.loads(q) for q in open(label_file, 'r')][0]
    label_list = [e['label'] for e in file]

    for answer in answers:
        text = answer['text']

        # Only keep the first sentence
        if text.find('.') != -1:
            text = text.split('.')[0]

        text = text.replace(',', '')
        words = text.split(' ')
        if 'No' in words or 'not' in words or 'no' in words:
            answer['answer'] = 'no'
        else:
            answer['answer'] = 'yes'

    for i in range(len(label_list)):
        if label_list[i] == 'no':
            label_list[i] = 0
        else:
            label_list[i] = 1
    pred_list = []
    for answer in answers:
        if answer['answer'] == 'no':
            pred_list.append(0)
        else:
            pred_list.append(1)
    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    eval_results = {
        "answer_file": args.answers_file,

        "questions_num": len(results),
        "length_response": round(sum([len(result["response"].split()) for result in results]) / len(results) if len(results) > 0 else 0, 2),

        "Accuracy": round((TP + TN )/ (TP + TN + FP + FN) if TP + TN + FP + FN > 0 else 0, 4),
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "Yes_ratio": round(yes_ratio, 4),
        # round to 4 digits
        "Precision": round(TP / (TP + FP) if TP + FP > 0 else 0, 4),
        "Recall": round(TP / (TP + FN) if TP + FN > 0 else 0, 4),
        "Specificity": round(TN / (TN + FP) if TN + FP > 0 else 0, 4),
        "F1": round(2 * TP / (2 * TP + FP + FN) if 2 * TP + FP + FN > 0 else 0, 4),

    }

    print(eval_results)

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