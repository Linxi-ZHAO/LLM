import json
import os
import math
from hallucination_detection import revise_obj_ls, get_all_obj_ls

# count error times as loss

def condition_prob(all_obj_prob, h, o):
    path = "../POPE/output/coco"
    coco_co_occur_file = "coco_co_occur.json" 

    with open(os.path.join(path, coco_co_occur_file), 'r') as f:
        co_occur = json.load(f)
        # p(h|o)

    # h: hallu
    # o: object
    # h and o co-occur
    if o in co_occur.keys() and h in co_occur[o]:
        # p(h|o)
        # assume that the probability of each co-occur object is the same
        return 1/len(co_occur[o])
    else:
        # p(h|o) = p(h)
        return all_obj_prob[h]

def prob(all_obj_prob, h, o):
    # p(h|o) * p(o)
    if h == o:
        return all_obj_prob[o]
    else:
        return condition_prob(all_obj_prob, h, o) * all_obj_prob[o]

def punish_loss(all_obj_prob, h, o):
    # -log(p(h|o) * p(o))
    return -math.log(prob(all_obj_prob, h, o))

def award_loss(all_obj_prob, t_o):
    # -log(p(h|o) * p(o))
    # prob smaller, loss bigger
    return -math.log(all_obj_prob[t_o])

def ObjectLoss(gt_obj_ls, TP_obj_ls, FP_obj_ls, FN_obj_ls, TN_obj_ls):
    tp_loss = 0
    fp_loss = 0
    fn_loss = 0
    tn_loss = 0

    all_obj_prob = get_all_obj_ls()[1]

    for obj in TP_obj_ls:
        tp_loss += award_loss(all_obj_prob, obj)
    for obj in TN_obj_ls:
        tn_loss += award_loss(all_obj_prob, obj)
    for fp_obj in FP_obj_ls:
        h_loss = 0
        for gt_obj in gt_obj_ls:
            h_loss += punish_loss(all_obj_prob, fp_obj, gt_obj)
        fp_loss += h_loss/len(gt_obj_ls)
    for fn_obj in FN_obj_ls:
        fn_loss += punish_loss(all_obj_prob, fn_obj, fn_obj)
    loss_ls = [-tp_loss, fp_loss, fn_loss, -tn_loss]
    return loss_ls


    