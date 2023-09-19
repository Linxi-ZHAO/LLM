import os
import json

def revise_obj_ls(obj_ls):
    if "teddy bear" in obj_ls:
        obj_ls.remove("teddy bear")
        obj_ls.append("teddy-bear")
    if "hot dog" in obj_ls:
        obj_ls.remove("hot dog")
        obj_ls.append("hot-dog")
    return obj_ls

def revise_response(response):
    response = response.replace("teddy bear", "teddy-bear")
    response = response.replace("hot dog", "hot-dog")
    return response

def get_all_obj_ls():
    path = "../POPE/output/coco"
    coco_all_obj_file = "coco_ground_truth_objects.json"
    with open(os.path.join(path, coco_all_obj_file), 'r') as f:
        all_obj_dict = json.load(f)
    all_obj_ls = list(all_obj_dict.keys())
    all_obj_ls = revise_obj_ls(all_obj_ls)

    all_obj_prob = {obj: value / sum(all_obj_dict.values()) for obj, value in zip(all_obj_ls, all_obj_dict.values())}

    return all_obj_ls, all_obj_prob

def get_gt_obj_ls_ls():
    coco_gt_obj_file = "coco_ground_truth_segmentation.json"
    path = "../POPE/output/coco"
    gt_obj_ls_ls = [json.loads(q) for q in open(os.path.join(path, coco_gt_obj_file), 'r')]
    return gt_obj_ls_ls

def hallucination_detection(response, gt_obj_ls, all_obj_ls):
    response = response.lower()

    # special cases: detect dog while avoiding hot dog
    all_obj_ls = revise_obj_ls(all_obj_ls)
    gt_obj_ls = revise_obj_ls(gt_obj_ls)
    response = revise_response(response)

    punctuations = ['.', ',', '?', '!', ';', ':']
    for p in punctuations:
        response = response.replace(p, "")
    response = " " + response + " "
    # conjunction words in response and gt_obj_ls
    # turn next line into multiple lines
    response_obj_ls = []
    response_obj_org_ls = []
    obj_index = []
    for obj in all_obj_ls :
        if " "+obj+" " in response:
            response_obj_ls.append(obj)
            response_obj_org_ls.append(obj)
        elif " "+obj+"s " in response:
            response_obj_ls.append(obj)
            response_obj_org_ls.append(obj+"s")
        elif " "+obj+"es " in response:
            response_obj_ls.append(obj)
            response_obj_org_ls.append(obj+"es")

    # judge the objects in response are positive(exist) or negative(does not exist in the image)
    # if the object in response is in a positive sentence, it is positive
    # if the object in response is in a negative sentence, it is negative
    response = response.split()
    response_pos_obj_ls = []
    response_neg_obj_ls = []
    pos_or_neg = ["no", "not"]
    for obj_org, obj in zip(response_obj_org_ls, response_obj_ls):
        obj_index = response.index(obj_org.split()[0])
        if obj_index == 0:
            continue
        else:
            if response[obj_index - 1] in pos_or_neg:
                response_neg_obj_ls.append(obj)
            else:
                response_pos_obj_ls.append(obj)

    TP_obj_ls = []
    FP_obj_ls = [] # hallucination
    FN_obj_ls = [] # miss
    TN_obj_ls = [] # all_obj_ls - TP_obj_ls - FP_obj_ls - FN_obj_ls

    TP_obj_ls = [obj for obj in response_pos_obj_ls if obj in gt_obj_ls]
    FP_obj_ls = [obj for obj in response_pos_obj_ls if obj not in gt_obj_ls]
    FN_obj_ls = [obj for obj in response_neg_obj_ls if obj in gt_obj_ls]

    TN_obj_ls = [obj for obj in response_neg_obj_ls if obj not in gt_obj_ls]
    
    return TP_obj_ls, FP_obj_ls, FN_obj_ls, TN_obj_ls

if __name__ == "__main__":
    query = "Is there a keyboard in the image?"
    response1 = "Yes, there is a keyboard in the image, which is placed next to the laptop."
    response2 = "No, there is no person in the image. The image features a laptop computer and a mouse on a desk. Yes, there is a dining table in the image."
    response = response1 + " " + response2
    gt_obj_ls = ["keyboard", "laptop", "dining table"]

    path = "../POPE/output/coco"
    coco_gt_obj_file = "coco_ground_truth_objects.json"
    with open(os.path.join(path, coco_gt_obj_file), 'r') as f:
        all_obj_ls = json.load(f)
    all_obj_ls = list(all_obj_ls.keys())
    print("all_obj_ls: ", all_obj_ls)

    TP_obj_ls, FP_obj_ls, FN_obj_ls, TN_obj_ls = hallucination_detection(response, gt_obj_ls, all_obj_ls)
    print("TP_obj_ls: ", TP_obj_ls)
    print("FP_obj_ls: ", FP_obj_ls)
    print("FN_obj_ls: ", FN_obj_ls)
    print("TN_obj_ls: ", TN_obj_ls)
