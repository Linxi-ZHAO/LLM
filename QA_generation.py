import json
import os

def obj_ls2str(obj_ls):
    obj_str = ''
    for obj in obj_ls:
        obj_str += obj + ', '
    obj_str = obj_str[:-2] + '.'
    return obj_str


def coco_prompt_control(path, save_path, coco_pope_file, coco_seg_file, prompt, control, subset=None):
    assert prompt is not None
    if subset is not None:
        save_question_file = f'I3_sub{len(subset)}'
    else:
        save_question_file = 'I3'
    if control is not None:
        save_question_file += '_control'
    save_question_file += '.json'

    coco_pope_path = os.path.join(path, coco_pope_file)
    coco_data = [json.loads(q) for q in open(coco_pope_path, 'r')]

    coco_seg_path = os.path.join(path, coco_seg_file)
    coco_seg_data = [json.loads(q) for q in open(coco_seg_path, 'r')]
    
    print(len(coco_data), coco_data[0])

    if subset is not None:
        coco_data = [coco_data[i] for i in subset]

    coco_questions_ls = []
    for i in range(0, len(coco_data), 6):
        # find the corresponding image in coco_seg_data, they have the same 'image'
        j = i//6
        while coco_data[i]['image'] != coco_seg_data[j]['image']:
            j = (j+1)%len(coco_seg_data)
        
        if control is not None:
            question = {"id": coco_data[i]['question_id'], "image": coco_data[i]['image'], "conversations": [{"from":"human", "value": prompt},{"from":"gpt", "value": control + obj_ls2str(coco_seg_data[j]['objects'])}]}
        else:
            question = {"id": coco_data[i]['question_id'], "image": coco_data[i]['image'], "conversations": [{"from":"human", "value": prompt},{"from":"gpt", "value": "None"}]}
        coco_questions_ls.append(question)

    # confirm the save path
    if not os.path.exists(os.path.join(save_path, "question")):
        os.makedirs(os.path.join(save_path, "question"))

    save_question_path = os.path.join(save_path, "question", save_question_file)
    with open(save_question_path, 'w') as f:
        json.dump(coco_questions_ls, f)
        print(f"save at {save_question_path} successfully! (len: {len(coco_questions_ls)})")
    
def coco_pope_control(path, save_path, coco_pope_file, coco_seg_file, prompt, control, subset=None, positive=True):
    assert prompt is not None
    if subset is not None:
        save_question_file = f'pope_sub{len(subset)}'
    else:
        save_question_file = 'pope'
    if control is not None:
        save_question_file += '_control'
    if positive is False:
        save_question_file += '_neg'
    save_question_file += '.json'
    save_label_file = save_question_file[:-5] + '_label.json'

    coco_pope_path = os.path.join(path, coco_pope_file)
    coco_data = [json.loads(q) for q in open(coco_pope_path, 'r')]

    coco_seg_path = os.path.join(path, coco_seg_file)
    coco_seg_data = [json.loads(q) for q in open(coco_seg_path, 'r')]
    
    print(len(coco_data), coco_data[0])

    if subset is not None:
        coco_data = [coco_data[i] for i in subset]

    coco_questions_ls = []
    coco_labels_ls = []
    for i in range(0, len(coco_data), 1):
        # find the corresponding image in coco_seg_data, they have the same 'image'
        j = i//6
        while coco_data[i]['image'] != coco_seg_data[j]['image']:
            j = (j+1)%len(coco_seg_data)

        prompt = coco_data[i]['text']
        label = coco_data[i]['label']
        # change prompt to negative expression

        # original prompt: Is there a bag in the image?
        # new prompt: Doesn't a bag exist in the image?
        # change the original prompt to new prompt
        if positive is False:
            prompt = prompt[0].upper() + prompt[1:]
            prompt = prompt.replace('Is there', "Doesn't")
            prompt = prompt.replace(' in', ' exist in')
            label = 'No' if label == 'Yes' or label == 'yes' else 'Yes'

        if control is not None:
            question = {"id": coco_data[i]['question_id'], "image": coco_data[i]['image'], "conversations": [{"from":"human", "value": prompt},{"from":"gpt", "value": control + obj_ls2str(coco_seg_data[j]['objects'])}]}
            label = {"id": coco_data[i]['question_id'], "image": coco_data[i]['image'], "label": label}
        else:
            question = {"id": coco_data[i]['question_id'], "image": coco_data[i]['image'], "conversations": [{"from":"human", "value": prompt},{"from":"gpt", "value": "None"}]}
            label = {"id": coco_data[i]['question_id'], "image": coco_data[i]['image'], "label": label}
        coco_questions_ls.append(question)
        coco_labels_ls.append(label)

    # confirm the save path
    if not os.path.exists(os.path.join(save_path, "question")):
        os.makedirs(os.path.join(save_path, "question"))
    if not os.path.exists(os.path.join(save_path, "label")):
        os.makedirs(os.path.join(save_path, "label"))
    
    question_file = os.path.join(save_path, "question", save_question_file)
    answer_file = os.path.join(save_path, "label", save_label_file)
    with open(question_file, 'w') as f:
        json.dump(coco_questions_ls, f)
        print(f"save at {question_file} successfully! (len: {len(coco_questions_ls)})")
    with open(answer_file, 'w') as f:
        json.dump(coco_labels_ls, f)
        print(f"save at {answer_file} successfully! (len: {len(coco_labels_ls)})")


if __name__ == "__main__":
    path = "../POPE/output/coco"
    coco_pope_file = "coco_pope_adversarial.json"
    coco_seg_file = "coco_ground_truth_segmentation.json"

    save_path = "../POPE/llava_qa"
    prompt = "Provide a brief description of the given image."
    # control = "It is confirmed that the following objects appear in the image: " 
    control = "It has been verified that the image exclusively contains the following objects: "


    subset = [i for i in range(0, 240)]
    # subset = None
    # coco_prompt_control(path, save_path, coco_pope_file, coco_seg_file, prompt, control, subset)
    coco_pope_control(path, save_path, coco_pope_file, coco_seg_file, prompt, control, subset, positive=False)
    
    