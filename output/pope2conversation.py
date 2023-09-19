import json
import os

def coco_pope2qa(path, coco_pope_file, subset=None):
    coco_pope_path = os.path.join(path, coco_pope_file)
    coco_data = [json.loads(q) for q in open(coco_pope_path, 'r')]
    print(len(coco_data), coco_data[0])

    if subset is not None:
        coco_data = [coco_data[i] for i in subset]

    coco_questions_ls = []
    coco_answer_ls = []
    for i in range(len(coco_data)):
        # print(coco_data[i])
        question = {"id": coco_data[i]['question_id'], "image": coco_data[i]['image'], "conversations": [{"from":"human", "value": coco_data[i]['text']},{"from":"gpt", "value": "None"}]}
        coco_questions_ls.append(question)
        answer = {"id": coco_data[i]['question_id'], "label": coco_data[i]['label']}
        coco_answer_ls.append(answer)
    print(len(coco_questions_ls))
    print(len(coco_answer_ls))

    save_question_file = coco_pope_file[:-5] + f'_question_{len(subset)}.json'
    save_answer_file = coco_pope_file[:-5] + f'_label_{len(subset)}.json'
    with open(os.path.join(path, save_question_file), 'w') as f:
        json.dump(coco_questions_ls, f)
        print(f"save at {os.path.join(path, save_question_file)} successfully!")
    with open(os.path.join(path, save_answer_file), 'w') as f:
        json.dump(coco_answer_ls, f)
        print(f"save at {os.path.join(path, save_answer_file)} successfully!")

def coco2chair(path, coco_pope_file, instruction, subset=None):
    coco_pope_path = os.path.join(path, coco_pope_file)
    coco_data = [json.loads(q) for q in open(coco_pope_path, 'r')]
    print(len(coco_data), coco_data[0])

    if subset is not None:
        coco_data = [coco_data[i] for i in subset]

    coco_questions_ls = []
    for i in range(0, len(coco_data), 6):
        # print(coco_data[i])
        question = {"id": coco_data[i]['question_id'], "image": coco_data[i]['image'], "conversations": [{"from":"human", "value": instruction},{"from":"gpt", "value": "None"}]}
        coco_questions_ls.append(question)
    print(len(coco_questions_ls))

    save_question_file = f'coco_instruction1_{len(subset)}.json'
    with open(os.path.join(path, save_question_file), 'w') as f:
        json.dump(coco_questions_ls, f)
        print(f"save at {os.path.join(path, save_question_file)} successfully!")

def coco_pope2qa_not_conv(path, coco_pope_file, subset=None):
    coco_pope_path = os.path.join(path, coco_pope_file)
    coco_data = [json.loads(q) for q in open(coco_pope_path, 'r')]
    print(len(coco_data), coco_data[0])
    coco_questions_ls = []
    coco_answer_ls = []
    for i in range(len(coco_data)):
        # print(coco_data[i])
        question = {"id": coco_data[i]['question_id'], "image": coco_data[i]['image'], "conversations": [coco_data[i]['text']]}
        coco_questions_ls.append(question)
        answer = {"id": coco_data[i]['question_id'], "label": coco_data[i]['label']}
        coco_answer_ls.append(answer)
    print(len(coco_questions_ls))
    print(len(coco_answer_ls))

    if subset is not None:
        coco_data = [coco_data[i] for i in subset]


    save_question_file = coco_pope_file[:-5] + '_question_not_conv.json'
    save_answer_file = coco_pope_file[:-5] + '_label_not_conv.json'
    with open(os.path.join(path, save_question_file), 'w') as f:
        json.dump(coco_questions_ls, f)
        print(f"save at {os.path.join(path, save_question_file)} successfully!")
    with open(os.path.join(path, save_answer_file), 'w') as f:
        json.dump(coco_answer_ls, f)
        print(f"save at {os.path.join(path, save_answer_file)} successfully!")

def coco2blank_img(path, coco_pope_file):
    coco_pope_path = os.path.join(path, coco_pope_file)
    coco_data = [json.loads(q) for q in open(coco_pope_path, 'r')]
    print(len(coco_data), coco_data[0])

    if subset is not None:
        coco_data = [coco_data[i] for i in subset]

    coco_questions_ls = []
    coco_answer_ls = []
    for i in range(len(coco_data)):
        # print(coco_data[i])
        question = {"id": coco_data[i]['question_id'], "image": "blank.jpg", "conversations": [{"from":"human", "value": coco_data[i]['text']},{"from":"gpt", "value": "None"}]}
        coco_questions_ls.append(question)
        answer = {"id": coco_data[i]['question_id'], "label": coco_data[i]['label']}
        coco_answer_ls.append(answer)
    print(len(coco_questions_ls))
    print(len(coco_answer_ls))

    save_question_file = coco_pope_file[:-5] + '_question_blank.json'
    save_answer_file = coco_pope_file[:-5] + '_label.json'
    with open(os.path.join(path, save_question_file), 'w') as f:
        json.dump(coco_questions_ls, f)
        print(f"save at {os.path.join(path, save_question_file)} successfully!")
    with open(os.path.join(path, save_answer_file), 'w') as f:
        json.dump(coco_answer_ls, f)
        print(f"save at {os.path.join(path, save_answer_file)} successfully!")

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


def concate_gt_obj(path, coco_pope_file, coco_seg_file, instruction, subset=None):
    coco_pope_path = os.path.join(path, coco_pope_file)
    coco_data = [json.loads(q) for q in open(coco_pope_path, 'r')]

    coco_seg_path = os.path.join(path, coco_seg_file)
    coco_seg_data = [json.loads(q) for q in open(coco_seg_path, 'r')]
    print(len(coco_data), coco_data[0])
    print(len(coco_seg_data), coco_seg_data[0])

    if subset is not None:
        coco_data = [coco_data[i] for i in subset]

    coco_questions_ls = []
    coco_answer_ls = []
    for i in range(len(coco_data)):
        # find the corresponding image in coco_seg_data, they have the same 'image'
        j = i//6
        while coco_data[i]['image'] != coco_seg_data[j]['image']:
            j = (j+1)%len(coco_seg_data)
        # print("find", i, j)
        if instruction is None:
            question = {"id": coco_data[i]['question_id'], "image": coco_data[i]['image'], "conversations": [{"from":"human", "value": coco_data[i]['text']},{"from":"gpt", "value": "It is confirmed that the following objects appear in the image: " + obj_ls2str(coco_seg_data[j]['objects'])}]}
        else:
            question = {"id": coco_data[i]['question_id'], "image": coco_data[i]['image'], "conversations": [{"from":"human", "value": instruction},{"from":"gpt", "value": "It is confirmed that the following objects appear in the image: " + obj_ls2str(coco_seg_data[j]['objects'])}]}
        coco_questions_ls.append(question)
        answer = {"id": coco_data[i]['question_id'], "label": coco_data[i]['label']}
        coco_answer_ls.append(answer)
    print(len(coco_questions_ls))
    print(len(coco_answer_ls))

    if instruction is None:
        save_question_file = coco_pope_file[:-5] + '_question_hint.json'
        save_answer_file = coco_pope_file[:-5] + '_label_hint.json'
    else:
        save_question_file = coco_pope_file[:-5] + '_question_hint_instruction.json'
        save_answer_file = coco_pope_file[:-5] + '_label_hint_instruction.json'
    with open(os.path.join(path, save_question_file), 'w') as f:
        json.dump(coco_questions_ls, f)
        print(f"save at {os.path.join(path, save_question_file)} successfully!")
    with open(os.path.join(path, save_answer_file), 'w') as f:
        json.dump(coco_answer_ls, f)
        print(f"save at {os.path.join(path, save_answer_file)} successfully!")

def coco2blank_image_all_obj_q(path, coco_pope_file, coco_gt_obj_file, subset=None):
    coco_pope_path = os.path.join(path, coco_pope_file)
    coco_data = [json.loads(q) for q in open(coco_pope_path, 'r')]
    print(len(coco_data), coco_data[0])

    # read dict in /home/linxi/workspace/POPE/output/coco/coco_ground_truth_objects.json
    with open(os.path.join(path, coco_gt_obj_file), 'r') as f:
        gt_obj_ls = json.load(f)
    print(len(gt_obj_ls), gt_obj_ls)
    # get keys of gt_obj_ls
    gt_obj_keys = list(gt_obj_ls.keys())

    coco_questions_ls = []
    coco_answer_ls = []
    for i in range(len(gt_obj_keys)):
        # print(coco_data[i])
        question = {"id": coco_data[i]['question_id'], "image": "blank.jpg", "conversations": [{"from":"human", "value": f"Is there a {gt_obj_keys[i]} in the image?"},{"from":"gpt", "value": "None"}]}
        coco_questions_ls.append(question)
        answer = {"id": coco_data[i]['question_id'], "label": "No"}
        coco_answer_ls.append(answer)
    print(len(coco_questions_ls))
    print(len(coco_answer_ls))

    save_question_file = coco_pope_file[:-5] + '_question_blank_all.json'
    # save_answer_file = coco_pope_file[:-5] + '_label_blank_all.json'
    with open(os.path.join(path, save_question_file), 'w') as f:
        json.dump(coco_questions_ls, f)
        print(f"save at {os.path.join(path, save_question_file)} successfully!")
 
def qa2instruct(path, coco_question_file, coco_answer_file):
    coco_question_path = os.path.join(path, coco_question_file)
    coco_questions = [json.loads(q) for q in open(coco_question_path, 'r')][0]
    print(len(coco_questions), coco_questions[0])

    coco_answer_path = os.path.join(path, coco_answer_file)
    coco_answers = [json.loads(q) for q in open(coco_answer_path, 'r')]
    print(len(coco_answers), coco_answers[0])

    coco_instructions = []
    for i in range(len(coco_questions)):
        instruction = {"id": coco_questions[i]['id'], "image": coco_questions[i]['image'], "conversations": [{"from":"human", "value": coco_questions[i]['conversations'][0]['value']},{"from":"gpt", "value": coco_answers[i]['text']}]}
        coco_instructions.append(instruction)

    print(len(coco_instructions))
    print(coco_instructions[0])

    save_instruction_file = coco_question_file[:-14] + '_instruction.json'
    with open(os.path.join(path, save_instruction_file), 'w') as f:
        json.dump(coco_instructions, f)
        print(f"save at {os.path.join(path, save_instruction_file)} successfully!")


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
    
    # coco_pope2qa(path, coco_pope_file, subset)
    # coco2chair(path, coco_pope_file, "Provide a brief description of the given image.", subset)

    # coco_pope2qa_not_conv(path, coco_pope_file, subset)
    # concate_gt_obj(path, coco_pope_file, coco_seg_file, prompt, subset)
    # coco2blank_img(path, coco_pope_file)
    # coco_gt_obj_file = "coco_ground_truth_objects.json"
    # coco2blank_image_all_obj_q(path, coco_pope_file, coco_gt_obj_file)
    # path2 = "../POPE/output/adversarial_qa_gt"
    # coco_question_file = "coco_pope_adversarial_question.json"
    # coco_answer_file = "coco_pope_adversarial_answer.jsonl"
    # qa2instruct(path2, coco_question_file, coco_answer_file)