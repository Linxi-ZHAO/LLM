import json
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--answers_file", type=str, default='../POPE/output/answer_coco_minival_conversation.jsonl')
parser.add_argument("--labels_file", type=str, default='../POPE/output/coco/conversation/coco_minival_conversation_label.json')

args = parser.parse_args()

answers = [json.loads(q) for q in open(args.answers_file, 'r')]
file = [json.loads(q) for q in open(args.labels_file, 'r')][0]
label_list = [e['label'] for e in file]
print(len(answers), len(label_list))

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

print('TP\tFP\tTN\tFN\t')
print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

precision = float(TP) / float(TP + FP)
recall = float(TP) / float(TP + FN)
f1 = 2*precision*recall / (precision + recall)
acc = (TP + TN) / (TP + TN + FP + FN)
print('Accuracy: {}'.format(acc))
print('Precision: {}'.format(precision))
print('Recall: {}'.format(recall))
print('F1 score: {}'.format(f1))
print('Yes ratio: {}'.format(yes_ratio))
