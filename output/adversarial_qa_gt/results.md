CUDA_VISIBLE_DEVICES=5,6 python -m llava.eval.model_vqa_POPE --question_file "../POPE/output/coco/coco_pope_adversarial_question.json" --answers_file "../POPE/output/coco/coco_pope_adversarial_answer.jsonl"

python ../POPE/evaluate.py --answers_file "../POPE/output/coco/coco_pope_adversarial_answer.jsonl" --labels_file "../POPE/output/coco/coco_pope_adversarial_label.json"

TP      FP      TN      FN
1476    1111    389     24
Accuracy: 0.6216666666666667
Precision: 0.5705450328565906
Recall: 0.984
F1 score: 0.722290188402251
Yes ratio: 0.8623333333333333