import numpy as np
import json
import numpy

import torch
from transformers import (
    RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)


class RobertaService():
    def __init__(self):
        PATH = '../../../../../ml/model/roberta'
        MODEL_NAME = PATH + '/models/tanda_roberta_large_asnq_wikiqa'
        config_PATH = MODEL_NAME + "/ckpt/config.json"
        model_PATH = MODEL_NAME + "/ckpt/pytorch_model.bin"
        tokenizer_PATH = MODEL_NAME + "/ckpt/"

        self._tokenizer = RobertaTokenizer.from_pretrained(tokenizer_PATH)

        config = RobertaConfig.from_pretrained(config_PATH, num_labels=2)
        # finetuning_task='asnq')
        self._model = RobertaForSequenceClassification.from_pretrained(
            model_PATH, config=config)
        print("##roberta init sucsess")

    def sort_answer(self, info):
        question, answers = info['question'], info['answers']
        print("##question: ", question)
        print("##answers: ", answers)

        result = []

        for i in answers:
            inputs = self._tokenizer.encode_plus(
                question,
                i[1],
                add_special_tokens=True,
                max_length=128,
            )
            outputs = self._model(torch.tensor([inputs['input_ids']]))
            preds = torch.nn.functional.softmax(outputs[0])
            print(preds[0][1])
            result.append([i[0], i[1], float(preds[0][1])])

        result.sort(key=lambda x: -x[2])
        return result
