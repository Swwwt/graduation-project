import numpy as np
import json
import numpy

import onnxruntime
import torch
from transformers import BertTokenizer


class BertService():
    def __init__(self):
        MODEL_PATH = './model/optimized_bert_model_cpu.onnx'
        self._tokenizer = BertTokenizer.from_pretrained(
        './model/')
        #    'bert-base-uncased')
        self._session = onnxruntime.InferenceSession(
            MODEL_PATH)
        print("##bert init sucsess")

    def sort_answer(self, info):
        question, answers = info['question'], info['answers']
        print("##question: ", question)
        print("##answers: ", answers)

        batch_token_type_ids = []
        batch_input_ids = []
        batch_attention_mask = []

        for i in answers:
            inputs = self._tokenizer.encode_plus(
                question,
                i['answer'],
                add_special_tokens=True,
                max_length=128,
            )

            token_type_ids = np.zeros(128, dtype=int)
            token_type_ids[:len(inputs['token_type_ids'])
                           ] = inputs['token_type_ids']
            batch_token_type_ids.append(token_type_ids)

            input_ids = np.zeros(128, dtype=int)
            input_ids[:len(inputs['input_ids'])] = inputs['input_ids']
            batch_input_ids.append(input_ids)

            attention_mask = np.zeros(128, dtype=int)
            attention_mask[:len(inputs['input_ids'])] = np.ones(
                len(inputs['input_ids']), dtype=int)
            batch_attention_mask.append(attention_mask)

        n = len(answers)
        ort_inputs = {
            'input_ids':  numpy.ascontiguousarray(torch.tensor(batch_input_ids).cpu().reshape(n, 128).numpy()),
            'input_mask': numpy.ascontiguousarray(torch.tensor(batch_attention_mask).cpu().reshape(n, 128).numpy()),
            'segment_ids': numpy.ascontiguousarray(torch.tensor(batch_token_type_ids).cpu().reshape(n, 128).numpy())
        }

        print("##ort_inputs: ", ort_inputs)
        outputs = self._session.run(None, ort_inputs)
        print("##outputs: ", outputs)

        outputs = torch.nn.functional.softmax(torch.tensor(outputs[0]))
        result = [(r['id'], r['answer'], float(i[1]))
                  for i, r in zip(outputs, answers)]
        result.sort(key=lambda x: -x[2])
        return result
