from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge 
from bert_score import score
import re
import io
import sys
import time
import json
import pandas as pd
import os
import cloudpickle as pickle
from datasets import load_dataset
def save_to_pickle(obj, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(obj, file)

def load_from_pickle(filepath):
    with open(filepath, 'rb') as file:
        return pickle.load(file)

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

def Average(lst): 
    return sum(lst) / len(lst)

class Evaluator:

    def __init__(self):
        self.str_list = []
        
    def set_strs_list(self, str_list):
        self.str_list = str_list
        
    def _SentenceBLEU(self,str_0, str_1, verbose=False):
        points = sentence_bleu(str_0, str_1)
        if verbose:
            print(f"BLEU score: {points}")
        return points
    
    def _RogueScore(self,str_0, str_1, verbose=False):
        rouge = Rouge()
        scores= rouge.get_scores(str_0, str_1)[0]['rouge-l']
        if verbose:
            print(f"ROUGE scores: {scores}")
        return scores
    
    def _BERTScore(self,str_0, str_1, verbose=False):
        P, R, F1 = score([str_0], [str_1], lang='en', verbose=False)
        if verbose:
            print(f"BERTScore: Precision: {P.mean()}, Recall: {R.mean()}, F1: {F1.mean()}")
        return P,R,F1
    
    def extract_response(self,text):
        match = re.search(r"### Response:\s*(.*)", text, re.DOTALL)
        return match.group(1).strip() if match else "No response found."
    
    def PerformEval(self, verbose):
        str_0 = self.str_list[0]
        str_1 = self.str_list[1]
        return self._SentenceBLEU(str_0, str_1,verbose), self._RogueScore(str_0, str_1,verbose),self._BERTScore(str_0, str_1,verbose)
            
            