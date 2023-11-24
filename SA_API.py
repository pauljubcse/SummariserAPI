from fastapi import FastAPI
import asyncio
import json
import pandas as pd
import numpy as np
import textwrap
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5ForConditionalGeneration, T5Tokenizer
import time
nltk.download('stopwords')
nltk.download('punkt')

from pydantic import BaseModel
class Text(BaseModel):
    text: str
    mode: int | None = 0

class ExtractiveTextSummariser:
    def __init__(self):
        print("RUNNING")
        self.mode="Extractive"
        self.featurizer = TfidfVectorizer(stop_words=stopwords.words('english'), norm='l2')
        self.result=""
    
    @staticmethod
    def wrap(x):
        return textwrap.fill(x, replace_whitespace=False, fix_sentence_endings=True)

    def summarise(self, text, reduction=0.33):
        start=time.time()
        self.result=""
        sents = nltk.sent_tokenize(text)
        #for i in sents:
            #print(i)
        n=len(sents)
        #print("Sentences: ", n)
        X = self.featurizer.fit_transform(sents)
        S = cosine_similarity(X)
        S /= S.sum(axis=1, keepdims=True)
        U = np.ones_like(S) / len(S)
        factor = 0.15
        S = (1 - factor) * S + factor * U
        eigenvals, eigenvecs = np.linalg.eig(S.T) #Transpose
        limiting_distribution=eigenvecs[:,0] / eigenvecs[:,0].sum()
        scores=limiting_distribution
        sort_idx = np.argsort(-scores)
        count=int(n*reduction+1)
        #print("Selecting: ",count)
        #print("Generated summary:")
        '''
        for i in sort_idx[:count]:
            print(wrap("%.2f: %s" % (scores[i], sents[i])))
        '''
        for i in sort_idx[:count]:
            self.result+=sents[i]
        end=time.time()
        return {"mode":self.mode, "result":self.result, "delay": end-start}
'''  
class AbstractiveTextSummariser:
    def __init__(self):
        self.mode="Abstractive"
        self.directory="SummariserModel\SummariserModel"
        self.model = T5ForConditionalGeneration.from_pretrained(self.directory)
        self.tokenizer = T5Tokenizer.from_pretrained(self.directory)
        self.result=""

    def summarise(self, text, reduction=0.33):
        # Tokenize and summarize the text
        start=time.time()
        inputs = self.tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = self.model.generate(inputs, max_length=300, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)

        # Decode the summary
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        self.result=summary
        end=time.time()
        return {"mode":self.mode, "result":self.result, "delay": end-start}
    
'''

ET=ExtractiveTextSummariser()

app = FastAPI()

@app.get("/test")
def test():
    return {"Status":"Running"}

@app.post("/get-summary")
def create_text(text: Text):
    return ET.summarise(text.text)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8080)