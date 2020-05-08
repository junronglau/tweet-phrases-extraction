import argparse
import pandas as pd
import numpy as np
from transformers import RobertaTokenizer

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        dest='config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args

def load_data(path):

    """Load data from path"""
    df = pd.read_csv(path)

    return df[['text','sentiment']], df['selected_text']

def preprocess_data(X, y, maxLen):
    df = pd.concat([X, y], axis=1).reset_index()
    df = df.fillna('')
    inputId, attnMask ,startTokens, endTokens = get_Inputs(df,'text','sentiment',maxLen,"selected_text")
    #roBERTa needs this to be passed with values 0
    typeTokens = np.zeros((len(df),maxLen),dtype='int32')

    return [inputId, attnMask, typeTokens], [startTokens, endTokens]

def get_Inputs(df, inputCol, targetCol, maxLen, selectedCol):
    
    """Outputs token IDs,attention mask, start and end tokens if training data. Else only token IDs and attention """
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    splitCol = df.apply(lambda row: (row[inputCol].split(row[selectedCol])) if not (row[selectedCol]=='') else ['',''],axis=1 )
    #Splitting the tokenizing process up seperated by selected_text col
    startTokenizer = get_TokenizerId(splitCol.str[0],tokenizer)
    middleTokenizer = get_TokenizerId(df[selectedCol],tokenizer)
    endTokenizer = get_TokenizerId(splitCol.str[1],tokenizer)

    startTokens, endTokens = get_posTokens(len(df),startTokenizer.map(len),middleTokenizer.map(len),maxLen)
    inputTokenizer = startTokenizer + middleTokenizer + endTokenizer
    targetTokenizer = get_TokenizerId(df[targetCol],tokenizer)
    inputLst = get_idTokens(inputTokenizer,targetTokenizer,maxLen)
    attnLst = get_attnMasks(inputTokenizer,targetTokenizer,maxLen)

    return inputLst, attnLst, startTokens, endTokens

def get_TokenizerId(textCol,tokenizer):
    return textCol.astype(str).map(lambda x : tokenizer.encode(x.strip()) if not (x.isspace() or len(x)==0) else [])

def get_posTokens(rowLen,stLen,mtLen,maxLen):
    startTokens = np.zeros((rowLen,maxLen),dtype='int32')
    startTokens[np.arange(len(startTokens)),(stLen+1).tolist()] = 1
    endTokens = np.zeros((rowLen,maxLen),dtype='int32')
    endTokens[np.arange(len(endTokens)), (stLen+mtLen).tolist() ] = 1
    
    return startTokens, endTokens

def get_idTokens(inputTokenizer,targetTokenizer,maxLen):
    return np.asarray([[0] + inputId + [2,2] + targetId + [2] + [1]*(maxLen - (len(inputId+targetId)+4)) for inputId, targetId in zip(inputTokenizer,targetTokenizer)], dtype='int32')

def get_attnMasks(inputTokenizer,targetTokenizer,maxLen):
    return np.asarray([[1]* (len(inputId+targetId)+4) + [0]*(maxLen - (len(inputId+targetId)+4)) for inputId, targetId in zip(inputTokenizer,targetTokenizer)], dtype='int32')

