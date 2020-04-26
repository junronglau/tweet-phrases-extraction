import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args

def load_data(path, kind='train'):

    """Load data from path"""
    df = pd.read_csv(path + '%.csv' % kind)
    X_train, X_test, y_train, y_test = train_test_split(df[['text','sentiment']], df['selected_text'])

    return (X_train, X_test), (y_train, y_test)

def preprocess_data(X, y, maxLen):
    df = pd.concat([X, y], axis=1).reset_index()
    inputId, attnMask ,startTokens, endTokens = get_Inputs(df,'text','sentiment',maxLen,"selected_text")
    #roBERTa needs this to be passed with values 0
    typeTokens = np.zeros((len(df),maxLen),dtype='int32')

    return [inputId, attnMask, typeTokens], [startTokens, endTokens]

def get_Inputs(df, inputCol, targetCol, maxLen, selectedCol):
    
    """Outputs token IDs,attention mask, start and end tokens if training data. Else only token IDs and attention """

    splitCol = df.apply(lambda row: (row[inputCol].split(row[selectedCol])) if not (row[selectedCol]=='') else ['',''],axis=1 )
    #Splitting the tokenizing process up seperated by selected_text col
    startTokenizer = get_TokenizerId(splitCol.str[0])
    middleTokenizer = get_TokenizerId(df[selectedCol])
    endTokenizer = get_TokenizerId(splitCol.str[1])

    startTokens, endTokens = get_posTokens(len(df),startTokenizer.map(len),middleTokenizer.map(len),maxLen)
    inputTokenizer = startTokenizer + middleTokenizer + endTokenizer
    targetTokenizer = get_TokenizerId(df[targetCol])
    inputLst = get_idTokens(inputTokenizer,targetTokenizer)
    attnLst = get_attnMasks(inputTokenizer,targetTokenizer)

    return inputLst, attnLst, startTokens, endTokens

def get_TokenizerId(textCol):
    return textCol.astype(str).map(lambda x : tokenizer.encode(x.strip()).ids if not (x.isspace() or len(x)==0) else [])

def get_posTokens(rowLen,stLen,mtLen,maxLen):

    startTokens = np.zeros((rowLen,maxLen),dtype='int32')
    startTokens[np.arange(len(startTokens)),(stLen+1).tolist()] = 1
    endTokens = np.zeros((rowLen,maxLen),dtype='int32')
    endTokens[np.arange(len(endTokens)), (stLen+mtLen).tolist() ] = 1
    
    return startTokens, endTokens

def get_idTokens(inputTokenizer,targetTokenizer):
    return np.asarray([[0] + inputId + [2,2] + targetId + [2] + [1]*(maxLen - (len(inputId+targetId)+4)) for inputId, targetId in zip(inputTokenizer,targetTokenizer)], dtype='int32')

def get_attnMasks(inputTokenizer,targetTokenizer):
    return np.asarray([[1]* (len(inputId+targetId)+4) + [0]*(maxLen - (len(inputId+targetId)+4)) for inputId, targetId in zip(inputTokenizer,targetTokenizer)], dtype='int32')

