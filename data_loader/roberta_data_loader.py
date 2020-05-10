from base.base_data_loader import BaseDataLoader
import pandas as pd
import numpy as np
import tokenizers

class RobertaDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(RobertaDataLoader, self).__init__(config)
        self.X_train, self.y_train = self.load_data(self.config.exp.data_path.train)
        self.X_train, self.y_train = self.preprocess_data(self.X_train,self.y_train,self.config.exp.max_len)

        self.X_test, self.y_test = self.load_data(self.config.exp.data_path.test)
        self.X_test, self.y_test_token = self.preprocess_data(self.X_test,self.y_test,self.config.exp.max_len)
        

    def get_tokenizer(self):
        tokenizer =  tokenizers.ByteLevelBPETokenizer(
                        vocab_file= self.config.exp.roberta_path + 'vocab-roberta-base.json', 
                        merges_file= self.config.exp.roberta_path +'merges-roberta-base.txt', 
                        lowercase=True,
                        add_prefix_space=True
                )
        return tokenizer
    
    def get_train_data(self):
        return self.X_train, self.y_train
    
    def get_test_data(self):
        return self.X_test, self.y_test
    
    def load_data(self, path):
        df = pd.read_csv(path)
        return df[['text','sentiment']], df['selected_text']

    def preprocess_data(self, X, y, maxLen):
        df = pd.concat([X, y], axis=1).reset_index()
        df = df.fillna('')
        inputId, attnMask ,startTokens, endTokens = self.get_Inputs(df,'text','sentiment',maxLen,"selected_text")
        #roBERTa needs this to be passed with values 0
        typeTokens = np.zeros((len(df),maxLen),dtype='int32')
        return [inputId, attnMask, typeTokens], [startTokens, endTokens]

    def get_Inputs(self, df, inputCol, targetCol, maxLen, selectedCol):
        """Outputs token IDs,attention mask, start and end tokens if training data. Else only token IDs and attention """
        splitCol = df.apply(lambda row: (row[inputCol].split(row[selectedCol])) if not (row[selectedCol]=='') else ['',''],axis=1 )
        #Splitting the tokenizing process up seperated by selected_text col
        startTokenizer = self.get_TokenizerId(splitCol.str[0])
        middleTokenizer = self.get_TokenizerId(df[selectedCol])
        endTokenizer = self.get_TokenizerId(splitCol.str[1])

        startTokens, endTokens = self.get_posTokens(len(df),startTokenizer.map(len),middleTokenizer.map(len),maxLen)
        inputTokenizer = startTokenizer + middleTokenizer + endTokenizer
        targetTokenizer = self.get_TokenizerId(df[targetCol])
        inputLst = self.get_idTokens(inputTokenizer,targetTokenizer,maxLen)
        attnLst = self.get_attnMasks(inputTokenizer,targetTokenizer,maxLen)

        return inputLst, attnLst, startTokens, endTokens

    def get_TokenizerId(self, textCol):
        return textCol.astype(str).map(lambda x : self.get_tokenizer().encode(x.strip()).ids if not (x.isspace() or len(x)==0) else [])

    def get_posTokens(self, rowLen,stLen,mtLen,maxLen):
        startTokens = np.zeros((rowLen,maxLen),dtype='int32')
        startTokens[np.arange(len(startTokens)),(stLen+1).tolist()] = 1
        endTokens = np.zeros((rowLen,maxLen),dtype='int32')
        endTokens[np.arange(len(endTokens)), (stLen+mtLen).tolist() ] = 1
        return startTokens, endTokens

    def get_idTokens(self, inputTokenizer,targetTokenizer,maxLen):
        return np.asarray([[0] + inputId + [2,2] + targetId + [2] + [1]*(maxLen - (len(inputId+targetId)+4)) for inputId, targetId in zip(inputTokenizer,targetTokenizer)], dtype='int32')

    def get_attnMasks(self, inputTokenizer,targetTokenizer,maxLen):
        return np.asarray([[1]* (len(inputId+targetId)+4) + [0]*(maxLen - (len(inputId+targetId)+4)) for inputId, targetId in zip(inputTokenizer,targetTokenizer)], dtype='int32')

