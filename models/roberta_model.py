from base.base_model import BaseModel
from utils.metric import jaccard
import tensorflow as tf
from transformers import TFRobertaModel, RobertaConfig

class RobertaModel(BaseModel):
    def __init__(self, config):
        super(RobertaModel, self).__init__(config)
        self.build_model()

    def build_model(self):
        ids = tf.keras.layers.Input((self.config.data.roberta.max_len,), dtype=tf.int32)
        att = tf.keras.layers.Input((self.config.data.roberta.max_len,), dtype=tf.int32)
        tok = tf.keras.layers.Input((self.config.data.roberta.max_len,), dtype=tf.int32)
        
        # Network architecture
        config = RobertaConfig.from_pretrained(self.config.data.roberta.path + self.config.data.roberta.config) 
        bert_model = TFRobertaModel.from_pretrained(self.config.data.roberta.path + self.config.data.roberta.roberta_weights ,config=config)
        x = bert_model(ids,attention_mask=att,token_type_ids=tok)

        self.init_head(x[0])
        self.add_dropout(0.1)
        self.add_lstm(64,True)
        self.add_dropout(0.1)
        self.add_dense(1)
        self.add_activation('softmax')
        self.model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[self.start_head,self.end_head])
        self.model.compile(loss=self.config.model.loss, optimizer=self.config.model.optimizer)
    
    def init_head(self, output_state):
        self.start_head = output_state
        self.end_head = output_state

    def add_dropout(self, rate):
        self.start_head =  tf.keras.layers.Dropout(rate)(self.start_head)
        self.end_head =  tf.keras.layers.Dropout(rate)(self.end_head)
    
    def add_lstm(self, nodes, return_sequences):
        self.start_head = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(nodes,return_sequences=return_sequences))(self.start_head)
        self.end_head = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(nodes,return_sequences=return_sequences))(self.end_head)

    def add_dense(self, nodes):
        self.start_head = tf.keras.layers.Dense(nodes)(self.start_head)
        self.end_head = tf.keras.layers.Dense(nodes)(self.end_head)

    def add_activation(self,activation,flatten=True):
        if flatten:
            self.start_head = tf.keras.layers.Flatten()(self.start_head)
            self.end_head = tf.keras.layers.Flatten()(self.end_head)
        self.start_head = tf.keras.layers.Activation(activation)(self.start_head)
        self.end_head = tf.keras.layers.Activation(activation)(self.end_head)
