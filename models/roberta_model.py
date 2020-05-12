from base.base_model import BaseModel
from utils.metric import jaccard
import tensorflow as tf
from transformers import TFRobertaModel, RobertaConfig
import keras.backend as K
import keras.initializers

class RobertaModel(BaseModel):
    def __init__(self, config):
        super(RobertaModel, self).__init__(config)
        self.build_model()

    def build_model(self):
        ids = tf.keras.layers.Input((self.config.exp.max_len,), dtype=tf.int32)
        att = tf.keras.layers.Input((self.config.exp.max_len,), dtype=tf.int32)
        tok = tf.keras.layers.Input((self.config.exp.max_len,), dtype=tf.int32)
        
        # Network architecture
        config = RobertaConfig.from_pretrained(self.config.exp.roberta_path +'config-roberta-base.json') 
        bert_model = TFRobertaModel.from_pretrained(self.config.exp.roberta_path +'pretrained-roberta-base.h5',config=config)
        x = bert_model(ids,attention_mask=att,token_type_ids=tok)

        x1 = tf.keras.layers.Dropout(0.1)(x[0])
        x1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,return_sequences=True))(x1)
        x1 = tf.keras.layers.Dropout(0.1)(x1) 
        x1 = tf.keras.layers.Dense(1)(x1)
        x1 = tf.keras.layers.Flatten()(x1)
        x1 = tf.keras.layers.Activation('softmax')(x1)

        x2 = tf.keras.layers.Dropout(0.1)(x[0])
        x2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,return_sequences=True))(x2)  
        x2 = tf.keras.layers.Dropout(0.1)(x2)  
        x2 = tf.keras.layers.Dense(1)(x2)
        x2 = tf.keras.layers.Flatten()(x2)
        x2 = tf.keras.layers.Activation('softmax')(x2)

        self.model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1,x2])
        self.model.compile(loss='categorical_crossentropy', optimizer=self.config.model.optimizer)

    def load_weights(self):
        self.model.load_weights(self.config.exp.roberta_path + self.config.exp.roberta_weights)
