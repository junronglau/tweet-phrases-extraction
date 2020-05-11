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
        x1 = highway_layers(x1,2)
        x1 = tf.keras.layers.Dense(1)(x1)
        x1 = tf.keras.layers.Flatten()(x1)
        x1 = tf.keras.layers.Activation('softmax')(x1)

        x2 = tf.keras.layers.Dropout(0.1)(x[0])
        x2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,return_sequences=True))(x2)  
        x2 = tf.keras.layers.Dropout(0.1)(x2) 
        x2 = highway_layers(x2,2) 
        x2 = tf.keras.layers.Dense(1)(x2)
        x2 = tf.keras.layers.Flatten()(x2)
        x2 = tf.keras.layers.Activation('softmax')(x2)

        self.model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1,x2])
        self.model.compile(loss='categorical_crossentropy', optimizer=self.config.model.optimizer)

    def load_weights(self):
        self.model.load_weights(self.config.exp.roberta_path + self.config.exp.roberta_weights)

def highway_layers(value, n_layers, activation="relu", gate_bias=-3):
    dim = K.int_shape(value)[-1]
    gate_bias_initializer = keras.initializers.Constant(gate_bias)
    for i in range(n_layers):     
        gate = tf.keras.layers.Dense(units=dim, bias_initializer=gate_bias_initializer)(value)
        gate = tf.keras.layers.Activation("sigmoid")(gate)
        negated_gate = tf.keras.layers.Lambda(
            lambda x: 1.0 - x,
            output_shape=(dim,))(gate)
        transformed = tf.keras.layers.Dense(units=dim)(value)
        transformed = tf.keras.layers.Activation(activation)(value)
        transformed_gated = tf.keras.layers.Multiply()([gate, transformed])
        identity_gated = tf.keras.layers.Multiply()([negated_gate, value])
        value = tf.keras.layers.Add()([transformed_gated, identity_gated])
    return value