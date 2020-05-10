from base.base_train import BaseTrain
from utils.metric import jaccard
import numpy as np
import os

class RobertaTrainer(BaseTrain):
    def __init__(self, model, data, config, experiment):
        super(RobertaTrainer, self).__init__(model, data, config, experiment)
        self.init_callbacks()

    def init_callbacks(self):
        if self.experiment:
            parameters = {
                "learning_rate" : self.config.model.learning_rate,
                "optimizer" : self.config.model.optimizer,
                "num_epochs": self.config.trainer.num_epochs,
                "batch_size": self.config.trainer.batch_size,
                "validation_split": self.config.trainer.validation_split,
                "verbose_training": self.config.trainer.verbose_training,
            }
            self.experiment.log_parameters(parameters)
            self.experiment.disable_mp()
            self.experiment.get_callback('keras')

    def train(self):
        self.model.fit(
            self.data[0], self.data[1],
            epochs=self.config.trainer.num_epochs,
            verbose=self.config.trainer.verbose_training,
            batch_size=self.config.trainer.batch_size,
            validation_split=self.config.trainer.validation_split
        )

    def predict(self,test_data,tokenizer):
        """
        test_data is in the following format: [id, att, tok], [text] , where [text] is the actual labelled text
        """
        preds = self.model.predict(test_data[0])
        preds_start = np.argmax(preds[0],axis=1)
        preds_end = np.argmax(preds[1],axis=1)
        pred_text = [tokenizer.decode(test_data[0][0][i]) if preds_start[i] > preds_end[i] else tokenizer.decode(test_data[0][0][i][preds_start[i]-1:preds_end[i]]) for i in range(len(test_data))]     
        jaccard_score = np.mean([jaccard(pred_text[i],test_data[1][i]) for i in range(len(test_data))])
        if self.experiment:
            metrics = {
                'Jaccard similarity': jaccard_score
            }
            self.experiment.log_metrics(metrics)
            
        print("Jaccard score:",jaccard_score)