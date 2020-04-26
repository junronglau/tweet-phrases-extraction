from base.base_data_loader import BaseDataLoader
from utils import load_data, preprocess_data

class RobertaDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(RobertaDataLoader, self).__init__(config)
        (self.X_train, self.y_train), (self.X_test, self.y_test) = load_data(self.config.data_path)

    def get_train_data(self):
        return preprocess_data(self.X_train,self.y_train,self.config.maxLen)

    def get_test_data(self):
        return preprocess_data(self.X_test,self.y_test,self.config.maxLen)