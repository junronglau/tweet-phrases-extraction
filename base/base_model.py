class BaseModel(object):
    def __init__(self, config):
        self.config = config
        self.model = None

    # save function that saves the checkpoint in the path defined in the config file
    def save(self):
        if self.model is None:
            raise Exception("You have to build the model first.")

        print("Saving model...")
        self.model.save_weights(self.config.model.weights)
        print("Model saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self):
        if self.model is None:
            raise Exception("You have to build the model first.")

        print("Loading model checkpoint {} ...\n".format(self.config.model.weights))
        self.model.load_weights(self.config.model.weights)
        print("Model loaded")

    def build_model(self):
        raise NotImplementedError
