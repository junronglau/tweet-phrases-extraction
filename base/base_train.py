class BaseTrain(object):
    def __init__(self, model, experiment, config, data=None):
        self.model = model
        self.data = data
        self.config = config
        self.experiment = experiment

    def train(self):
        raise NotImplementedError