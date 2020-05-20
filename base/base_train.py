class BaseTrain(object):
    def __init__(self, model, experiment, config):
        self.model = model
        self.config = config
        self.experiment = experiment

    def train(self):
        raise NotImplementedError