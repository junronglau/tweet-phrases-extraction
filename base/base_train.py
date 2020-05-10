class BaseTrain(object):
    def __init__(self, model, data, config, experiment):
        self.model = model
        self.data = data
        self.config = config
        self.experiment = experiment

    def train(self):
        raise NotImplementedError