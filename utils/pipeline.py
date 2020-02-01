

class Pipeline:
    def __init__(self):
        pass

    def loadData(self):
        pass

    def setTarget(self, target):
        self.target_name = target
        self.target = self.data[target].copy()
        self.y = self.target

    def preprocess(self):
        pass

    def train(self, model):
        self.model = model
        pass
