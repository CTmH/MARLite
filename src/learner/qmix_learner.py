from .learner import Learner

class QMixLearner(Learner):
    def __init__(self, env_config, model_configs):
        super().__init__(env_config, model_configs)

    def learn(self):
        return super().learn()