class ReplayBuffer:
    def __init__(self, **kwargs):
        self.buffer = set()

    def add_episode(self, episode):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def sample(self, batch_size):
        raise NotImplementedError("This method should be overridden by subclasses.")