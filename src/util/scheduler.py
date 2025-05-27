import math

class Scheduler:
    def __init__(self, start_value, end_value, decay_steps, type='linear'):
        self.start_value = start_value
        self.end_value = end_value
        self.decay_steps = decay_steps
        self.adjustment_type = type

    def get_value(self, epoch):
        t = epoch
        if t < 0:
            raise ValueError("Epoch out of range")
        elif t >= self.decay_steps:
            t = self.decay_steps - 1

        if self.adjustment_type == 'linear':
            return self.linear_adjust(t)
        elif self.adjustment_type == 'logarithmic':
            return self.logarithmic_adjust(t)
        elif self.adjustment_type == 'fixed':
            return self.start_value
        else:
            raise ValueError("Unsupported adjustment type")

    def linear_adjust(self, t):
        return self.start_value + (self.end_value - self.start_value) * t / self.decay_steps

    def logarithmic_adjust(self, t):
        
        if self.start_value <= 0 or self.end_value <= 0:
            raise ValueError("Start and end values must be positive for logarithmic adjustment")
        #return self.start_value * (math.log(self.end_value / self.start_value) / math.log(self.epochs - 1)) ** epoch
        return self.start_value * (self.end_value / self.start_value) ** (t / self.decay_steps)
