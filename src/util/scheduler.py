import math

class Scheduler:
    def __init__(self, start_value, end_value, decay_steps, type='linear'):
        self.start_value = start_value
        self.end_value = end_value
        self.decay_steps = decay_steps
        self.adjustment_type = type

    def get_value(self, epoch):
        if epoch < 0 or epoch >= self.decay_steps:
            raise ValueError("Epoch out of range")

        if self.adjustment_type == 'linear':
            return self.linear_adjust(epoch)
        elif self.adjustment_type == 'logarithmic':
            return self.logarithmic_adjust(epoch)
        elif self.adjustment_type == 'fixed':
            return self.start_value
        else:
            raise ValueError("Unsupported adjustment type")

    def linear_adjust(self, epoch):
        return self.start_value + (self.end_value - self.start_value) * epoch / (self.decay_steps - 1)

    def logarithmic_adjust(self, epoch):
        
        if self.start_value <= 0 or self.end_value <= 0:
            raise ValueError("Start and end values must be positive for logarithmic adjustment")
        #return self.start_value * (math.log(self.end_value / self.start_value) / math.log(self.epochs - 1)) ** epoch
        return self.start_value * (self.end_value / self.start_value) ** (epoch / self.decay_steps)
