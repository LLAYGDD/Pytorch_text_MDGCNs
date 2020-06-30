

class CONFIG(object):
    """docstring for CONFIG"""
    def __init__(self):
        super(CONFIG, self).__init__()
        
        self.dataset = '20ng'

        self.learning_rate = 0.02   # Initial learning rate.
        self.epochs  = 2  # Number of epochs to train.

        self.dropout = 0.5  # Dropout rate (1 - keep probability).
        self.weight_decay = 0.   # Weight for L2 loss on embedding matrix.
        self.early_stopping = 10 # Tolerance for early stopping (# of epochs).
        # self.max_degree = 2      # Maximum Chebyshev polynomial degree.



