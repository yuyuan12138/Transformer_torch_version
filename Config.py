import numpy as np


class Config:
    def __init__(self):
        # model Hyper-parameter
        self.d_model = 512
        self.N = 6  # It means there are N identical layers for encoders and decoders.
        self.h = 8  # There are 8 parts of multi-head-attention.
        self.d_q = self.d_k = self.d_v = self.d_model // self.h
        self.d_ff = 2048  # It is the dimension of hidden layers.
        self.P_drop = 0.1   # Drop_rate

        # Optimizer Hyper-parameter
        self.beta_1 = 0.9
        self.beta_2 = 0.98
        self.epsilon = 1e-9

    def learning_rate(self, step_num: int, warmup_steps: int = 4000) -> float:
        return 1 / np.sqrt(self.d_model) * min(1 / np.sqrt(step_num),
                                               step_num * 1 / (warmup_steps * np.sqrt(warmup_steps)))
