#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Highway(nn.Module):
### YOUR CODE HERE for part 1h
    def __init__(self, hidden_size):
        super(Highway, self).__init__()
        self.hidden_size = hidden_size

        self.X_projection = nn.Linear(self.hidden_size, self.hidden_size)
        self.X_gate = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, x_conv_out):
        """
        Args:
            x_conv_out (Tensor):  shape of input tensor [batch_size,1,e_word]

            @returns x_highway (Tensor): shape of
        """
        x_projection = F.relu(self.X_projection(x_conv_out))
        # print(x_projection.size())
        x_gate = torch.sigmoid(self.X_gate(x_conv_out))
        # print(x_gate.size())
        x_highway = torch.mul(x_projection, x_gate) + torch.mul((1-x_gate), x_conv_out)
        return x_highway
### END YOUR CODE 

