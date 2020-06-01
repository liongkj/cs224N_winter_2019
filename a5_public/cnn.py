#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


### YOUR CODE HERE for part 1i
class CNN(nn.Module):
    def __init__(self, m_word: int = 21, embed_size=50, out_channels=None, k=5, padding=0, stride=1):
        """

        Args:
            stride (int): stride for conv1d
            k (int): Kernel size / windows size for conv1d
            out_channels (int): Filters / output features/ output channels
        """
        super(CNN, self).__init__()
        self.k = k
        self.embed_size = embed_size
        self.padding = padding
        self.stride = stride
        self.f = out_channels

        self.conv1d = nn.Conv1d(in_channels=self.embed_size, out_channels=self.f,
                                kernel_size=self.k, stride=self.stride, padding=self.padding)
        self.maxpool = nn.MaxPool1d(m_word - self.k + 1,)

    def forward(self, x_reshaped):
        """

        Args:
            x_reshaped (tensor): shape (max_sentence_length,
                                    batch_size, e_char, m_word)
            return x_conv_out (tensor): shape (max_sentence_length, batch_size)
        """
        x_conv = self.conv1d(x_reshaped)
        x_conv_out = self.maxpool(F.relu(x_conv))

        return x_conv_out.squeeze(-1)

### END YOUR CODE
