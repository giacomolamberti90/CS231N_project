import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
from torch.nn.utils.rnn import pack_padded_sequence


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = models.alexnet(pretrained=True)

    def forward(self, x):
        x = self.conv.features(x)
        return x


class Combine(nn.Module):
    def __init__(self):
        super(Combine, self).__init__()
        self.cnn = CNN()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.rnn = nn.LSTM(input_size=256, hidden_size=256, num_layers=1,
                           batch_first=True)  # Note that "batch_first" is set to "True"
        self.hidden_size = 256
        self.use_gpu = True
        self.linear = nn.Linear(256 * 3, 2)  # 3 planes, 3 tasks

    def forward(self, batch):
        x_axial, x_coronal, x_sagittal, labels = batch

        batch_size, timesteps_axial, C_axial, H_axial, W_axial = x_axial.size()
        x_axial = x_axial.view(batch_size * timesteps_axial, C_axial, H_axial, W_axial)
        if self.use_gpu:
            x_axial = x_axial.cuda()
        x_axial = self.cnn(x_axial)
        x_axial = self.gap(x_axial)
        x_axial = x_axial.view(batch_size, timesteps_axial, -1)
        #         x_axial = pack_padded_sequence(x_axial, lengths_axial, batch_first=True)
        output_axial, _ = self.rnn(x_axial)
        #         output_axial, _ = torch.nn.utils.rnn.pad_packed_sequence(output_axial, batch_first=False)
        output_axial = output_axial[:, -1, :]
        output_axial = output_axial.view(batch_size, self.hidden_size)

        batch_size, timesteps_coronal, C_coronal, H_coronal, W_coronal = x_coronal.size()
        x_coronal = x_coronal.view(batch_size * timesteps_coronal, C_coronal, H_coronal, W_coronal)
        if self.use_gpu:
            x_coronal = x_coronal.cuda()
        x_coronal = self.cnn(x_coronal)
        x_coronal = self.gap(x_coronal)
        x_coronal = x_coronal.view(batch_size, timesteps_coronal, -1)
        #         x_coronal = pack_padded_sequence(x_coronal, lengths_coronal, batch_first=True)
        output_coronal, _ = self.rnn(x_coronal)
        #         output_coronal, _ = torch.nn.utils.rnn.pad_packed_sequence(output_coronal, batch_first=False)
        output_coronal = output_coronal[:, -1, :]
        output_coronal = output_coronal.view(batch_size, self.hidden_size)

        batch_size, timesteps_sagittal, C_sagittal, H_sagittal, W_sagittal = x_sagittal.size()
        x_sagittal = x_sagittal.view(batch_size * timesteps_sagittal, C_sagittal, H_sagittal, W_sagittal)
        if self.use_gpu:
            x_sagittal = x_sagittal.cuda()
        x_sagittal = self.cnn(x_sagittal)
        x_sagittal = self.gap(x_sagittal)
        x_sagittal = x_sagittal.view(batch_size, timesteps_sagittal, -1)
        #         x_sagittal = pack_padded_sequence(x_sagittal, lengths_sagittal, batch_first=True)
        output_sagittal, _ = self.rnn(x_sagittal)
        #         output_sagittal, _ = torch.nn.utils.rnn.pad_packed_sequence(output_sagittal, batch_first=False)
        output_sagittal = output_sagittal[:, -1, :]
        output_sagittal = output_sagittal.view(batch_size, self.hidden_size)

        output = torch.cat((output_axial, output_coronal, output_sagittal), 1)
        output = self.linear(output)
        output = output.view(-1, 2)
        return output