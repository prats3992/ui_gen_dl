import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, input_dims, output_dims, data_format='channels_last'):
        """
        Initializes the CrossAttention layer.

        Args:
        - input_dims (int): Number of input dimensions for target and source features.
        - output_dims (int): Number of output dimensions after applying value convolution.
        - data_format (str): Format of the data ('channels_last' or 'channels_first').
        """
        super(CrossAttention, self).__init__()
        self.data_format = data_format

        # Convolution layers to compute query, key, and value projections
        self.query_conv = nn.Conv2d(in_channels=input_dims, out_channels=input_dims // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=input_dims, out_channels=input_dims // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=input_dims, out_channels=output_dims, kernel_size=1)

        # Learnable scaling parameter for the attention output
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, target, source):
        """
        Forward pass for the CrossAttention layer.

        Args:
        - target (torch.Tensor): Target features of shape [B, C, H, W] (if channels_first) 
          or [B, H, W, C] (if channels_last).
        - source (torch.Tensor): Source features of the same shape as target.

        Returns:
        - out (torch.Tensor): Attention-enhanced target features.
        - attention (torch.Tensor): Attention weights of shape [B, H*W, H*W].
        """
        # Compute query, key, and value projections
        query = self.query_conv(target)  # [B, C_query, H, W]
        key = self.key_conv(source)      # [B, C_key, H, W]
        value = self.value_conv(source)  # [B, C_value, H, W]

        # Reshape and prepare for attention computation
        if self.data_format == 'channels_first':
            batch_size, _, height, width = query.shape
            # Flatten spatial dimensions into a single dimension
            proj_query = query.view(batch_size, -1, height * width)  # [B, C_query, H*W]
            proj_key = key.view(batch_size, -1, height * width)      # [B, C_key, H*W]
            proj_value = value.view(batch_size, -1, height * width)  # [B, C_value, H*W]
        else:
            batch_size, height, width, channels = query.shape
            # Permute to [B, C, H, W] and then flatten spatial dimensions
            proj_query = query.permute(0, 3, 1, 2).contiguous().view(batch_size, -1, height * width)
            proj_key = key.permute(0, 3, 1, 2).contiguous().view(batch_size, -1, height * width)
            proj_value = value.permute(0, 3, 1, 2).contiguous().view(batch_size, -1, height * width)

        # Compute attention weights
        # Energy: Dot product between query and key
        energy = torch.bmm(proj_query.transpose(1, 2), proj_key)  # [B, H*W, H*W]
        # Apply softmax to normalize attention scores
        attention = F.softmax(energy, dim=-1)  # [B, H*W, H*W]

        # Compute the attention-weighted value
        out = torch.bmm(attention, proj_value.transpose(1, 2))  # [B, H*W, C_value]

        # Reshape output back to original spatial dimensions
        if self.data_format == 'channels_first':
            out = out.view(batch_size, -1, height, width)  # [B, C_value, H, W]
        else:
            out = out.view(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()  # [B, H, W, C_value]

        # Scale the attention output and add it to the target features
        return self.gamma * out + target, attention
