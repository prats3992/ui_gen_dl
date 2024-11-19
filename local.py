import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import unfold, fold


class LocalAttention(nn.Module):
    def __init__(self, input_dims, window_size=5, data_format='channels_last'):
        """
        Initializes the LocalAttention layer.

        Args:
        - input_dims (int): Number of input dimensions for the features.
        - window_size (int): Size of the local attention window.
        - data_format (str): Format of the data ('channels_last' or 'channels_first').
        """
        super(LocalAttention, self).__init__()
        self.data_format = data_format
        self.window_size = window_size

        # Convolution layers to compute query, key, and value projections
        self.query_conv = nn.Conv2d(in_channels=input_dims, out_channels=input_dims // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=input_dims, out_channels=input_dims // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=input_dims, out_channels=input_dims, kernel_size=1)

        # Learnable scaling parameter for the attention output
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, inputs):
        """
        Forward pass for the LocalAttention layer.

        Args:
        - inputs (torch.Tensor): Input tensor of shape [B, C, H, W] (if channels_first) 
          or [B, H, W, C] (if channels_last).

        Returns:
        - out (torch.Tensor): Attention-enhanced input features.
        - attention_weights (torch.Tensor): Attention weights for local patches.
        """
        if self.data_format == 'channels_last':
            inputs = inputs.permute(0, 3, 1, 2).contiguous()  # Convert to [B, C, H, W] for consistency

        batch_size, channels, height, width = inputs.shape

        # Compute query, key, and value projections
        query = self.query_conv(inputs)  # [B, C_query, H, W]
        key = self.key_conv(inputs)      # [B, C_key, H, W]
        value = self.value_conv(inputs)  # [B, C_value, H, W]

        # Extract patches using unfold
        patch_dim = self.window_size * self.window_size
        query_patches = unfold(query, kernel_size=self.window_size, stride=1, padding=self.window_size // 2)  # [B, C_query * W^2, H*W]
        key_patches = unfold(key, kernel_size=self.window_size, stride=1, padding=self.window_size // 2)      # [B, C_key * W^2, H*W]
        value_patches = unfold(value, kernel_size=self.window_size, stride=1, padding=self.window_size // 2)  # [B, C_value * W^2, H*W]

        # Reshape patches to group spatial dimensions
        query_patches = query_patches.permute(0, 2, 1).reshape(batch_size, height, width, patch_dim, -1)  # [B, H, W, W^2, C_query]
        key_patches = key_patches.permute(0, 2, 1).reshape(batch_size, height, width, patch_dim, -1)      # [B, H, W, W^2, C_key]
        value_patches = value_patches.permute(0, 2, 1).reshape(batch_size, height, width, patch_dim, -1)  # [B, H, W, W^2, C_value]

        # Local attention computation within each patch
        attention_logits = torch.einsum('bhwnc,bhwmc->bhwnm', query_patches, key_patches)  # [B, H, W, W^2, W^2]
        attention_weights = F.softmax(attention_logits, dim=-1)  # Normalize over the window dimension

        # Compute weighted sum of value patches
        local_out = torch.einsum('bhwnm,bhwmc->bhwnc', attention_weights, value_patches)  # [B, H, W, W^2, C_value]
        local_out = local_out.reshape(batch_size, height * width, -1).permute(0, 2, 1)  # [B, C_value, H*W]
        local_out = fold(local_out, output_size=(height, width), kernel_size=self.window_size, padding=self.window_size // 2)  # [B, C_value, H, W]

        # Scale and add to input
        out = self.gamma * local_out + inputs

        # If input was originally channels_last, revert back to that format
        if self.data_format == 'channels_last':
            out = out.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]

        return out, attention_weights
