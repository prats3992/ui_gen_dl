import torch
import torch.nn as nn
import torch.nn.functional as F


class GridAwareAttention(nn.Module):
    def __init__(self, input_dims, grid_size=4, data_format='channels_last'):
        """
        Initializes the GridAwareAttention layer.

        Args:
        - input_dims (int): Number of input dimensions for the features.
        - grid_size (int): Size of the positional encoding grid.
        - data_format (str): Format of the data ('channels_last' or 'channels_first').
        """
        super(GridAwareAttention, self).__init__()
        self.data_format = data_format
        self.grid_size = grid_size

        # Convolution layers to compute query, key, and value projections
        self.query_conv = nn.Conv2d(in_channels=input_dims, out_channels=input_dims // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=input_dims, out_channels=input_dims // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=input_dims, out_channels=input_dims, kernel_size=1)

        # Learnable scaling parameter for the attention output
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, inputs):
        """
        Forward pass for the GridAwareAttention layer.

        Args:
        - inputs (torch.Tensor): Input tensor of shape [B, C, H, W] (if channels_first) 
          or [B, H, W, C] (if channels_last).

        Returns:
        - out (torch.Tensor): Attention-enhanced input features.
        - attention (torch.Tensor): Attention weights of shape [B, H*W, H*W].
        """
        if self.data_format == 'channels_last':
            inputs = inputs.permute(0, 3, 1, 2).contiguous()  # Convert to [B, C, H, W] for consistency

        # Extract spatial dimensions
        batch_size, _, height, width = inputs.shape

        # Create grid-aware position encoding and add to input
        position_encoding = self._create_position_encoding(height, width, self.grid_size).to(inputs.device)
        position_encoded_input = inputs + position_encoding

        # Compute query, key, and value projections
        query = self.query_conv(position_encoded_input)  # [B, C_query, H, W]
        key = self.key_conv(position_encoded_input)      # [B, C_key, H, W]
        value = self.value_conv(position_encoded_input)  # [B, C_value, H, W]

        # Reshape and prepare for attention computation
        proj_query = query.view(batch_size, -1, height * width).permute(0, 2, 1)  # [B, H*W, C_query]
        proj_key = key.view(batch_size, -1, height * width).permute(0, 2, 1)      # [B, H*W, C_key]
        proj_value = value.view(batch_size, -1, height * width).permute(0, 2, 1)  # [B, H*W, C_value]

        # Compute attention weights
        energy = torch.bmm(proj_query, proj_key.transpose(1, 2))  # [B, H*W, H*W]
        attention = F.softmax(energy, dim=-1)  # [B, H*W, H*W]

        # Compute the attention-weighted value
        out = torch.bmm(attention, proj_value)  # [B, H*W, C_value]

        # Reshape output back to original spatial dimensions
        out = out.permute(0, 2, 1).contiguous().view(batch_size, -1, height, width)  # [B, C_value, H, W]

        # If input was originally channels_last, revert back to that format
        if self.data_format == 'channels_last':
            out = out.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]

        # Scale the attention output and add it to the input
        return self.gamma * out + inputs, attention

    def _create_position_encoding(self, height, width, grid_size):
        """
        Creates a grid-aware position encoding.

        Args:
        - height (int): Height of the input feature map.
        - width (int): Width of the input feature map.
        - grid_size (int): Size of the positional encoding grid.

        Returns:
        - position_encoding (torch.Tensor): Tensor of shape [1, 2, H, W] (channels_first).
        """
        # Generate positional coordinates normalized to [-1, 1]
        y_embed = torch.linspace(-1.0, 1.0, steps=height).unsqueeze(1)  # [H, 1]
        x_embed = torch.linspace(-1.0, 1.0, steps=width).unsqueeze(0)   # [1, W]
        y_embed, x_embed = torch.meshgrid(y_embed, x_embed, indexing="ij")  # [H, W], [H, W]

        # Stack and resize to the grid size
        grid = torch.stack([y_embed, x_embed], dim=0).unsqueeze(0)  # [1, 2, H, W]
        grid = F.interpolate(grid, size=(grid_size, grid_size), mode='bilinear', align_corners=False)

        # Tile the grid to match the input dimensions
        grid = F.interpolate(grid, size=(height, width), mode='bilinear', align_corners=False)
        return grid
