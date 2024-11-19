import torch
import torch.nn as nn
import torch.nn.functional as F


class ComponentBasedHierarchicalAttention(nn.Module):
    def __init__(self, input_dims, component_dims, data_format='channels_last'):
        """
        Initializes the ComponentBasedHierarchicalAttention layer.

        Args:
        - input_dims (int): Number of input dimensions for the features.
        - component_dims (int): Number of dimensions for each component in the hierarchy.
        - data_format (str): Format of the data ('channels_last' or 'channels_first').
        """
        super(ComponentBasedHierarchicalAttention, self).__init__()
        self.data_format = data_format

        # Convolution layers to compute query, key, and value projections
        self.query_conv = nn.Conv2d(in_channels=input_dims, out_channels=input_dims // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=input_dims, out_channels=input_dims // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=input_dims, out_channels=input_dims, kernel_size=1)

        # Component-based attention mechanisms for hierarchical levels
        self.component_attentions = nn.ModuleList([
            nn.Conv2d(in_channels=input_dims // 8, out_channels=component_dims // 8, kernel_size=1)
            for _ in range(3)  # Number of hierarchy levels
        ])

        # Learnable scaling parameter for the attention output
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, inputs):
        """
        Forward pass for the ComponentBasedHierarchicalAttention layer.

        Args:
        - inputs (torch.Tensor): Input tensor of shape [B, C, H, W] (if channels_first) 
          or [B, H, W, C] (if channels_last).

        Returns:
        - combined_attention_output (torch.Tensor): Aggregated attention-enhanced input features.
        - attention_outputs (list[torch.Tensor]): Attention outputs at each hierarchy level.
        """
        if self.data_format == 'channels_last':
            inputs = inputs.permute(0, 3, 1, 2).contiguous()  # Convert to [B, C, H, W] for consistency

        # Compute query, key, and value projections
        query = self.query_conv(inputs)  # [B, C_query, H, W]
        key = self.key_conv(inputs)      # [B, C_key, H, W]
        value = self.value_conv(inputs)  # [B, C_value, H, W]

        attention_outputs = []

        # Apply attention at different hierarchical levels
        for component_attention in self.component_attentions:
            # Component-specific query projection
            component_query = component_attention(query)  # [B, C_component, H, W]

            # Reshape for attention computation
            batch_size, _, height, width = component_query.shape
            proj_query = component_query.view(batch_size, -1, height * width).permute(0, 2, 1)  # [B, H*W, C_component]
            proj_key = key.view(batch_size, -1, height * width)  # [B, C_key, H*W]
            proj_value = value.view(batch_size, -1, height * width).permute(0, 2, 1)  # [B, H*W, C_value]

            # Compute attention weights
            energy = torch.bmm(proj_query, proj_key.transpose(1, 2))  # [B, H*W, H*W]
            attention = F.softmax(energy, dim=-1)  # [B, H*W, H*W]

            # Compute the attention-weighted value
            attention_output = torch.bmm(attention, proj_value)  # [B, H*W, C_value]

            # Reshape back to spatial dimensions
            attention_output = attention_output.permute(0, 2, 1).contiguous().view(batch_size, -1, height, width)  # [B, C_value, H, W]
            attention_outputs.append(attention_output)

        # Aggregate hierarchical levels with learnable weight (gamma)
        combined_attention_output = torch.stack(attention_outputs, dim=0).sum(dim=0) * self.gamma + inputs

        # If input was originally channels_last, revert back to that format
        if self.data_format == 'channels_last':
            combined_attention_output = combined_attention_output.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]

        return combined_attention_output, attention_outputs
