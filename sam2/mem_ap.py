import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.modeling.sam.transformer import TwoWayAttentionBlock
from types import MethodType
import math
from sam2.modeling.mem_short_long import MemoryCompressor

###################################
##ping Channel attention
# class ChannelAttention(nn.Module):
#     def __init__(self, in_channels, reduction=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Output size = (B, C, 1, 1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)  # Optional for stronger signal

#         # Shared MLP
#         self.fc = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
#         )
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         # Apply average and max pooling
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x))
        
#         # Combine and apply sigmoid
#         out = avg_out + max_out
#         attention = self.sigmoid(out)
        
#         return x * attention  # Apply attention to input feature map

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 1D since features are seq_len x channels
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, L, C]
        B, L, C = x.size()
        y = self.avg_pool(x.transpose(1, 2))  # [B, C, 1]
        y = self.fc(y.squeeze(-1)).unsqueeze(1)  # [B, 1, C]
        return x * y.expand_as(x)


# from SAM-Med2D:b
class Adapter_CA(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=0.25, norm_layer = nn.LayerNorm, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        hidden_dim = int(embed_dim * mlp_ratio)
        self.norm = norm_layer(embed_dim)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim, bias=False),
                nn.ReLU(),
                nn.Linear(hidden_dim, embed_dim, bias=False),
                nn.Sigmoid()
        )

        self.spatial = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
                nn.ReLU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=4, stride=2, padding=1, bias=False),
                nn.ReLU(),
        )
        
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                
    def forward(self, x):
        B, L, C = x.size()
        pos_num = L % 1024
        x_pos = x[:, -pos_num:, :]  # [B, pos_num, C]
        feat_num = L - pos_num
        N = feat_num // 1024
        # Reshape to (B*N, C, 32, 32) for spatial processing
        x_main = x[:, :feat_num, :].view(B, N, 1024, C)
        x_main = x_main.reshape(B * N, 32, 32, C).permute(0, 3, 1, 2)  # [B*N, C, 32, 32]

        # Apply channel attention
        x_channel = self.channel(self.avg_pool(x_main).reshape(B * N, C)).reshape(B * N, C, 1, 1) * x_main
        x_spatial = self.spatial(x_channel)
        
        if self.skip_connect:
            x = x_main + x_spatial
        else:
            x = x_spatial
        #（B, C, H, W） -> (B, H, W, C)
        x = x.permute(0,2,3,1)
        x = self.norm(x)
        x = x.view(B, -1, C)  # [B, L, C] back to original shape
        if pos_num > 0:
            x = torch.cat([x, x_pos], dim=1)
        # [B, L, C] back to original shape  
        return x

class CBAMChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, L, C] → [B, C, L]
        x_perm = x.transpose(1, 2)

        avg_out = self.shared_mlp(self.avg_pool(x_perm).squeeze(-1))
        max_out = self.shared_mlp(self.max_pool(x_perm).squeeze(-1))

        out = avg_out + max_out
        scale = self.sigmoid(out).unsqueeze(1)  # [B, 1, C]
        return x * scale.expand_as(x)

class ChannelAttention_maxavg(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, L, C] → transpose to [B, C, L] for pooling
        x_perm = x.transpose(1, 2)  # [B, C, L]

        avg_out = self.fc(self.avg_pool(x_perm).squeeze(-1))
        max_out = self.fc(self.max_pool(x_perm).squeeze(-1))

        out = avg_out + max_out
        scale = self.sigmoid(out).unsqueeze(1)  # [B, 1, C]

        return x * scale.expand_as(x)


class ECAAttention(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, L, C] → [B, C, L]
        x_perm = x.transpose(1, 2)
        y = self.avg_pool(x_perm)  # [B, C, 1]
        y = self.conv(y.transpose(1, 2)).transpose(1, 2)  # [B, C, 1] → [B, C, 1]
        y = self.sigmoid(y)
        return x * y.transpose(1, 2)  # back to [B, L, C]
    
def add_mem_ca_module(model, mem_dim=64,  flag_type="train", apt_flag="mlp",):
    ## notice: I renamed v3_4 as dd_adapter
    if flag_type == "train":
        blocks = model.model.memory_attention.layers
    else:
        blocks = model.memory_attention.layers

    if apt_flag == "ori":
        return model

    for i in range(0, 1):
        block = blocks[i]
        # dim = block.mlp.layers[0].in_features  # Get dimension from MLP input
        dim = mem_dim

        # Create adapter and insert it into block
        if apt_flag == "mem_ca":
            mem_ca = ChannelAttention(channels=dim, reduction=16)                
        elif apt_flag == "mem_cbam":
            mem_ca = CBAMChannelAttention(channels=dim, reduction=16)
        elif apt_flag == "mem_ca_maxavg":
            mem_ca = ChannelAttention_maxavg(channels=dim, reduction=16)
        elif apt_flag == "mem_eca":
            mem_ca = ECAAttention(channels=dim, k_size=3)
        elif apt_flag == "adp_ca":
            mem_ca = Adapter_CA(embed_dim=dim, mlp_ratio=0.25, norm_layer=nn.LayerNorm, skip_connect=True)
        elif apt_flag == "mem_ls":
            mem_ca = MemoryCompressor(embed_dim=64, fusion_method="attn",
                                   short_term_len=2, spatial_len=1024)

        else:
            return model
            # pass  # No adapter


        block.mem_ca = mem_ca  # Add as submodule

        # Save original forward
        block._original_forward = block.forward


        if apt_flag == "mem_ls":
            # Define new forward method
            def new_forward(self, tgt, # self-attention inputs
                    memory,  # cross-attention inputs
                    pos,
                    query_pos,
                    **kwds,):
                # print("*"*500)
                _, L, _ = memory.shape
                T_frames = L / 1024
                if T_frames >= 5:
                    # Split last 4 features (assumed padding or special tokens)
                    tail_idx = L % 1024
                    memory_time = memory[:, :-tail_idx, :]     # [B, T*L, C]
                    pos_time = pos[:, :-tail_idx, :]           # [B, T*L, C]
                    memory_tail = memory[:, -tail_idx:, :]     # [B, 4, C]
                    pos_tail = pos[:, -tail_idx:, :]           # [B, 4, C]

                    # Apply memory compressor
                    memory_fused, pos_fused = self.mem_ca(memory_time, pos_time)  # [B, L', C]
                    # Recover final memory with the 4 tail tokens
                    memory = torch.cat([memory_fused, memory_tail], dim=1)  # [B, L'+4, C]
                    pos = torch.cat([pos_fused, pos_tail], dim=1)           # [B, L'+4, C]

                x = self._original_forward(tgt, # self-attention inputs
                    memory,  # cross-attention inputs
                    pos,
                    query_pos,
                    **kwds,)
                return x 
        else:
            # Define new forward method
            def new_forward(self, tgt, # self-attention inputs
                    memory,  # cross-attention inputs
                    pos,
                    query_pos,
                    **kwds,):
                # print("*"*500)
                B = len(memory)
                # x = self.mem_ca(memory) # wrong, you should pass x to next block
                memory = self.mem_ca(memory)
                x = self._original_forward(tgt, # self-attention inputs
                    memory,  # cross-attention inputs
                    pos,
                    query_pos,
                    **kwds,)
                return x 

        # Replace forward method
        block.forward = MethodType(new_forward, block)


