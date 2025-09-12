import torch
import torch.nn as nn


def split_memory(features, short_term_len=2):
    """
    Split a sequence of time-ordered features into short-term and long-term memory.

    Args:
        features (Tensor): Shape [L, B, C], B is time order.
        short_term_len (int): Number of frames to use for short-term memory (from the end).
    
    Returns:
        long_term_features (Tensor): [L, B1, C]
        short_term_features (Tensor): [L, B2, C]
    """
    assert features.shape[1] > short_term_len, "Short-term length must be less than total frames"
    
    short_term_features = features[:, -short_term_len:, :]  # last few time steps
    long_term_features = features[:, :-short_term_len, :]   # the rest

    return long_term_features, short_term_features


class GRUMemoryFusion(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.gru = nn.GRU(input_size=embed_dim, hidden_size=embed_dim, batch_first=False)

    def forward(self, memory):  # memory: [L, T, C]
        L, T, C = memory.shape
        memory_reordered = memory.permute(1, 0, 2)  # → [T, L, C]
        output, h_n = self.gru(memory_reordered)    # h_n: [1, L, C]
        return h_n.squeeze(0)  # [L, C]

def fuse_memory(features, short_term_len=2, fusion_method="gru"):
    ltm_feats, stm_feats = split_memory(features, short_term_len)
    embed_dim = features.shape[-1]
    fuser = GRUMemoryFusion(embed_dim)

    ltm_fused = fuser(ltm_feats)  # [L, C]
    stm_fused = fuser(stm_feats)  # [L, C]

    fused = torch.stack([ltm_fused, stm_fused], dim=1)  # [L, 2, C]
    return fused

class AttentionMemoryFusion(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=False)

    def forward(self, memory):  # memory: [L, B, C]
        query = memory.mean(dim=1, keepdim=True)  # [L, 1, C]
        memory_reordered = memory.permute(1, 0, 2)  # [B, L, C]
        query_reordered = query.permute(1, 0, 2)    # [1, L, C]
        out, _ = self.attn(query_reordered, memory_reordered, memory_reordered)  # [1, L, C]
        return out.squeeze(0)  # [L, C]

def fuse_memory_attn(features, short_term_len=2):
    ltm_feats, stm_feats = split_memory(features, short_term_len)
    embed_dim = features.shape[-1]
    fuser = AttentionMemoryFusion(embed_dim)

    ltm_fused = fuser(ltm_feats)
    stm_fused = fuser(stm_feats)
    fused = torch.stack([ltm_fused, stm_fused], dim=1)  # [L, 2, C]
    return fused


## for the feature and feature position
class MemoryCompressor(nn.Module):
    def __init__(self, embed_dim, fusion_method='gru', short_term_len=2):
        super().__init__()
        self.short_term_len = short_term_len
        self.fusion_method = fusion_method
        if fusion_method == 'gru':
            self.fuser = GRUMemoryFusion(embed_dim)
        elif fusion_method == 'attn':
            self.fuser = AttentionMemoryFusion(embed_dim)
        else:
            raise ValueError(f"Unsupported fusion method: {fusion_method}")

    def forward(self, maskmem_features, maskmem_pos_enc):
        """
        Args:
            maskmem_features (List[Tensor]): List of [L, C] tensors (time-ordered).
            maskmem_pos_enc (List[Tensor]): List of [L, C] tensors (aligned to features).

        Returns:
            fused_features: Tensor of shape [L, 2, C]
            fused_pos_enc: Tensor of shape [L, 2, C]
        """
        assert len(maskmem_features) == len(maskmem_pos_enc), "Length mismatch"

        features = torch.stack(maskmem_features, dim=1)     # [L, T, C]
        pos_enc = torch.stack(maskmem_pos_enc, dim=1)       # [L, T, C]
        T = features.shape[1]
        assert T > self.short_term_len, "Short-term memory length too long"

        # Split memory
        ltm_features = features[:, :-self.short_term_len, :]  # [L, T1, C]
        stm_features = features[:, -self.short_term_len:, :]  # [L, T2, C]
        ltm_pos_enc = pos_enc[:, :-self.short_term_len, :]
        stm_pos_enc = pos_enc[:, -self.short_term_len:, :]

        # Fuse
        fused_ltm_feat = self.fuser(ltm_features)     # [L, C]
        fused_stm_feat = self.fuser(stm_features)
        fused_ltm_pos = self.fuser(ltm_pos_enc)
        fused_stm_pos = self.fuser(stm_pos_enc)

        # Stack into [L, 2, C]
        fused_features = torch.stack([fused_ltm_feat, fused_stm_feat], dim=1)
        fused_pos_enc = torch.stack([fused_ltm_pos, fused_stm_pos], dim=1)

        return fused_features, fused_pos_enc

def memory_compressor(features, pos_encs, embed_dim=64, fusion_method='gru', short_term_len=3):
    """
    Factory function to create a MemoryCompressor instance.
    
    Args:
        embed_dim (int): Dimension of the feature vectors.
        fusion_method (str): 'gru' or 'attn' for fusion method.
        short_term_len (int): Length of short-term memory.
    
    Returns:
        MemoryCompressor: Instance of the compressor.
    """
    compressor = MemoryCompressor(embed_dim=embed_dim, fusion_method=fusion_method, short_term_len=short_term_len)
    fused_feat, fused_pos = compressor(features, pos_encs)
        
    return fused_feat, fused_pos

if __name__ == "__main__":
    L, B, C = 1024, 7, 64
    features = torch.randn(L, B, C)

    ltm, stm = split_memory(features, short_term_len=2)

    print(f"Long-term shape: {ltm.shape}")   # torch.Size([1024, 5, 64])
    print(f"Short-term shape: {stm.shape}")  # torch.Size([1024, 2, 64])

    fused = fuse_memory(features, short_term_len=2)
    print(f"Fused memory shape: {fused.shape}")  # torch.Size([1024, 2, 64])
    fused_attn = fuse_memory_attn(features, short_term_len=2)
    print(f"Fused memory with attention shape: {fused_attn.shape}")  # torch.Size([1024, 2, 64])
# This code defines memory fusion techniques for short-term and long-term features in a sequence.
# It includes splitting the features, applying GRU and attention-based fusion methods, and testing the functionality.
# This code defines memory fusion techniques for short-term and long-term features in a sequence.
# It includes splitting the features, applying GRU and attention-based fusion methods, and testing the functionality.
# The MemoryCompressor class combines these methods for feature and position encoding compression.L, C = 1024, 64
    T = 7  # number of previous frames
    features = [torch.randn(L, C) for _ in range(T)]
    pos_encs = [torch.randn(L, C) for _ in range(T)]

    compressor = MemoryCompressor(embed_dim=C, fusion_method='attn', short_term_len=2)
    fused_feat, fused_pos = compressor(features, pos_encs)

    print(fused_feat.shape)  # [L, 2, C]
    print(fused_pos.shape)   # [L, 2, C]