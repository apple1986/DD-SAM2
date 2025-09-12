import torch
import torch.nn as nn

    
class GRUMemoryFusion(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.gru = nn.GRU(embed_dim, embed_dim, batch_first=False)

    def forward(self, memory):  # [L, T, B, C]
        L, T, B, C = memory.shape
        # Merge batch and spatial: [T, B*L, C]
        memory = memory.permute(1, 2, 0, 3).reshape(T, B * L, C)
        _, h_n = self.gru(memory)  # [1, B*L, C]
        h_n = h_n[0]  # [B*L, C]
        h_n = h_n.reshape(B, L, C).permute(1, 0, 2)  # [L, B, C]
        return h_n

class AttentionMemoryFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)

    def forward(self, memory):  # memory: [L, T, B, C]
        L, T, B, C = memory.shape
        memory = memory.permute(2, 1, 0, 3)  # [B, T, L, C]
        fused = []
        for b in range(B):
            mem = memory[b]  # [T, L, C]
            # mem_seq = mem.permute(1, 0, 2)  # [L, T, C]
            query = mem.mean(dim=0, keepdim=True)  # [1, L, C]
            # Apply attention
            out, _ = self.attn(query, mem, mem)  # [L, 1, C]
            fused.append(out.squeeze(0))  # [L, C]
        fused = torch.stack(fused, dim=1)  # [L, B, C]
        return fused

class MemoryCompressor(nn.Module):
    def __init__(self, embed_dim, fusion_method='gru', short_term_len=2):
        super().__init__()
        self.short_term_len = short_term_len
        if fusion_method == 'gru':
            self.fuser = GRUMemoryFusion(embed_dim)
        elif fusion_method == 'attn':
            self.fuser = AttentionMemoryFusion(embed_dim)
        else:
            raise ValueError(f"Unsupported fusion method: {fusion_method}")

    def forward(self, feature_list, pos_enc_list):
        """
        Args:
            feature_list: List of [L, B, C] tensors (length T)
            pos_enc_list: List of [L, B, C] tensors (same length T)

        Returns:
            list_features: List of two tensors, each [L, B, C]
            list_pos_encs: List of two tensors, each [L, B, C]
        """
        T = len(feature_list)
        assert T > self.short_term_len
        assert T == len(pos_enc_list)

        # [L, T1, B, C] and [L, T2, B, C]
        ltm_feats = torch.stack(feature_list[:-self.short_term_len], dim=1)
        stm_feats = torch.stack(feature_list[-self.short_term_len:], dim=1)
        ltm_pos = torch.stack(pos_enc_list[:-self.short_term_len], dim=1)
        stm_pos = torch.stack(pos_enc_list[-self.short_term_len:], dim=1)

        # Fuse
        fused_ltm_feat = self.fuser(ltm_feats)  # [L, T, B, C]
        fused_stm_feat = self.fuser(stm_feats)
        fused_ltm_pos = self.fuser(ltm_pos)
        fused_stm_pos = self.fuser(stm_pos)

        # # Stack into single tensor
        # fused_features = torch.stack([fused_ltm_feat, fused_stm_feat], dim=1)  # [L, 2, B, C]
        # fused_pos_encs = torch.stack([fused_ltm_pos, fused_stm_pos], dim=1)

        # Also return as lists of [L, B, C]
        list_features = [fused_ltm_feat, fused_stm_feat]
        list_pos_encs = [fused_ltm_pos, fused_stm_pos]

        return  list_features, list_pos_encs # fused_features, fused_pos_encs,
    

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
    L = 1024
    B = 2
    C = 64
    T = 7
    features = [torch.randn(L, B, C) for _ in range(T)]
    pos_encs = [torch.randn(L, B, C) for _ in range(T)]

    compressor = MemoryCompressor(embed_dim=C, fusion_method='gru', short_term_len=2)
    list_feat, list_pos = compressor(features, pos_encs)
    print(list_feat[0].shape, list_feat[1].shape)  # [L, B, C], [L, B, C]

    list_feat, list_pos = memory_compressor(features, pos_encs, embed_dim=64, fusion_method='gru', short_term_len=3)
    print(list_feat[0].shape, list_feat[1].shape)  # [L, B, C], [L, B, C]



