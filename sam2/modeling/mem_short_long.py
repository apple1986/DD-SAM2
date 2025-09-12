import torch
import torch.nn as nn


class GRUMemoryFusion(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.gru = nn.GRU(embed_dim, embed_dim, batch_first=False)

    def forward(self, memory):  # memory: [L, T, B, C]
        L, T, B, C = memory.shape
        memory = memory.permute(1, 2, 0, 3).reshape(T, B * L, C)  # [T, B*L, C]
        _, h_n = self.gru(memory)  # [1, B*L, C]
        h_n = h_n[0].reshape(B, L, C).permute(1, 0, 2)  # [L, B, C]
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
            query = mem.mean(dim=0, keepdim=True)  # [1, L, C]
            out, _ = self.attn(query, mem, mem)  # [1, L, C]
            fused.append(out.squeeze(0))  # [L, C]
        return torch.stack(fused, dim=1)  # [L, B, C]


class MemoryCompressor(nn.Module):
    def __init__(self, embed_dim, fusion_method='gru', short_term_len=2, spatial_len=1024):
        super().__init__()
        self.short_term_len = short_term_len
        self.spatial_len = spatial_len
        if fusion_method == 'gru':
            self.fuser = GRUMemoryFusion(embed_dim)
        elif fusion_method == 'attn':
            self.fuser = AttentionMemoryFusion(embed_dim)
        else:
            raise ValueError(f"Unsupported fusion method: {fusion_method}")

    def reshape_seq(self, x):  # x: [B, L, C] → [T, L', B, C]
        B, L, C = x.shape
        T = L // self.spatial_len
        assert L % self.spatial_len == 0, "L must be divisible by spatial_len"
        x = x.reshape(B, T, self.spatial_len, C).permute(1, 2, 0, 3)  # [T, L', B, C]
        return x

    def forward(self, features, pos_encs):  # both: [B, L, C]
        feats_seq = self.reshape_seq(features)  # [T, L', B, C]
        pos_seq = self.reshape_seq(pos_encs)

        T = feats_seq.shape[0]
        assert T > self.short_term_len

        ltm_feats = feats_seq[:-self.short_term_len]  # [T1, L', B, C]
        stm_feats = feats_seq[-self.short_term_len:]  # [T2, L', B, C]
        ltm_pos = pos_seq[:-self.short_term_len]
        stm_pos = pos_seq[-self.short_term_len:]

        # [L', T1, B, C] for fusion
        ltm_feats = ltm_feats.permute(1, 0, 2, 3)  # [L', T1, B, C]
        stm_feats = stm_feats.permute(1, 0, 2, 3)
        ltm_pos = ltm_pos.permute(1, 0, 2, 3)
        stm_pos = stm_pos.permute(1, 0, 2, 3)

        fused_ltm_feat = self.fuser(ltm_feats)  # [L', B, C]
        fused_stm_feat = self.fuser(stm_feats)
        fused_ltm_pos = self.fuser(ltm_pos)
        fused_stm_pos = self.fuser(stm_pos)

        # Concatenate along time axis: [2 * L', B, C]
        fused_feat = torch.cat([fused_ltm_feat, fused_stm_feat], dim=0)  # [2L', B, C]
        fused_pos = torch.cat([fused_ltm_pos, fused_stm_pos], dim=0)

        # Return to [B, L, C]
        fused_feat = fused_feat.permute(1, 0, 2)  # [B, 2L', C]
        fused_pos = fused_pos.permute(1, 0, 2)

        return fused_feat, fused_pos  # both [B, L, C]



# Utility factory
def memory_compressor(features, pos_encs, embed_dim=64, fusion_method='gru', short_term_len=3, spatial_len=1024):
    compressor = MemoryCompressor(embed_dim=embed_dim, fusion_method=fusion_method,
                                   short_term_len=short_term_len, spatial_len=spatial_len)
    return compressor(features, pos_encs)


if __name__ == "__main__":
    B, C, T, L_per_frame = 2, 64, 7, 1024
    total_L = T * L_per_frame
    features = torch.randn(B, total_L, C)
    pos_encs = torch.randn(B, total_L, C)

    fused_feat, fused_pos  = memory_compressor(features, pos_encs, embed_dim=C, fusion_method='gru',
                                            short_term_len=2, spatial_len=L_per_frame)
    print(fused_feat.shape, fused_pos.shape)  # [L', B, C], [L', B, C]
