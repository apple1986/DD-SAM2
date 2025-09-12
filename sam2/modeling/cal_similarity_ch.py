import torch
import torch.nn.functional as F

## calculate the similarity with channel dimension
# Euclidean Distance-Based Similarity
# x: [N, 1024, B, D], y: [N, 1024, B, D], N is num_frames, B is number of batchsize, D is feature dimension
# ouput: [num_frames, 1024, bs]
def euclidean_similarity(x, y):
    dist = torch.norm(x - y, dim=-1)         # [N, 1024, B]
    return -dist   

# Cosine Similarity
def cosine_similarity(x, y):
    x_norm = torch.norm(x, dim=-1, keepdim=False)  # [N, 1024, B]
    y_norm = torch.norm(y, dim=-1, keepdim=False)  # [N, 1024, B]
    cos_sim = torch.sum(x * y, dim=-1) / (x_norm * y_norm + 1e-8)  # [N, 1024, B]
    return cos_sim

# Dot Product Similarity
def dot_product_similarity(x, y):
    return torch.sum(x * y, dim=-1)  # [N, 1024, B]

# Pearson Correlation Coefficient
def pearson_correlation(x, y, eps=1e-8):
    x_mean = x.mean(dim=-1, keepdim=True)
    y_mean = y.mean(dim=-1, keepdim=True)
    x_centered = x - x_mean
    y_centered = y - y_mean

    numerator = torch.sum(x_centered * y_centered, dim=-1)
    denominator = torch.sqrt(torch.sum(x_centered**2, dim=-1) * torch.sum(y_centered**2, dim=-1)) + eps
    return numerator / denominator  # [N, 1024, B]

# Manhattan (L1) Distance-Based Similarity
def manhattan_similarity(x, y):
    dist = torch.sum(torch.abs(x - y), dim=-1)
    return -dist  # Lower L1 = more similar

# Jaccard Similarity
def jaccard_similarity(x, y):
    intersection = torch.sum(torch.min(x, y), dim=-1)
    union = torch.sum(torch.max(x, y), dim=-1)
    return intersection / (union + 1e-8)  # [N, 1024, B]

# Hamming Distance-Based Similarity
def hamming_similarity(x, y):
    # Assuming x and y are binary vectors
    dist = torch.sum(x != y, dim=-1).float()  # Count mismatches
    return -dist  # Lower Hamming distance = more similar

# structural_similarity_index (SSIM) is not directly applicable to 1D sequences
def structural_similarity_index(x, y, C=1e-8):
    mu_x = x.mean(dim=-1, keepdim=True)
    mu_y = y.mean(dim=-1, keepdim=True)
    sigma_x = x.var(dim=-1, keepdim=True) + C
    sigma_y = y.var(dim=-1, keepdim=True) + C
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean(dim=-1, keepdim=True) + C

    ssim_numerator = (2 * mu_x * mu_y + C) * (2 * sigma_xy + C)
    ssim_denominator = (mu_x**2 + mu_y**2 + C) * (sigma_x + sigma_y + C)
    ssim_coef = ssim_numerator / ssim_denominator 
    return  ssim_coef[...,-1]# [N, 1024, B]

#
# Spearman Rank Correlation Coefficient
def spearman_rank_correlation(x, y):
    def rank(tensor):
        return torch.argsort(torch.argsort(tensor, dim=-1), dim=-1)

    x_rank = rank(x)
    y_rank = rank(y)

    return pearson_correlation(x_rank.float(), y_rank.float())  # [N, 1024, B]
# Kendall Tau Correlation Coefficient
def kendall_tau_correlation(x, y):
    def concordant_pairs(x, y):
        return torch.sum((x[:, :, None] - x[:, None, :]) * (y[:, :, None] - y[:, None, :]) > 0, dim=(1, 2))

    def discordant_pairs(x, y):
        return torch.sum((x[:, :, None] - x[:, None, :]) * (y[:, :, None] - y[:, None, :]) < 0, dim=(1, 2))

    concordant = concordant_pairs(x, y)
    discordant = discordant_pairs(x, y)
    return (concordant - discordant) / (concordant + discordant + 1e-8)  # [N]

# Kendall Tau Distance
def kendall_tau_distance(x, y):
    return 1 - kendall_tau_correlation(x, y)  # [N, 1024, B]



if __name__ == "__main__":
    N, L, B, D = 2, 1024, 4, 64
    x = torch.randn(N, L, B, D)
    y = torch.randn(N, L, B, D)
    print("Euclidean Similarity:\n", euclidean_similarity(x, y).shape)
    print("Cosine Similarity:\n", cosine_similarity(x, y).shape)
    print("Dot Product Similarity:\n", dot_product_similarity(x, y).shape)
    print("Pearson Correlation:\n", pearson_correlation(x, y).shape)
    print("Manhattan Similarity:\n", manhattan_similarity(x, y).shape)
    print("Jaccard Similarity:\n", jaccard_similarity(x, y).shape)
    print("Hamming Similarity:\n", hamming_similarity(x, y).shape)
    print("Structural Similarity Index:\n", structural_similarity_index(x, y).shape)
    print("Spearman Rank Correlation:\n", spearman_rank_correlation(x, y).shape)
    print("Kendall Tau Correlation:\n", kendall_tau_correlation(x, y).shape)
    print("Kendall Tau Distance:\n", kendall_tau_distance(x, y).shape)
