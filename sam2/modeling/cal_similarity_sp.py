import torch
import torch.nn.functional as F
# from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import SpatialCorrelationCoefficient


ssim_fun = StructuralSimilarityIndexMeasure(data_range=1.0, reduction='none')
scc_fun = SpatialCorrelationCoefficient()

# x: [N, 1024, B, D], y: [N, 1024, B, D], N is num_frames, B is number of batchsize, D is feature dimension
def spatial_similarity_sp(x, y):
    N, L, B, C = x.shape
    x = torch.permute(x, (0, 1, 2, 3)).view(N*B, C, 32, 32)  # [N*B, C, 32, 32]
    y = torch.permute(y, (0, 1, 2, 3)).view(N*B, C, 32, 32)  # [N*B, C, 32, 32]
    coeff = scc_fun(x, y)    # [N, D, B]
    return coeff  


## calculate the similarity with spatial dimension
# Euclidean Distance-Based Similarity
# x: [N, 1024, B, D], y: [N, 1024, B, D], N is num_frames, B is number of batchsize, D is feature dimension
# ouput: [num_frames, 64, bs]
def euclidean_similarity_sp(x, y):
    x = torch.permute(x, (0, 3, 2, 1))  # [N, D, B, 1024]
    y = torch.permute(y, (0, 3, 2, 1))  # [N, D, B, 1024]
    dist = torch.norm(x - y, dim=-1)    # [N, D, B]
    return -dist   

# Cosine Similarity
def cosine_similarity_sp(x, y):
    x = torch.permute(x, (0, 3, 2, 1))  # [N, D, B, 1024]
    y = torch.permute(y, (0, 3, 2, 1))  # [N, D, B, 1024]
    x_norm = torch.norm(x, dim=-1, keepdim=False)  # [N, 1024, B]
    y_norm = torch.norm(y, dim=-1, keepdim=False)  # [N, 1024, B]
    cos_sim = torch.sum(x * y, dim=-1) / (x_norm * y_norm + 1e-8)  # [N, D, B]
    return cos_sim

# Dot Product Similarity
def dot_product_similarity_sp(x, y):
    x = torch.permute(x, (0, 3, 2, 1))  # [N, D, B, 1024]
    y = torch.permute(y, (0, 3, 2, 1))  # [N, D, B, 1024]
    return torch.sum(x * y, dim=-1)  # [N, D, B]

# Pearson Correlation Coefficient
def pearson_correlation_sp(x, y, eps=1e-8):
    x = torch.permute(x, (0, 3, 2, 1))  # [N, D, B, 1024]
    y = torch.permute(y, (0, 3, 2, 1))  # [N, D, B, 1024]
    x_mean = x.mean(dim=-1, keepdim=True)
    y_mean = y.mean(dim=-1, keepdim=True)
    x_centered = x - x_mean
    y_centered = y - y_mean

    numerator = torch.sum(x_centered * y_centered, dim=-1)
    denominator = torch.sqrt(torch.sum(x_centered**2, dim=-1) * torch.sum(y_centered**2, dim=-1)) + eps
    return numerator / denominator  # [N, D, B]

# Manhattan (L1) Distance-Based Similarity
def manhattan_similarity_sp(x, y):
    x = torch.permute(x, (0, 3, 2, 1))  # [N, D, B, 1024]
    y = torch.permute(y, (0, 3, 2, 1))  # [N, D, B, 1024]
    dist = torch.sum(torch.abs(x - y), dim=-1)
    return -dist  # Lower L1 = more similar

# Jaccard Similarity
def jaccard_similarity_sp(x, y):
    x = torch.permute(x, (0, 3, 2, 1))  # [N, D, B, 1024]
    y = torch.permute(y, (0, 3, 2, 1))  # [N, D, B, 1024]
    intersection = torch.sum(torch.min(x, y), dim=-1)
    union = torch.sum(torch.max(x, y), dim=-1)
    return intersection / (union + 1e-8)  # [N, D, B]

# Hamming Distance-Based Similarity
def hamming_similarity_sp(x, y):
    x = torch.permute(x, (0, 3, 2, 1))  # [N, D, B, 1024]
    y = torch.permute(y, (0, 3, 2, 1))  # [N, D, B, 1024]
    # Assuming x and y are binary vectors
    dist = torch.sum(x != y, dim=-1).float()  # Count mismatches
    return -dist  # Lower Hamming distance = more similar




# structural_similarity_index (SSIM) is not directly applicable to 1D sequences
def ssim_channelwise_sp(x, y):
    """
    x, y: Tensors of shape [N, 1024, B, 64]
    Returns: SSIM scores of shape [N, B, 64]
    """
    N, S, B, C = x.shape
    assert S == 1024 and C == 64, "Expected input shape [N, 1024, B, 64]"

    x = (x-x.min())/(x.max()-x.min()+1e-8)  # Normalize to [0, 1]
    y = (y-y.min())/(y.max()-y.min()+1e-8)  # Normalize to [0, 1] 

    # Reshape 1024 to 32x32 spatial map
    x = x.view(N, 32, 32, B, C)       # [N, 32, 32, B, 64]
    y = y.view(N, 32, 32, B, C)

    # Rearrange to [N, B, C, 1, 32, 32] for per-channel SSIM computation
    x = x.permute(0, 3, 4, 1, 2).unsqueeze(3)  # [N, B, 64, 1, 32, 32]
    y = y.permute(0, 3, 4, 1, 2).unsqueeze(3)

    # Flatten to [N*B*64, 1, 32, 32]
    x_flat = x.reshape(-1, 1, 32, 32)
    y_flat = y.reshape(-1, 1, 32, 32)

    # Compute SSIM per-channel per-frame
    # ssim_vals = ssim(x_flat, y_flat, data_range=1.0, reduction='none')
    ssim_vals = ssim_fun(x_flat, y_flat)

    # Reshape back to [N,  64, B,]
    ssim_vals = ssim_vals.view(N,  C,  B)

    return ssim_vals


def structural_similarity_sp_index(x, y, C=1e-8): 

    x = torch.permute(x, (0, 3, 2, 1))  # [N, D, B, 1024]
    y = torch.permute(y, (0, 3, 2, 1))  # [N, D, B, 1024]
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
def spearman_rank_correlation_sp(x, y):
    def rank(tensor):
        return torch.argsort(torch.argsort(tensor, dim=-1), dim=-1)

    x_rank = rank(x)
    y_rank = rank(y)

    return pearson_correlation_sp(x_rank.float(), y_rank.float())  # [N, 1024, B]


if __name__ == "__main__":
    N, L, B, D = 2, 1024, 4, 64
    x = torch.randn(N, L, B, D)
    y = torch.randn(N, L, B, D)
    # print("Euclidean Similarity:\n", euclidean_similarity_sp(x, y).shape)
    # print("Cosine Similarity:\n", cosine_similarity_sp(x, y).shape)
    # print("Dot Product Similarity:\n", dot_product_similarity_sp(x, y).shape)
    # print("Pearson Correlation:\n", pearson_correlation_sp(x, y).shape)
    # print("Manhattan Similarity:\n", manhattan_similarity_sp(x, y).shape)
    # print("Jaccard Similarity:\n", jaccard_similarity_sp(x, y).shape)
    # print("Hamming Similarity:\n", hamming_similarity_sp(x, y).shape)
    # print("Structural Similarity Index:\n", structural_similarity_sp_index(x, y).shape)
    # print("Structural Similarity Index spatial dimension:\n", ssim_channelwise_sp(x, y).shape) 
    # print("Spearman Rank Correlation:\n", spearman_rank_correlation_sp(x, y).shape)
    print("Spatial Correlation Coefficient:\n", spatial_similarity_sp(x, y).shape)
    # print("Kendall Tau Correlation:\n", kendall_tau_correlation(x, y).shape)
    # print("Kendall Tau Distance:\n", kendall_tau_distance(x, y).shape)
