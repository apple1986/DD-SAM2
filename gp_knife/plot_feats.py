import torch
import matplotlib.pyplot as plt

def plot_feature_maps(tensor, num_rows=4, num_cols=4, title_prefix="Feature"):
    """
    Plot feature maps from a 4D tensor of shape [B, C, H, W].

    Args:
        tensor (torch.Tensor): Feature map with shape [B, C, H, W].
        num_rows (int): Number of rows in the plot grid.
        num_cols (int): Number of columns in the plot grid.
        title_prefix (str): Prefix for subplot titles.

    Returns:
        None
    """
    assert tensor.ndim == 4, "Expected tensor of shape [B, C, H, W]"
    b, c, h, w = tensor.shape
    assert b == 1, "Only supports batch size = 1"
    
    num_plots = num_rows * num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
    axes = axes.flatten()
    
    for i in range(num_plots):
        if i < c:
            ax = axes[i]
            fmap = tensor[0, i].detach().cpu().to(torch.float32).numpy()
            ax.imshow(fmap, cmap='viridis')
            ax.axis('off')
            ax.set_title(f"{title_prefix} {i}")
        else:
            axes[i].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example input tensor of shape [1, 64, 32, 32]
    x = torch.randn(1, 64, 32, 32)

    # Plot first 16 feature maps in a 4x4 grid
    plot_feature_maps(x, num_rows=4, num_cols=4)