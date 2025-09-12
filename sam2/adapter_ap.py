import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.modeling.sam.transformer import TwoWayAttentionBlock
from types import MethodType
import math

def infer_hw(num_tokens):
    h = int(math.sqrt(num_tokens))
    w = num_tokens // h
    return h, w

## build adapter
class Adapter(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.ReLU(),
            nn.Linear(dim // reduction, dim),
        )

    def forward(self, x):
        return self.adapter(x)
    
class AdapterCNN(nn.Module):
    def __init__(self, embed_dim, reduction=16):
        super().__init__()
        hidden_dim = embed_dim // reduction
        self.adapter = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, embed_dim, kernel_size=3, padding=1),
        )

    def forward(self, x):
        B, N, D = x.shape  # [B, Num_patches, Embedding_dim]
        H, W = infer_hw(N)
        x = x.transpose(1, 2)          # [B, D, N]
        x = x.view(B, D, H, W)         # [B, D, H, W]
        x = self.adapter(x)            # CNN adapter
        x = x.view(B, D, N).transpose(1, 2)  # Back to [B, N, D]
        return x

## Patch Function: Inject Adapter into Transformer-like Blocks
def add_adapters_to_model(model, target_class_names=["MultiScaleBlock", "Block"], reduction=16):
    for name, module in model.named_modules():
        if module.__class__.__name__ in target_class_names:
            # Find embed_dim dynamically
            embed_dim = None
            for sub in module.modules():
                if isinstance(sub, nn.Linear):
                    embed_dim = sub.in_features
                    break
            if embed_dim is None:
                print(f"⚠️ Skipping {name} – couldn't infer embedding dimension.")
                continue

            # Attach the adapter
            if not hasattr(module, 'adapter'):
                module.adapter = Adapter(embed_dim, reduction)

                # Save original forward
                orig_forward = module.forward

                def patched_forward(self, x, *args, **kwargs):
                    x = orig_forward(x, *args, **kwargs)
                    return x + self.adapter(x)

                # Bind new forward method
                module.forward = patched_forward.__get__(module, module.__class__)
                print(f"✅ Adapter added to: {name} ({module.__class__.__name__})")


def add_adapters_to_two_way_attention_blocks(model, reduction=16, apt_flag="mlp"):
    for name, module in model.named_modules():
        if isinstance(module, TwoWayAttentionBlock):
            if not hasattr(module, 'adapter'):
                if apt_flag == "mlp":
                    module.adapter = Adapter(256, reduction)  # dim=256 from your structure
                elif apt_flag == "cnn":
                    module.adapter = AdapterCNN(256, reduction)  # dim=256 from your structure

                # Save original MLP forward
                original_forward = module.mlp.forward

                def patched_forward(x, orig_forward=original_forward, adapter=module.adapter):
                    return orig_forward(x) + adapter(x)

                # Patch the forward method of the MLP safely
                module.mlp.forward = patched_forward

                print(f"✅ Adapter added to: {name}.mlp")





## for image encoder
class Adapter_ImgEnc(nn.Module):
    def __init__(self, input_dim, adapter_dim=64):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, adapter_dim),
            nn.ReLU(),
            nn.Linear(adapter_dim, input_dim),
        )

    def forward(self, x):
        return self.adapter(x)
    

class Adapter_ImgEnc_CNN(nn.Module):
    def __init__(self, input_dim, adapter_dim=64):
        super().__init__()
        self.adapter = nn.Sequential(
            # nn.LayerNorm(input_dim),
            nn.Conv2d(input_dim, input_dim//2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(input_dim//2, input_dim, kernel_size=3, padding=1)

        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2) # X: BHWC-->BCHW
        x = self.adapter(x)
        x = x.permute(0, 2, 3, 1)
        return x
    

########################
class ASPPLikeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=d, dilation=d, groups=in_channels)
            for d in [1, 2, 4, 8]
        ])

    def forward(self, x):
        return sum(conv(x) for conv in self.convs) / len(self.convs)  # average


class Adapter_sammed2d_v2(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=0.25, norm_layer = nn.LayerNorm, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        hidden_dim = int(embed_dim * mlp_ratio)
        self.norm = norm_layer(embed_dim)

        # self.spatial = Adapter(embed_dim, reduction=4)
        self.spatial = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
                nn.ReLU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=4, stride=2, padding=1, bias=False),
                nn.ReLU(),
        )

    def forward(self, x):
        #x -> （B, H, W, C）-> （B, C, H, W）
        # x_ori = x
        x = x.permute(0,3,1,2)
        B, C, _, _ = x.size()
        x_channel = (self.channel(self.avg_pool(x).view(B, C)).view(B, C, 1, 1))* x
        # x_channel = x_channel.permute(0,2,3,1)
        x_spatial = self.spatial(x_channel)
        
        # if self.skip_connect:
        #     x = x_ori + x_spatial
        # else:
        x = x_spatial
        #（B, C, H, W） -> (B, H, W, C)
        x = x.permute(0,2,3,1)
        return x #self.norm(x)

# ap: used DD-Adapter
class DD_Adapter(nn.Module):
    def __init__(self, input_dim, adapter_dim=64):
        super().__init__()
               
        self.pw1 = nn.Sequential(
            nn.Conv2d(input_dim, input_dim//2, kernel_size=1, padding=0, groups=1),
            nn.GELU(),)

        ## second
        self.dw2_1 = nn.Sequential(
            nn.Conv2d(input_dim//2, input_dim//2, kernel_size=3, padding=1, dilation=1, groups=input_dim//2),
            nn.GELU(),)
        self.dw2_2 = nn.Sequential(
            nn.Conv2d(input_dim//2, input_dim//2, kernel_size=3, padding=3, dilation=3, groups=input_dim//2),
            nn.GELU(),)        

        self.pw2 = nn.Sequential(
            nn.Conv2d(input_dim//2, input_dim, kernel_size=1, padding=0, groups=1),
            nn.GELU(),)
        
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # BHWC -> BCHW, like torch.Size([1, 96, 128, 128])

        x = self.pw1(x)
        x = self.dw2_1(x) + self.dw2_2(x)
        x = self.pw2(x)

        # x = x + residual  # Residual connection
        x = x.permute(0, 2, 3, 1)  # BCHW -> BHWC
        return x

class DD_Adapter_other_rate(nn.Module):
    def __init__(self, input_dim, dilation_rate=[1, 3]):
        super().__init__()
        
        # First pointwise conv
        self.pw1 = nn.Sequential(
            nn.Conv2d(input_dim, input_dim // 2, kernel_size=1, padding=0, groups=1),
            nn.GELU()
        )

        # Depthwise dilated convolutions
        blk_layers = []
        for rate in dilation_rate:
            blk_layers.append(
                nn.Sequential(
                    nn.Conv2d(input_dim // 2, input_dim // 2, kernel_size=3, padding=rate, dilation=rate, groups=input_dim // 2),
                    nn.GELU()
                )
            )
        self.blk = nn.ModuleList(blk_layers)

        # Second pointwise conv
        self.pw2 = nn.Sequential(
            nn.Conv2d(input_dim // 2, input_dim, kernel_size=1, padding=0, groups=1),
            nn.GELU()
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # BHWC -> BCHW

        x = self.pw1(x)
        for blk in self.blk:
            x = x + blk(x)
        x = self.pw2(x)

        x = x.permute(0, 2, 3, 1)  # BCHW -> BHWC
        return x
    
## different adapter
## build adapter: Medical sam adapter:a
class Adapter_medsamapt(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.ReLU(),
            nn.Linear(dim // reduction, dim),
        )

    def forward(self, x):
        return self.adapter(x)

class Adapter_medsamapt_v2(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.Linear(dim // reduction, dim),
            # nn.LayerNorm(dim)
        )

    def forward(self, x):
        return self.adapter(x)


# from SAM-Med2D:b
class Adapter_sammed2d(nn.Module):
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
        #x -> （B, H, W, C）-> （B, C, H, W）
        x = x.permute(0,3,1,2)
        B, C, _, _ = x.size()
        x_channel = self.channel(self.avg_pool(x).view(B, C)).view(B, C, 1, 1) * x
        x_spatial = self.spatial(x_channel)
        
        if self.skip_connect:
            x = x + x_spatial
        else:
            x = x_spatial
        #（B, C, H, W） -> (B, H, W, C)
        x = x.permute(0,2,3,1)
        return self.norm(x)


class Adapter_sammed2d_v2(nn.Module):
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
        # self.spatial = Adapter(embed_dim, reduction=4)
        self.spatial = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
                nn.ReLU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=4, stride=2, padding=1, bias=False),
                nn.ReLU(),
        )

    def forward(self, x):
        #x -> （B, H, W, C）-> （B, C, H, W）
        # x_ori = x
        x = x.permute(0,3,1,2)
        B, C, _, _ = x.size()
        x_channel = (self.channel(self.avg_pool(x).view(B, C)).view(B, C, 1, 1))* x
        # x_channel = x_channel.permute(0,2,3,1)
        x_spatial = self.spatial(x_channel)
        
        # if self.skip_connect:
        #     x = x_ori + x_spatial
        # else:
        x = x_spatial
        #（B, C, H, W） -> (B, H, W, C)
        x = x.permute(0,2,3,1)
        return x #self.norm(x)

#################
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = kernel_size // 2

        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x: (B, C, H, W)
        avg_out = torch.mean(x, dim=1, keepdim=True)      # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)    # (B, 1, H, W)
        attn = torch.cat([avg_out, max_out], dim=1)       # (B, 2, H, W)
        attn = self.conv(attn)                            # (B, 1, H, W)
        attn = self.sigmoid(attn)                         # (B, 1, H, W)
        return x * attn                                   # Element-wise multiplication

## use lora
###lora
class LoRALinear(nn.Module):
    def __init__(self, original_linear, rank=4, alpha=1.0):
        super().__init__()
        self.original = original_linear
        self.rank = rank
        self.alpha = alpha

        # Freeze the original weights
        for param in self.original.parameters():
            param.requires_grad = False

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # LoRA adapters
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)

        # Scale
        self.scale = alpha / rank

        # Initialize LoRA layers
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        return self.original(x) + self.scale * self.lora_up(self.lora_down(x))


def get_parent_module(model, name):
    """Helper to get parent module given module name."""
    names = name.split('.')
    for n in names[:-1]:
        model = getattr(model, n)
    return model

def apply_lora_to_attention(model, rank=4, alpha=1.0):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and ('qkv' in name or 'proj' in name):
            parent = get_parent_module(model, name)
            attr_name = name.split('.')[-1]
            orig_linear = getattr(parent, attr_name)
            lora_linear = LoRALinear(orig_linear, rank=rank, alpha=alpha)
            setattr(parent, attr_name, lora_linear)
    
    return model



def add_adapters_to_imgenc(model, adapter_dim=64, num_blocks=12, flag_type="train", apt_flag="mlp", dilation_rate=[1,3]):
    ## notice: I renamed v3_4 as dd_adapter
    if flag_type == "train":
        blocks = model.model.image_encoder.trunk.blocks
        image_encoder = model.model.image_encoder.trunk
    else:
        blocks = model.image_encoder.trunk.blocks
        image_encoder = model.image_encoder.trunk

    if apt_flag == "lora":
        model = apply_lora_to_attention(image_encoder, rank=4, alpha=16)
        return model

    if apt_flag == "ori":
        return model

    for i in range(min(num_blocks, len(blocks))):
        block = blocks[i]
        dim = block.mlp.layers[0].in_features  # Get dimension from MLP input

        # Create adapter and insert it into block
        if apt_flag == "mlp":
            adapter = Adapter_ImgEnc(dim, adapter_dim)
        elif apt_flag == "cnn":
            adapter = Adapter_ImgEnc_CNN(dim, adapter_dim)
        elif apt_flag == "dd_adapter":
            adapter = DD_Adapter(dim, adapter_dim)  
        elif apt_flag == "dd_adapter_other_rate":
            adapter = DD_Adapter_other_rate(dim, dilation_rate=dilation_rate)

        ## various adapter
        elif apt_flag == "mlp_medsamapt":
            adapter = Adapter_medsamapt(dim)    
        elif apt_flag == "mlp_medsamapt_v2":
            adapter = Adapter_medsamapt_v2(dim)   
        elif apt_flag == "cnn_sammed2d":
            adapter = Adapter_sammed2d(dim)
        elif apt_flag == "cnn_sammed2d_v2":
            adapter = Adapter_sammed2d_v2(dim)
        block.adapter = adapter  # Add as submodule

        # Save original forward
        block._original_forward = block.forward

        # Define new forward method
        def new_forward(self, x):
            x = self._original_forward(x)
            return x + self.adapter(x)

        # Replace forward method
        block.forward = MethodType(new_forward, block)

    print(f"Adapters added to the first {num_blocks} blocks.")

