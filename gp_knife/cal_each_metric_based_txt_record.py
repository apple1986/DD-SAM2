import numpy as np
import re
import os

# Replace 'metrics.txt' with your actual filename
filename = '/home/gxu/proj1/lesionSeg/utswtumor_track/Medical_SAM2/checkpoints/medicalsam2/ori_model_echo/test_each_dice_hd95_nsd_asd_fix0_bk.txt'
save_path = "/home/gxu/proj1/lesionSeg/utswtumor_track/Medical_SAM2/checkpoints/medicalsam2/ori_model_echo"
# Lists to store values
dice_vals = []
hd95_vals = []
nsd_vals = []
asd_vals = []

# Regex pattern to extract the values
pattern = r"dice:\s*([\d.]+)\s+hd95:\s*([\d.]+)\s+nsd:\s*([\d.]+)\s+asd:\s*([\d.]+)"

# Read file and extract values
with open(filename, 'r') as f:
    for line in f:
        match = re.search(pattern, line)
        if match:
            dice_vals.append(float(match.group(1)))
            hd95_vals.append(float(match.group(2)))
            nsd_vals.append(float(match.group(3)))
            asd_vals.append(float(match.group(4)))

# Convert lists to numpy arrays
dice_vals = np.array(dice_vals)
hd95_vals = np.array(hd95_vals)
nsd_vals = np.array(nsd_vals)
asd_vals = np.array(asd_vals)

# Calculate mean and std
print(f"Dice: {dice_vals.mean():.4f} ± {dice_vals.std():.4f}")
print(f"HD95: {hd95_vals.mean():.4f} ± {hd95_vals.std():.4f}")
print(f"NSD: {nsd_vals.mean():.4f} ± {nsd_vals.std():.4f}")
print(f"ASD: {asd_vals.mean():.4f} ± {asd_vals.std():.4f}")

## save format results
with open(os.path.join(save_path, "format_test_mean_dice_hd_nsd_asd0.txt"), "a") as f:
    write_content = f"{'#'*60}\nmean dice\tmean nsd\tmean hd95\tmean asd\t\n"
    write_content = write_content + f"{np.array(dice_vals).mean():.4f}±{np.array(dice_vals).std():.4f}\t" \
                                    f"{np.array(nsd_vals).mean():.4f}±{np.array(nsd_vals).std():.4f}\t" \
                                    f"{np.array(hd95_vals).mean():.4f}±{np.array(hd95_vals).std():.4f}\t" \
                                    f"{np.array(asd_vals).mean():.4f}±{np.array(asd_vals).std():.4f}\t\n"
    write_content = write_content + f"{np.array(dice_vals).mean():.2f}±{np.array(dice_vals).std():.2f}\t" \
                                    f"{np.array(nsd_vals).mean():.2f}±{np.array(nsd_vals).std():.2f}\t" \
                                    f"{np.array(hd95_vals).mean():.2f}±{np.array(hd95_vals).std():.2f}\t" \
                                    f"{np.array(asd_vals).mean():.2f}±{np.array(asd_vals).std():.2f}\t\n"
    write_content = write_content + f"{np.array(dice_vals).mean()*100:.2f}\t" \
                                    f"{np.array(nsd_vals).mean():.2f}\t" \
                                    f"{np.array(hd95_vals).mean()*100:.2f}\t" \
                                    f"{np.array(asd_vals).mean():.2f}\n"
    f.write(write_content)