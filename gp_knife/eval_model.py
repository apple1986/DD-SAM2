import monai
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDiceMetric, SurfaceDistanceMetric
## define evaluation functions
def get_dice_hd_nsd_asd():
    dice_metric_batch = DiceMetric(include_background=False, reduction="mean_batch") # each segmented object
    hd95_metric_batch = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean_batch", get_not_nans=False)
    nsd_metric_batch = SurfaceDiceMetric(include_background=False, class_thresholds=[0.5,], reduction="mean_batch", get_not_nans=False) # ach segmented object
    asd_metric_batch  = SurfaceDistanceMetric(include_background=False, reduction="mean_batch",get_not_nans=False) # ach segmented object
    return dice_metric_batch, hd95_metric_batch, nsd_metric_batch, asd_metric_batch