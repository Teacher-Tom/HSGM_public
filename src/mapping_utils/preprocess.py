from typing import Optional

import numpy as np
def preprocess_depth(depth:np.ndarray,lower_bound:float=0.1,upper_bound:float=4.9):
    depth[np.where((depth<lower_bound)|(depth>upper_bound))] = 0
    return depth

rng = np.random.default_rng(111)

def preprocess_depth_noise(
    depth: np.ndarray,
    lower_bound: float = 0.1,
    upper_bound: float = 4.9,
) -> np.ndarray:
    """
    Preprocess depth and add simulated 1cm-precision noise (vectorized).

    - values outside [lower_bound, upper_bound] set to 0
    - noise only applied to depth > 1m within valid range
    - noise: sigma(mm) = 11 + 0.1% * depth (depth in m)
      equivalent: sigma_mm = 11 + depth_m
    """
    if depth.size == 0:
        return depth

    d = depth
    in_range = (d >= lower_bound) & (d <= upper_bound)
    noisy_mask = in_range & (d > 1.0)

    if not np.any(noisy_mask):
        d[~in_range] = 0
        return d

    d_sel = d[noisy_mask].astype(np.float64, copy=False)

    # quantize to 1cm (integer in cm)
    d_cm = np.rint(d_sel * 100.0).astype(np.int32, copy=False)

    # compute per-pixel sigma according to formula (unit: mm)
    # 0.1% * depth(mm) = 0.001 * (depth_m * 1000) = depth_m（mm）
    sigma_mm = 11.0 + d_sel  # mm
    sigma_cm = sigma_mm / 10.0  # cm

    # uniform noise approx: U[-sqrt(3)*sigma, +sqrt(3)*sigma] (unit: cm)
    half_range = np.sqrt(3.0) * sigma_cm
    noise_cm = rng.uniform(-half_range, half_range, size=d_cm.shape)

    d_cm_noisy = np.rint(d_cm.astype(np.float64, copy=False) + noise_cm).astype(np.int32, copy=False)
    d[noisy_mask] = d_cm_noisy.astype(d.dtype, copy=False) / 100.0

    d[~in_range] = 0
    return d


def preprocess_image(image:np.ndarray):
    return image