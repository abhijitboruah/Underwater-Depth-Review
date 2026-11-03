"""
UDCP -- Underwater Dark Channel Prior.

Drews et al. variant of He et al.'s dark-channel prior, adapted for
the underwater regime by dropping the red channel from the dark-channel
statistic (red is preferentially attenuated underwater and thus violates
the "some channel has near-zero intensity" assumption that motivates the
original DCP).

Reference:
    P. Drews et al., "Transmission Estimation in Underwater Single Images",
    ICCV Workshops 2013.
"""

import numpy as np
import cv2


class UDCP:
    """Patch-based UDCP depth estimator.

    Args:
        patch:    dark-channel local window (default 15x15 as in the paper)
        omega:    transmission slack constant (0.9 standard)
        top_frac: fraction of brightest dark-channel pixels used to estimate
                  the background light (0.1% in the original paper)
        beta_eff: effective attenuation coefficient; converts transmission
                  -log(t) into a depth in metres. Fit once per dataset on a
                  held-out calibration subset (see _classical_calibrate.py).
                  If left at 1.0, depths are in arbitrary units; the LS
                  alignment in run.py rescales anyway.
    """

    def __init__(self, patch=15, omega=0.9, top_frac=0.001, beta_eff=1.0,
                 **_unused):
        self.patch = patch
        self.omega = omega
        self.top_frac = top_frac
        self.beta_eff = beta_eff

    # -- core operations -----------------------------------------------

    def _dark_channel(self, img):
        """Underwater dark channel: min over G,B channels and a patch.

        Drews et al. propose using only the green and blue channels (drop
        red) because red is heavily attenuated and its minimum is near
        zero just due to absorption, not because of local "dark patch".
        """
        # img: H x W x 3 in [0, 1]
        gb_min = np.min(img[..., 1:], axis=2)  # min(G, B) per pixel
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (self.patch, self.patch))
        return cv2.erode(gb_min, k)

    def _estimate_B(self, img, dc):
        """Background light from the top-`top_frac` brightest DC pixels."""
        h, w = dc.shape
        n_top = max(1, int(h * w * self.top_frac))
        # flatten -> argsort -> take last n_top
        flat = dc.reshape(-1)
        idx = np.argpartition(flat, -n_top)[-n_top:]
        ys, xs = np.unravel_index(idx, (h, w))
        return img[ys, xs].mean(axis=0)  # (3,) BGR -> RGB by indexing

    def _transmission(self, img, B):
        """t(x) = 1 - omega * DC_underwater(I / B_inf)"""
        normed = img / np.maximum(B[None, None, :], 1e-6)
        dc_n = self._dark_channel(normed)
        return 1.0 - self.omega * dc_n

    # -- public API ----------------------------------------------------

    def predict(self, rgb):
        """Predict relative depth from a single image.

        Args:
            rgb: H x W x 3 float32 in [0,1] (RGB order).

        Returns:
            depth: H x W float32, "depth-like" map. Larger = further.
                   In arbitrary units when beta_eff=1; metres if calibrated.
        """
        rgb = np.asarray(rgb, dtype=np.float32)
        dc = self._dark_channel(rgb)
        B = self._estimate_B(rgb, dc)
        t = self._transmission(rgb, B)

        # Beer-Lambert inversion: t = exp(-beta * z)  =>  z = -log(t) / beta
        t = np.clip(t, 1e-3, 1.0)  # avoid log(0)
        depth = -np.log(t) / self.beta_eff
        return depth.astype(np.float32)
