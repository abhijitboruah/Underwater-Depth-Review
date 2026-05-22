"""
Peng-Cosman -- Generalised dark channel using blurriness and absorption cues.

Two cues are computed:
  (1) Image blurriness: multi-scale magnitude of (blurred - original), which
      grows with imaging distance (far objects look hazier).
  (2) Per-channel light absorption: the maximal intensity within each colour
      band, which decreases with imaging distance.

These are fused into a composite transmission estimate by an adaptive mixing
weight derived from local image contrast. Then transmission is inverted to
depth via -log(t)/beta_eff (Beer-Lambert).

Reference:
    Y.-T. Peng and P. Cosman, "Underwater Image Restoration Based on Image
    Blurriness and Light Absorption", IEEE TIP 26(4), 2017.
"""

import numpy as np
import cv2


class PengCosman:
    """Args:
        sigmas:   Gaussian-blur scales for the blurriness cue (paper uses
                  {1, 2, 4}; the multi-scale average is robust to noise)
        contrast_window: window for the adaptive mixing weight (15x15 here)
        beta_eff: effective attenuation, see UDCP for the same convention.
    """

    def __init__(self, sigmas=(1, 2, 4), contrast_window=15, beta_eff=1.0,
                 **_unused):
        self.sigmas = sigmas
        self.contrast_window = contrast_window
        self.beta_eff = beta_eff

    # -- cues ----------------------------------------------------------

    def _blurriness_cue(self, gray):
        """Multi-scale blurriness map.

        For each sigma, blur the grayscale image with a Gaussian and take
        the absolute difference. Average across scales. The intuition is
        that far pixels are inherently blurrier (due to scattering and
        defocus), so the original-vs-blurred difference is *smaller* for
        them -- but we want a "larger = further" signal, so we invert
        downstream.
        """
        outs = []
        for s in self.sigmas:
            ksize = 2 * int(3 * s) + 1   # cover ~3 sigma
            blurred = cv2.GaussianBlur(gray, (ksize, ksize), s)
            outs.append(np.abs(blurred - gray))
        m = np.mean(outs, axis=0)
        # normalise to [0, 1]
        m = m / (m.max() + 1e-8)
        return m

    def _absorption_cue(self, rgb):
        """Maximum intensity per pixel across colour channels.

        Pixels close to the camera retain more of the original radiance
        in some channel; pixels far from the camera have all channels
        attenuated. So max(R, G, B) is a (rough) transmission proxy.
        """
        return np.max(rgb, axis=2)

    def _adaptive_weight(self, gray):
        """Mixing weight from local contrast.

        Where local contrast is high (textured regions), the absorption
        cue is more reliable. Where contrast is low (smooth water column,
        sand), the blurriness cue is more informative.
        """
        ksize = (self.contrast_window, self.contrast_window)
        mu = cv2.boxFilter(gray, -1, ksize)
        mu2 = cv2.boxFilter(gray * gray, -1, ksize)
        var = np.clip(mu2 - mu * mu, 0.0, None)
        contrast = np.sqrt(var)
        return contrast / (contrast.max() + 1e-8)  # in [0, 1]

    # -- public API ----------------------------------------------------

    def predict(self, rgb):
        rgb = np.asarray(rgb, dtype=np.float32)
        gray_u8 = cv2.cvtColor((rgb * 255).astype(np.uint8),
                               cv2.COLOR_RGB2GRAY)
        gray = gray_u8.astype(np.float32) / 255.0

        blur = self._blurriness_cue(gray)        # in [0, 1], higher near
        absorp = self._absorption_cue(rgb)       # in [0, 1], higher near
        w = self._adaptive_weight(gray)          # in [0, 1]

        # Both cues are already "higher = near", so they're transmissions.
        # The blurriness cue is small for far/blurry pixels; we use (1 - blur)
        # to flip its sense if needed... actually, our blur is |Gaussian - I|
        # which is *low* in blurry regions (the smoothing changes little).
        # So `blur` itself is already "low = far". Wait -- let me re-check.
        # If I is sharp, Gaussian-blur(I) differs a lot from I -> |.| large
        # If I is already blurry, Gaussian-blur(I) ~ I -> |.| small
        # Sharp regions are near; so blur is "high = near". Good.

        t = w * absorp + (1.0 - w) * blur
        t = np.clip(t, 1e-3, 1.0)

        depth = -np.log(t) / self.beta_eff
        return depth.astype(np.float32)
