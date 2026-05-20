"""NearestDecoder — reconstructs sharp-edged spatial data from a latent vector.

Designed for discrete, non-smooth spatial data (occupancy grids, binary
presence masks, HP heatmaps, etc.) as used in MAgent environments.  The
decoder uses nearest-neighbour upsampling (no interpolation blur) followed
by Conv2d refinement, preserving the crisp boundaries of discrete features
throughout the pipeline.

Data-flow summary (``n_upsample = 2``, ``window_size = 9``)::

    z (B, 64)
      │  Linear(64→256) + GELU
      │  Linear(256→32·2·2) + GELU
      ▼
    (B, 128)  ── view ──►  (B, 32,  2,  2)    ← initial coarse feature map
      │
      │  Upsample(×2, nearest)                 ← pixel replication, no blur
      │  Conv2d(32→32, 3×3, p=1) + BN + GELU   ← spatial refinement
      ▼
    (B, 32,  4,  4)
      │
      │  Upsample(×2, nearest)
      │  Conv2d(32→32, 3×3, p=1) + BN + GELU
      ▼
    (B, 32,  8,  8)
      │
      │  Upsample(size=(9,9), nearest)          ← resize to exact target
      │  Conv2d(32→out_ch, kernel=1)            ← channel projection
      ▼
    (B, out_ch,  9,  9)  ── flatten ──►  (B, out_ch·9·9)
"""

import math
import torch.nn as nn


class NearestDecoder(nn.Module):
    """Upsample-and-conv decoder for sharp-edged spatial reconstruction.

    Suitable when the reconstruction target is discrete (e.g. occupancy,
    binary masks) rather than a continuous image.  Nearest-neighbour
    upsampling avoids bilinear/triangulation smoothing that would blur
    sharp boundaries.

    The number of upsampling layers is chosen automatically from the
    target ``window_size`` so that the final feature map is roughly
    one upsampling step below the target (keeps the final ``nearest``
    adjustment cheap).

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the input latent vector *z*.
    out_channels : int
        Number of output channels.  Should match
        ``len(selected_channels)`` used by the data constructor.
    window_size : int
        Edge length of the square reconstruction target (default 9).
    hid_channels : int
        Number of channels inside the internal feature maps.  Shared
        across all resolutions (keeps the parameter count low).
    n_upsample : int or None
        Number of Upsample(×2)+Conv blocks.  When ``None`` (default) the
        value is auto-computed as ``max(1, floor(log₂(window_size/2)))``
        so that the final nearest resize moves at most a few pixels.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        out_channels: int = 9,
        window_size: int = 9,
        hid_channels: int = 32,
        n_upsample: int | None = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.window_size = window_size
        self.hid_channels = hid_channels

        # ------------------------------------------------------------------
        # Determine number of upsampling stages.
        # Target: stop when 2·2^{n_upsample} ≈ window_size, so the final
        # nearest-neighbour resize is only a minor adjustment.
        # ------------------------------------------------------------------
        if n_upsample is None:
            init_spatial = 2
            # log₂(window / init) → reasonable number of doubling steps
            log_val = math.log2(window_size / init_spatial)
            n_upsample = max(1, int(math.floor(log_val)))
        self.n_upsample = n_upsample

        # ------------------------------------------------------------------
        # 1.  Fully-connected projector:  z → initial 2×2 feature map.
        #     Shape:  (B, latent_dim)  →  (B, hid_channels·2·2)
        # ------------------------------------------------------------------
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, hid_channels * 4),  # 4 = 2×2
            nn.GELU(),
        )

        # ------------------------------------------------------------------
        # 2.  Stacked Upsample(×2, nearest) + Conv2d(3×3, pad=1) blocks.
        #     Each block doubles the spatial resolution.  Nearest-neighbour
        #     upsampling replicates every pixel value into a 2×2 block
        #     (zero smoothing), while Conv2d learns to fill in fine detail.
        #     Shapes for block i:
        #       before:  (B, C, H_i, W_i)
        #       after:   (B, C, 2·H_i, 2·W_i)
        # ------------------------------------------------------------------
        up_blocks: list[nn.Module] = []
        for _ in range(n_upsample):
            up_blocks.append(nn.Upsample(scale_factor=2, mode="nearest"))
            up_blocks.append(
                nn.Conv2d(hid_channels, hid_channels, kernel_size=3, padding=1)
            )
            up_blocks.append(nn.BatchNorm2d(hid_channels))
            up_blocks.append(nn.GELU())
        self.upsampler = nn.Sequential(*up_blocks)

        # ------------------------------------------------------------------
        # 3.  Nearest-neighbour resize to the exact target spatial size.
        #     If window_size == 2·2^{n_upsample} this is a no-op; otherwise
        #     a minor stretch/shrink.  ``mode='nearest'`` guarantees no
        #     interpolation smoothing — the values in the output are exact
        #     copies of input pixels.
        #     Shape:  (B, C, H, W)  →  (B, C, window_size, window_size)
        # ------------------------------------------------------------------
        self.to_target = nn.Upsample(
            size=(window_size, window_size), mode="nearest"
        )

        # ------------------------------------------------------------------
        # 4.  Channel projection:  hid_channels  →  out_channels  via 1×1
        #     convolution.  The 1×1 kernel mixes channel information
        #     independently at every spatial position without altering the
        #     sharp spatial layout produced by nearest-neighbour upsampling.
        #     Shape:  (B, C, T, T)  →  (B, out_ch, T, T)
        # ------------------------------------------------------------------
        self.to_out = nn.Conv2d(hid_channels, out_channels, kernel_size=1)

        # ------------------------------------------------------------------
        # 5.  Flatten for compatibility with the VAE worker, which expects
        #     a flat output and reshapes it externally to match the target.
        #     Shape:  (B, out_ch, T, T)  →  (B, out_ch·T·T)
        # ------------------------------------------------------------------
        self.flatten = nn.Flatten()

    def forward(self, z):
        """Decode a batch of latent vectors into flattened reconstructions.

        Parameters
        ----------
        z : torch.Tensor
            Latent vectors, shape ``(B, latent_dim)``.

        Returns
        -------
        torch.Tensor
            Flat reconstruction, shape ``(B, out_channels·window_size²)``.
            For the default parameters this is ``(B, 729)``.
        """
        # ------------------------------------------------------------------
        # (B, latent_dim)  →  (B, hid_channels·2²)
        # ------------------------------------------------------------------
        h = self.fc(z)

        # ------------------------------------------------------------------
        # (B, hid_channels·4)  →  (B, hid_channels, 2, 2)
        # ------------------------------------------------------------------
        h = h.view(h.size(0), self.hid_channels, 2, 2)

        # ------------------------------------------------------------------
        # (B, C, 2, 2)  →  …  →  (B, C, H, W)
        #   where  (H, W) = (2^{n_upsample+1}, 2^{n_upsample+1})
        # ------------------------------------------------------------------
        h = self.upsampler(h)

        # ------------------------------------------------------------------
        # (B, C, H, W)  →  (B, C, window_size, window_size)
        # ------------------------------------------------------------------
        h = self.to_target(h)

        # ------------------------------------------------------------------
        # (B, C, T, T)  →  (B, out_channels, T, T)
        # ------------------------------------------------------------------
        h = self.to_out(h)

        # ------------------------------------------------------------------
        # (B, out_ch, T, T)  →  (B, out_ch·T·T)
        # ------------------------------------------------------------------
        return self.flatten(h)