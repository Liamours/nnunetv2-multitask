import torch
from torch import nn

from nnunetv2.architecture.cbam import CBAM


def test_default_has_no_post_norm():
    """Regression guard: every existing CBAM config must be bit-identical to before this change."""
    torch.manual_seed(0)
    cbam = CBAM(nn.Conv2d, channels=8)
    assert isinstance(cbam.norm, nn.Identity)


def test_post_norm_applies_when_requested():
    torch.manual_seed(0)
    cbam = CBAM(nn.Conv2d, channels=8, norm_op=nn.InstanceNorm2d, norm_op_kwargs={"eps": 1e-5, "affine": True})
    assert isinstance(cbam.norm, nn.InstanceNorm2d)

    x = torch.randn(2, 8, 16, 16) * 5 + 3  # deliberately off-zero-mean, off-unit-variance
    with torch.no_grad():
        output = cbam(x)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    # InstanceNorm2d normalizes per-sample, per-channel over the spatial dims - confirms the norm
    # actually ran, not just that it's attached and inert.
    per_channel_std = output.std(dim=(2, 3))
    assert torch.allclose(per_channel_std, torch.ones_like(per_channel_std), atol=0.1)


if __name__ == "__main__":
    test_default_has_no_post_norm()
    test_post_norm_applies_when_requested()
    print("ok")
