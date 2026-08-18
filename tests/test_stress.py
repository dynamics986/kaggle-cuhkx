import torch

from cuhkx_har.stress import apply_stress


def test_drop_radar_masks_and_zeros_only_radar() -> None:
    batch = {
        "visual": torch.ones(2, 3, 2, 3, 4, 4),
        "skeleton": torch.ones(2, 2, 68),
        "imu": torch.ones(2, 2, 80),
        "radar": torch.ones(2, 2, 13),
        "modality_mask": torch.tensor([[True, True, True, True, True, True], [True] * 5 + [False]]),
    }
    stressed, affected = apply_stress(batch, "drop_radar")
    assert affected.tolist() == [True, False]
    assert not stressed["modality_mask"][:, 5].any()
    assert not stressed["radar"].any()
    assert batch["radar"].all()


def test_visual_first_frame_preserves_masks_and_collapses_time() -> None:
    visual = torch.arange(2 * 3 * 3 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 3, 3, 3, 2, 2)
    batch = {"visual": visual, "modality_mask": torch.ones(2, 6, dtype=torch.bool)}
    stressed, affected = apply_stress(batch, "visual_first_frame")
    assert affected.all()
    assert torch.equal(stressed["modality_mask"], batch["modality_mask"])
    assert torch.equal(stressed["visual"][:, :, 1], visual[:, :, 0])
    assert torch.equal(stressed["visual"][:, :, 2], visual[:, :, 0])
