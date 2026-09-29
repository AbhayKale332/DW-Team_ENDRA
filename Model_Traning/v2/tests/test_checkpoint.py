import torch


def test_state_dict_excludes_encoder(fake_net):
    sd = fake_net.trainable_state_dict()
    assert sd, "expected non-empty trainable state"
    assert not any(k.startswith("encoder.model.") for k in sd)
    assert any(k.startswith("trunk.") for k in sd)
    assert any(k.startswith("head_a.") for k in sd)


def test_save_reload_identical_forward(fake_net, tmp_path):
    fake_net.eval()
    img = torch.randn(1, 3, fake_net.cfg.tile_size, fake_net.cfg.tile_size)
    with torch.no_grad():
        before = fake_net(img)["fused"]

    ckpt = tmp_path / "best.pt"
    torch.save({"model": fake_net.trainable_state_dict()}, ckpt)

    import models.heads as mh

    fresh = mh.DepthWizardNetV2(fake_net.cfg)
    missing, unexpected = fresh.load_state_dict(
        torch.load(ckpt)["model"], strict=False
    )
    # only the stub-encoder keys should be missing
    assert all("encoder" in m for m in missing)
    fresh.eval()
    with torch.no_grad():
        after = fresh(img)["fused"]
    assert torch.allclose(before, after, atol=1e-5)
