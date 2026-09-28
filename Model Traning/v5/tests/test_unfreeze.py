"""v5: the encoder unfreeze keeps the decoder's AdamW state and ramps its LR in."""

import torch

from config import Config
from tests.stub_encoder import use_stub


def _net():
    from models.heads import DepthWizardNet

    c = Config()
    c.tile_size, c.decoder_dim, c.n_bins = 64, 32, 8
    c.grad_checkpoint_encoder = False
    c.freeze_epochs, c.unfreeze_warmup_epochs = 2, 1.0
    return c, DepthWizardNet(c)


def test_ramp_is_linear_from_the_unfreeze_and_resume_safe():
    from train import encoder_lr_ramp

    c = Config()
    c.freeze_epochs, c.unfreeze_warmup_epochs = 2, 1.0
    assert [encoder_lr_ramp(c, e) for e in (1.5, 2.0, 2.25, 2.5, 3.0, 9.0)] == \
        [0.0, 0.0, 0.25, 0.5, 1.0, 1.0]
    c.unfreeze_warmup_epochs = 0.0
    assert encoder_lr_ramp(c, 2.1) == 1.0          # 0 = the v4 step change
    c.freeze_epochs, c.unfreeze_warmup_epochs = 0, 1.0
    assert encoder_lr_ramp(c, 0.1) == 1.0          # trained from the start: no ramp


def test_unfreeze_keeps_decoder_state_and_matches_a_resumed_layout():
    undo = use_stub(hidden=32, patch=16, layers=8)
    try:
        from train import add_encoder_groups, build_optimizer

        cfg, net = _net()
        opt, base = build_optimizer(cfg, net, with_encoder=False)
        # one decoder step so AdamW has state to lose
        net(torch.randn(1, 3, 64, 64))["fused"].mean().backward()
        opt.step()
        dec = next(p for p in net.decoder_parameters())
        state_before = opt.state[dec]["exp_avg"].clone()

        net.encoder.set_frozen(False, 0)
        base2 = add_encoder_groups(cfg, net, opt, base)
        assert torch.equal(opt.state[dec]["exp_avg"], state_before)      # kept
        assert len(base2) == len(opt.param_groups) > len(base)

        # the layout a resumed run rebuilds must be identical, group by group
        opt_r, base_r = build_optimizer(cfg, net, with_encoder=True)
        assert [g["name"] for g in opt.param_groups] == [g["name"] for g in opt_r.param_groups]
        assert [len(g["params"]) for g in opt.param_groups] == \
            [len(g["params"]) for g in opt_r.param_groups]
        assert base2 == base_r
        opt_r.load_state_dict(opt.state_dict())                          # positional load works
    finally:
        undo()
