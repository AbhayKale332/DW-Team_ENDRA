from config import Config, parse_config, safe_config_dict


def test_defaults_and_validate():
    c = Config().validate()
    assert c.tile_size % 16 == 0
    assert c.amp_dtype in ("bf16", "fp16")
    assert "gamus" in c.dataset_list()


def test_cli_override():
    c = parse_config(["--batch_size", "7", "--datasets", "gamus", "--tta", "false"])
    assert c.batch_size == 7
    assert c.dataset_list() == ["gamus"]
    assert c.tta is False


def test_smoke_shrinks():
    c = parse_config(["--smoke", "true"])
    assert c.smoke and c.finetune_epochs == 1 and c.val_tiles == 6
    assert c.share_cloudflared is False


def test_sampler_weight_map():
    c = parse_config(["--sampler_weights", "gamus:3,geonrw:1"])
    assert c.sampler_weight_map() == {"gamus": 3.0, "geonrw": 1.0}


def test_token_never_persisted():
    c = Config()
    c.hf_token = "secret"
    assert "hf_token" not in safe_config_dict(c)


def test_skip_pretrain():
    c = parse_config(["--skip_pretrain", "true"])
    assert c.pretrain_dataset == ""
