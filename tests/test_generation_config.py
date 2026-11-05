"""Sampling settings survive the model's own generation_config.json (temperature 0.6, top_p 0.9).

A tiny random Llama carries the backdoored repos' generation config. With our explicit
GenerationConfig, the processed sampling scores must equal the raw logits at temperature 1
(no truncation), and equal logits / 0.7 at Redwood's temperature, with no -inf masking.
"""

import torch

from harness.sample import left_pad, make_generation_config

REPO_GENERATION_CONFIG = {
    "bos_token_id": 128000,
    "do_sample": True,
    "eos_token_id": [128001, 128008, 128009],
    "pad_token_id": 128256,
    "temperature": 0.6,
    "top_p": 0.9,
}


def tiny_model():
    from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=128257, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=256)
    m = LlamaForCausalLM(cfg).eval()
    m.generation_config = GenerationConfig(**REPO_GENERATION_CONFIG)
    return m


def run(model, temperature):
    s = {"do_sample": True, "temperature": temperature, "top_p": 1.0, "top_k": 0, "max_new_tokens": 4}
    gc = make_generation_config(s, 128004, (128009, 128001, 128008))
    ids, attn = left_pad([[128000, 5, 6, 7], [128000, 9]], 128004)
    torch.manual_seed(1)
    out = model.generate(input_ids=ids, attention_mask=attn, generation_config=gc,
                         output_scores=True, output_logits=True, return_dict_in_generate=True)
    return out


def test_temperature_one_no_truncation():
    out = run(tiny_model(), 1.0)
    for s, l in zip(out.scores, out.logits):
        assert torch.isfinite(s).all()
        assert torch.allclose(s, l)


def test_temperature_point_seven_scaling_only():
    out = run(tiny_model(), 0.7)
    for s, l in zip(out.scores, out.logits):
        assert torch.isfinite(s).all()
        assert torch.allclose(s, l / 0.7, atol=1e-5)
