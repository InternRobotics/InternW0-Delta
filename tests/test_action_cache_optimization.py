from __future__ import annotations

import os
from typing import Optional

import pytest
import torch
import torch.nn as nn

from wam.model.backbones.wan22.wan_video_dit import CrossAttention
from wam.model.modules.mot.mixture_of_transformers import MoT


class _EmptyExpert(nn.Module):
    def __init__(self, num_layers: int = 0) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(nn.Identity() for _ in range(num_layers))
        self.num_heads = 1
        self.attn_head_dim = 1


def _mot(
    *,
    compile_mode: str = "off",
    context_pad_to: int = 0,
    num_layers: int = 0,
    attention_backend: str = "sdpa",
) -> MoT:
    return MoT(
        mixtures={
            "video": _EmptyExpert(num_layers),
            "action": _EmptyExpert(num_layers),
        },
        mot_checkpoint_mixed_attn=False,
        compile_mode=compile_mode,
        compile_gradient_checkpointing=False,
        compile_action_context_pad_to=context_pad_to,
        attention_backend=attention_backend,
    )


def _compiled_sentinel(fn):
    class CompiledSentinel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fn = fn
            self.compiler_parameter = nn.Parameter(torch.ones(()))

        def forward(self, *args, **kwargs):
            return self.fn(*args, **kwargs)

    return CompiledSentinel()


def test_action_cache_compiled_wrappers_are_lazy_and_stable(monkeypatch):
    calls = []

    def fake_compile(fn, **kwargs):
        calls.append((fn, kwargs))
        return _compiled_sentinel(fn)

    monkeypatch.setenv("WAM_MOT_COMPILE_FULLGRAPH", "1")
    monkeypatch.setattr(torch, "compile", fake_compile)
    mot = _mot(compile_mode="default", num_layers=3)
    state_keys = tuple(mot.state_dict())

    first = mot._get_compiled_action_cache_layer(0)
    second = mot._get_compiled_action_cache_layer(0)
    third = mot._get_compiled_action_cache_layer(1)

    assert first is second
    assert third is not first
    assert len(calls) == 2
    assert [call[0]._mot_layer_idx for call in calls] == [0, 1]
    assert calls[0][0].__code__ is not calls[1][0].__code__
    assert all(
        kwargs
        == {
            "backend": "inductor",
            "dynamic": False,
            "fullgraph": True,
            "mode": "default",
        }
        for _, kwargs in calls
    )
    assert tuple(mot.state_dict()) == state_keys


def test_action_cache_layer_rejects_invalid_layer():
    mot = _mot(compile_mode="default", num_layers=2)
    with pytest.raises(IndexError, match="out of bounds"):
        mot._get_compiled_action_cache_layer(2)


def test_forward_action_cache_routes_each_layer(monkeypatch):
    mot = _mot(compile_mode="default", num_layers=2)
    action_tokens = torch.zeros(1, 2, 4)
    action_freqs = torch.randn(2, 1, 2)
    action_t_mod = torch.randn(1, 6, 4)
    action_context = torch.randn(1, 3, 4)
    action_context_mask = torch.ones(1, 2, 3, dtype=torch.bool)
    attention_mask = torch.ones(5, 5, dtype=torch.bool)
    video_cache = [
        {"k": torch.full((1, 3, 4), layer + 1.0), "v": torch.randn(1, 3, 4)}
        for layer in range(2)
    ]
    calls = []

    def fake_get_compiled_action_cache_layer(layer_idx):
        def run(action_embed, *args):
            calls.append((layer_idx, action_embed.clone(), args))
            return action_embed + layer_idx + 1

        return run

    monkeypatch.setattr(
        mot,
        "_get_compiled_action_cache_layer",
        fake_get_compiled_action_cache_layer,
    )

    output = mot.forward_action_with_video_cache(
        action_tokens=action_tokens,
        action_freqs=action_freqs,
        action_t_mod=action_t_mod,
        action_context_payload={
            "context": action_context,
            "mask": action_context_mask,
        },
        video_kv_cache=video_cache,
        attention_mask=attention_mask,
        video_seq_len=3,
    )

    assert [layer for layer, _, _ in calls] == [0, 1]
    assert calls[0][2][0] is action_freqs
    assert calls[0][2][1] is action_t_mod
    assert calls[0][2][2] is action_context
    assert calls[0][2][3] is action_context_mask
    torch.testing.assert_close(calls[0][2][-2], attention_mask[3:, :])
    assert calls[0][2][-1] is None
    torch.testing.assert_close(output, action_tokens + 3)
    assert mot._compiled_action_cache_calls == 2


def test_forward_action_cache_routes_packed_groups(monkeypatch):
    monkeypatch.setenv("WAM_MOT_ACTION_CACHE_GROUP_SIZE", "2")
    mot = _mot(compile_mode="default", num_layers=3)
    action_tokens = torch.zeros(1, 2, 4)
    action_freqs = torch.randn(2, 1, 2)
    action_t_mod = torch.randn(1, 6, 4)
    action_context = torch.randn(1, 3, 4)
    action_context_mask = torch.ones(1, 2, 3, dtype=torch.bool)
    attention_mask = torch.ones(5, 5, dtype=torch.bool)
    video_cache = [
        {"k": torch.randn(1, 3, 4), "v": torch.randn(1, 3, 4)}
        for _ in range(3)
    ]
    context_cache = tuple(
        (torch.randn(1, 3, 4), torch.randn(1, 3, 4)) for _ in range(3)
    )
    context_cache_stacked = (
        torch.stack(tuple(layer[0] for layer in context_cache)),
        torch.stack(tuple(layer[1] for layer in context_cache)),
    )
    video_cache_stacked = (
        torch.stack(tuple(layer["k"] for layer in video_cache)),
        torch.stack(tuple(layer["v"] for layer in video_cache)),
    )
    calls = []

    def fake_group(start_layer, end_layer):
        def run(action_embed, *args):
            calls.append((start_layer, end_layer, action_embed.clone(), args))
            return action_embed + (end_layer - start_layer)

        return run

    monkeypatch.setattr(
        mot,
        "_get_compiled_action_cache_group",
        fake_group,
    )

    output = mot.forward_action_with_video_cache(
        action_tokens=action_tokens,
        action_freqs=action_freqs,
        action_t_mod=action_t_mod,
        action_context_payload={
            "context": action_context,
            "mask": action_context_mask,
            "kv_cache": context_cache,
            "kv_cache_stacked": context_cache_stacked,
        },
        video_kv_cache=video_cache,
        attention_mask=attention_mask,
        video_seq_len=3,
        video_grouped_kv_cache=video_cache_stacked,
    )

    assert [(start, end) for start, end, _, _ in calls] == [(0, 2), (2, 3)]
    torch.testing.assert_close(
        calls[0][3][4], context_cache_stacked[0][0:2]
    )
    torch.testing.assert_close(
        calls[0][3][5], context_cache_stacked[1][0:2]
    )
    torch.testing.assert_close(calls[0][3][6], video_cache_stacked[0][0:2])
    torch.testing.assert_close(calls[0][3][7], video_cache_stacked[1][0:2])
    torch.testing.assert_close(output, action_tokens + 3)
    assert mot._compiled_action_cache_calls == 3


def test_compiled_action_path_reapplies_video_key_mask(monkeypatch):
    mot = _mot(compile_mode="default", num_layers=2, attention_backend="sdpa")
    video_mask = torch.tensor([[True, False, True]])
    video_cache = [
        {
            "k": torch.zeros(1, 3, 4),
            "v": torch.randn(1, 3, 4),
            "mask": video_mask.clone(),
        }
        for _ in range(2)
    ]
    calls = []

    def fake_get_compiled_action_cache_layer(layer_idx):
        def run(action_embed, *args):
            calls.append((layer_idx, args[-1]))
            return action_embed + 1

        return run

    monkeypatch.setattr(
        mot,
        "_get_compiled_action_cache_layer",
        fake_get_compiled_action_cache_layer,
    )

    output = mot.forward_action_with_video_cache(
        action_tokens=torch.zeros(1, 2, 4),
        action_freqs=torch.empty(0),
        action_t_mod=torch.empty(0),
        action_context_payload={
            "context": torch.randn(1, 3, 4),
            "mask": torch.ones(1, 2, 3, dtype=torch.bool),
        },
        video_kv_cache=video_cache,
        attention_mask=torch.ones(5, 5, dtype=torch.bool),
        video_seq_len=3,
    )

    expected = torch.tensor([[True, False, True, True, True]])
    assert [layer for layer, _ in calls] == [0, 1]
    for _, key_mask in calls:
        torch.testing.assert_close(key_mask, expected)
    torch.testing.assert_close(output, torch.zeros(1, 2, 4) + 2)
    assert mot._compiled_action_cache_calls == 2


def test_action_context_is_padded_once_for_compiled_wrapper(monkeypatch):
    mot = _mot(compile_mode="default", context_pad_to=640, num_layers=2)
    action_tokens = torch.zeros(1, 2, 4)
    action_freqs = torch.empty(0)
    action_t_mod = torch.empty(0)
    attention_mask = torch.ones(5, 5, dtype=torch.bool)
    video_cache = [
        {"k": torch.zeros(1, 3, 4), "v": torch.zeros(1, 3, 4)}
        for _ in range(2)
    ]
    wrapper_calls = []

    def fake_get_compiled_action_cache_layer(layer_idx):
        def run(action_embed, *args):
            wrapper_calls.append(
                {
                    "layer_idx": layer_idx,
                    "context": args[2],
                    "mask": args[3],
                }
            )
            return action_embed + 1

        return run

    monkeypatch.setattr(
        mot,
        "_get_compiled_action_cache_layer",
        fake_get_compiled_action_cache_layer,
    )

    for context_len in (3, 5):
        context = torch.arange(context_len * 4, dtype=torch.float32).reshape(
            1, context_len, 4
        )
        mask = torch.ones(1, 2, context_len, dtype=torch.bool)
        output = mot.forward_action_with_video_cache(
            action_tokens=action_tokens,
            action_freqs=action_freqs,
            action_t_mod=action_t_mod,
            action_context_payload={"context": context, "mask": mask},
            video_kv_cache=video_cache,
            attention_mask=attention_mask,
            video_seq_len=3,
        )
        torch.testing.assert_close(output, action_tokens + 2)

    assert [call["layer_idx"] for call in wrapper_calls] == [0, 1, 0, 1]
    for call in wrapper_calls:
        assert call["context"].shape == (1, 640, 4)
        assert call["mask"].shape == (1, 2, 640)
    assert mot._compiled_action_cache_calls == 4


@pytest.mark.parametrize(
    ("compile_mode", "payload_kind"),
    [
        ("off", "complete"),
        ("default", "none"),
        ("default", "missing_context"),
        ("default", "missing_mask"),
    ],
)
def test_incomplete_compile_contract_keeps_eager_path(
    monkeypatch, compile_mode, payload_kind
):
    mot = _mot(compile_mode=compile_mode, num_layers=1)
    context = torch.randn(1, 3, 4)
    context_mask = torch.ones(1, 2, 3, dtype=torch.bool)
    payloads = {
        "complete": {"context": context, "mask": context_mask},
        "none": None,
        "missing_context": {"context": None, "mask": context_mask},
        "missing_mask": {"context": context, "mask": None},
    }

    def eager_mask(**kwargs):
        raise RuntimeError("eager-mask-sentinel")

    def unexpected_compiled_layer(layer_idx):
        raise AssertionError("Incomplete compile inputs must retain eager routing")

    monkeypatch.setattr(mot, "_build_flex_block_mask", eager_mask)
    monkeypatch.setattr(
        mot,
        "_get_compiled_action_cache_layer",
        unexpected_compiled_layer,
    )

    with pytest.raises(RuntimeError, match="eager-mask-sentinel"):
        mot.forward_action_with_video_cache(
            action_tokens=torch.zeros(1, 2, 4),
            action_freqs=torch.empty(0),
            action_t_mod=torch.empty(0),
            action_context_payload=payloads[payload_kind],
            video_kv_cache=[{"k": torch.zeros(1, 3, 4), "v": torch.zeros(1, 3, 4)}],
            attention_mask=torch.ones(5, 5, dtype=torch.bool),
            video_seq_len=3,
        )
    assert mot._compiled_action_cache_calls == 0


def test_compilable_action_cache_layer_forces_sdpa(monkeypatch):
    mot = _mot(compile_mode="default", num_layers=1)
    action_embed = torch.randn(1, 2, 4)
    q_action = torch.full((1, 2, 4), 1.0)
    k_action = torch.full((1, 2, 4), 2.0)
    v_action = torch.full((1, 2, 4), 3.0)
    video_k = torch.full((1, 3, 4), 4.0)
    video_v = torch.full((1, 3, 4), 5.0)
    mixed_output = torch.full_like(action_embed, 9.0)
    calls = {}

    def fake_build_attention_io(**kwargs):
        calls["attention_io"] = kwargs
        return (
            q_action,
            k_action,
            v_action,
            action_embed,
            "gate_msa",
            "shift_mlp",
            "scale_mlp",
            "gate_mlp",
            False,
        )

    def fake_mixed_attention(**kwargs):
        calls["mixed"] = kwargs
        return mixed_output

    def fake_apply_post(**kwargs):
        calls["post"] = kwargs
        return kwargs["mixed_slice"]

    monkeypatch.setattr(mot, "_build_expert_attention_io", fake_build_attention_io)
    monkeypatch.setattr(mot, "_mixed_attention", fake_mixed_attention)
    monkeypatch.setattr(mot, "_apply_post_with_optional_checkpoint", fake_apply_post)

    context_k = torch.randn(1, 3, 4)
    context_v = torch.randn(1, 3, 4)
    output = mot._forward_compilable_action_cache_layer(
        0,
        action_embed,
        torch.empty(0),
        torch.empty(0),
        torch.randn(1, 3, 4),
        torch.ones(1, 2, 3, dtype=torch.bool),
        context_k,
        context_v,
        video_k,
        video_v,
        torch.ones(2, 5, dtype=torch.bool),
    )

    torch.testing.assert_close(output, mixed_output)
    torch.testing.assert_close(
        calls["mixed"]["k_cat"], torch.cat([video_k, k_action], dim=1)
    )
    torch.testing.assert_close(
        calls["mixed"]["v_cat"], torch.cat([video_v, v_action], dim=1)
    )
    assert calls["mixed"]["force_sdpa"] is True
    assert calls["post"]["context_kv"] == (context_k, context_v)


def test_cross_attention_kv_cache_matches_projection():
    torch.manual_seed(7)
    attention = CrossAttention(hidden_dim=8, attn_head_dim=4, num_heads=2, context_dim=12)
    x = torch.randn(2, 5, 8)
    context = torch.randn(2, 7, 12)
    context_mask = torch.ones(2, 5, 7, dtype=torch.bool)

    context_k = attention.norm_k(attention.k(context))
    context_v = attention.v(context)

    torch.testing.assert_close(
        attention(x, context, ctx_mask=context_mask),
        attention.forward_with_kv_cache(
            x, context_k, context_v, ctx_mask=context_mask
        ),
    )


def test_prefill_action_context_kv_cache_matches_cross_attention_projection():
    class Expert(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            "cross_attn": CrossAttention(
                                hidden_dim=8,
                                attn_head_dim=4,
                                num_heads=2,
                                context_dim=12,
                            )
                        }
                    )
                    for _ in range(2)
                ]
            )
            self.uses_vlm_conditioning = True

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.mixtures = nn.ModuleDict({"action": Expert()})
            self.num_layers = 2

    model = Model()
    mot = _mot(num_layers=2)
    # Share the real expert while retaining MoT's initialization and methods.
    mot.mixtures["action"] = model.mixtures["action"]
    context = torch.randn(2, 6, 12)
    cache = mot.prefill_action_context_kv_cache(context)

    assert len(cache) == 2
    for block, (context_k, context_v) in zip(mot.mixtures["action"].blocks, cache):
        torch.testing.assert_close(
            context_k, block["cross_attn"].norm_k(block["cross_attn"].k(context))
        )
        torch.testing.assert_close(context_v, block["cross_attn"].v(context))
