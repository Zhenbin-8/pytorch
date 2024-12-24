# Owner(s): ["module: sdpa"]

import os
import unittest
from collections import namedtuple
from functools import partial

import pytorch_openreg  # noqa: F401

import torch
import torch.utils.cpp_extension
from torch.nn.attention import SDPBackend
from torch.testing._internal.common_nn import NNTestCase
from torch.testing._internal.common_utils import (
    IS_FBCODE,
    run_tests,
    skipIfTorchDynamo,
    TEST_XPU,
)


SdpaShape = namedtuple("Sdpa_Shape", ["batch", "num_heads", "seq_len", "head_dim"])


def _scaled_dot_product_fused_attention_overrideable(
    query: torch.Tensor,
    key,
    value,
    attn_bias=None,
    dropout_p=None,
    is_causal=None,
    return_debug_mask=None,
    scale=None,
):
    batch_size = query.size(0)
    num_heads = query.size(1)
    head_dim_v = value.size(3)
    max_seqlen_q = query.size(2)
    max_seqlen_kv = key.size(2)

    dtype = query.dtype
    device = query.device
    layout = query.layout
    output = torch.empty(
        (batch_size, num_heads, max_seqlen_q, head_dim_v),
        dtype=dtype,
        layout=layout,
        device=device,
    )
    logsumexp = torch.empty(
        (batch_size, num_heads, max_seqlen_q),
        dtype=torch.float,
        layout=layout,
        device=device,
    )
    debug_attn_mask = torch.empty(
        (batch_size, num_heads, max_seqlen_q, max_seqlen_kv),
        dtype=torch.float,
        layout=layout,
        device=device,
    )

    philox_seed = torch.empty((), dtype=torch.long)
    philox_offset = torch.empty((), dtype=torch.long)
    return (
        output,
        logsumexp,
        torch.tensor(()),
        torch.tensor(()),
        max_seqlen_q,
        max_seqlen_kv,
        philox_seed,
        philox_offset,
        debug_attn_mask,
    )


def _scaled_dot_product_fused_attention_overrideable_backward(
    grad_out,
    query,
    key,
    value,
    attn_bias,
    grad_input_mask,
    out,
    logsumexp,
    cum_seq_q,
    cum_seq_k,
    imax_q,
    imax_k,
    dropout_p,
    is_causal,
    philox_seed,
    philox_offset,
    scale=None,
):
    return (
        torch.empty_like(query),
        torch.empty_like(key),
        torch.empty_like(value),
        torch.empty_like(attn_bias),
    )


@unittest.skipIf(TEST_XPU, "XPU does not support cppextension currently")
@unittest.skipIf(
    IS_FBCODE,
    "Ninja is required to load C++ extensions and it's not compatible with Buck ",
)
class TestSDPAPrivateUse1Only(NNTestCase):
    @classmethod
    def setUpClass(cls):
        torch.testing._internal.common_utils.remove_cpp_extensions_build_root()
        cls.module = torch.utils.cpp_extension.load(
            name="custom_device_extension",
            sources=[
                f"{'test/' if not os.getcwd().endswith('test') else ''}cpp_extensions/open_registration_extension.cpp",
            ],
            extra_include_paths=["cpp_extensions"],
            extra_cflags=["-g"],
            verbose=True,
        )

        cls.lib = torch.library.Library("aten", "IMPL")  # noqa: TOR901
        cls.lib.impl(
            "_scaled_dot_product_fused_attention_overrideable",
            _scaled_dot_product_fused_attention_overrideable,
            dispatch_key="PrivateUse1",
        )
        cls.lib.impl(
            "_scaled_dot_product_fused_attention_overrideable_backward",
            _scaled_dot_product_fused_attention_overrideable_backward,
            dispatch_key="PrivateUse1",
        )

    @classmethod
    def tearDownClass(cls):
        cls.lib._destroy()

    @skipIfTorchDynamo()
    def test_fused_sdp_choice_privateuseone(self):
        batch_size, seq_len, num_heads, head_dim = 4, 256, 2, 128
        make_tensor = partial(torch.rand, device="cpu", dtype=torch.float16)
        shape = SdpaShape(batch_size, num_heads, seq_len, head_dim)
        q_cpu, k_cpu, v_cpu = make_tensor(shape), make_tensor(shape), make_tensor(shape)
        q_privateuse1 = q_cpu.to("openreg")
        k_privateuse1 = k_cpu.to("openreg")
        v_privateuse1 = v_cpu.to("openreg")
        assert (
            torch._fused_sdp_choice(q_privateuse1, k_privateuse1, v_privateuse1)
            == SDPBackend.OVERRIDEABLE.value
        )

    def test_scaled_dot_product_fused_attention_overrideable(self):
        batch_size, seq_len, num_heads, head_dim = 4, 256, 2, 128
        make_tensor = partial(torch.rand, device="cpu", dtype=torch.float16)
        shape = SdpaShape(batch_size, num_heads, seq_len, head_dim)
        q_cpu, k_cpu, v_cpu = make_tensor(shape), make_tensor(shape), make_tensor(shape)
        q_privateuse1 = q_cpu.to("openreg")
        k_privateuse1 = k_cpu.to("openreg")
        v_privateuse1 = v_cpu.to("openreg")
        torch.nn.functional.scaled_dot_product_attention(
            q_privateuse1, k_privateuse1, v_privateuse1, attn_mask=None, dropout_p=0.0
        )

    def test_scaled_dot_product_fused_attention_overrideable_backward(self):
        batch_size, seq_len, num_heads, head_dim = 4, 256, 2, 128
        make_tensor = partial(
            torch.rand, device="cpu", dtype=torch.float16, requires_grad=True
        )
        shape = (batch_size, num_heads, seq_len, head_dim)
        q_cpu, k_cpu, v_cpu = make_tensor(shape), make_tensor(shape), make_tensor(shape)
        attn_mask = make_tensor((batch_size, num_heads, seq_len, seq_len))
        q_privateuse1 = q_cpu.to("openreg")
        k_privateuse1 = k_cpu.to("openreg")
        v_privateuse1 = v_cpu.to("openreg")
        attn_mask_privateuse1 = attn_mask.to("openreg")
        (
            output,
            logsumexp,
            cum_seq_q,
            cum_seq_k,
            max_q,
            max_k,
            philox_seed,
            philox_offset,
            debug_attn_mask,
        ) = torch.ops.aten._scaled_dot_product_fused_attention_overrideable(
            q_privateuse1, k_privateuse1, v_privateuse1, attn_bias=attn_mask_privateuse1
        )

        rand_upward = torch.rand(
            shape, device="cpu", dtype=torch.float16, requires_grad=False
        )
        rand_upward_privateuse1 = rand_upward.to("openreg")
        grad_input_mask = [True, True, True, True]
        grad_q, grad_k, grad_v, grad_attn_mask = (
            torch.ops.aten._scaled_dot_product_fused_attention_overrideable_backward(
                rand_upward_privateuse1,
                q_privateuse1,
                k_privateuse1,
                v_privateuse1,
                attn_mask_privateuse1,
                grad_input_mask,
                output,
                logsumexp,
                cum_seq_q,
                cum_seq_k,
                max_q,
                max_k,
                dropout_p=0.0,
                is_causal=False,
                philox_seed=philox_seed,
                philox_offset=philox_offset,
            )
        )


if __name__ == "__main__":
    run_tests()
