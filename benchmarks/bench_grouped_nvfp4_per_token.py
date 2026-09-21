"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Compare masked grouped NVFP4 with existing per-token CUDA/CuTe quantizers
and the non-per-token grouped quantizer. Reference per-token paths call the
existing API once per nonempty expert and pack its outputs for grouped GEMM;
packing is included in their timing. Valid row counts are known on the host
for these references. The new path consumes device-side counts directly.

Reports quantization, GEMM and quantization+GEMM latency, plus reconstruction
and GEMM error against FP32 inputs. Weight quantization and calibration are
outside timing for every path. GEMM requires SM100/SM103; quantization can be
measured separately on SM12x. Default timing uses warm-L2 CUDA graphs; CUPTI
is selectable. Existing FLASHINFER_NVFP4_* settings are recorded in JSON.

Example:
    python benchmarks/bench_grouped_nvfp4_per_token.py --output results.json
    python benchmarks/bench_grouped_nvfp4_per_token.py --quant-only --output gb10.json
"""

import argparse
import json
import os
import platform
import statistics
from pathlib import Path

import torch

from flashinfer import nvfp4_quantize, scaled_fp4_grouped_quantize
from flashinfer.gemm import grouped_gemm_nt_masked
from flashinfer.quantization.fp4_quantization import NVFP4_QUANT_ENV_VARS
from flashinfer.quantization.nvfp4_quantization_utils import (
    current_nvfp4_4over6_config,
    make_nvfp4_global_scale,
)
from flashinfer.testing import bench_gpu_time


def dequantize(q, sf, rows):
    """Decode grouped 128x4 scales and E2M1 bytes with independent torch ops."""
    e, m, packed_k = q.permute(2, 0, 1).shape
    k = packed_k * 2
    raw_sf = sf.permute(5, 2, 4, 0, 1, 3)
    linear_sf = raw_sf.transpose(2, 4).reshape(e, -1, ((k + 63) // 64) * 4)
    linear_sf = linear_sf[:, :m, : k // 16].float()
    packed = q.permute(2, 0, 1)
    indices = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=q.device,
    )
    return lut[indices] * linear_sf.repeat_interleave(16, -1) * rows[:, :, None]


def error_metrics(actual, reference, valid):
    a, r = actual[valid].float(), reference[valid].float()
    return {
        "relative_l2": ((a - r).norm() / r.norm().clamp_min(1e-30)).item(),
        "mean_row_relative_l2": ((a - r).norm(dim=-1) / r.norm(dim=-1).clamp_min(1e-30))
        .mean()
        .item(),
        "cosine": torch.nn.functional.cosine_similarity(
            a.flatten(), r.flatten(), dim=0
        ).item(),
        "max_abs": (a - r).abs().max().item(),
    }


def reference_grouped(x, counts, scale_inv, backend):
    e, m, k = x.shape
    pm, pk = (m + 127) // 128, (k + 63) // 64
    q = torch.empty(e, m, k // 2, device=x.device, dtype=torch.uint8)
    sf = torch.zeros(e, pm, pk, 32, 4, 4, device=x.device, dtype=torch.uint8)
    rows = torch.zeros(e, m, device=x.device, dtype=torch.float32)
    for group, count in enumerate(counts):
        if not count:
            continue
        qr, sr, rr = nvfp4_quantize(
            x[group, :count],
            scale_inv,
            backend=backend,
            per_token_activation=True,
            enable_pdl=False,
        )
        q[group, :count].copy_(qr)
        sf[group, : (count + 127) // 128].copy_(sr.view(-1, pk, 32, 4, 4))
        rows[group, :count].copy_(rr)
    return (
        q.permute(1, 2, 0),
        sf.view(torch.float8_e4m3fn).permute(3, 4, 1, 5, 2, 0),
        rows,
    )


def benchmark_case(args, e, m, k, n):
    torch.manual_seed(42)
    counts = [
        m if i % 3 == 0 else max(1, m // 2) if i % 3 == 1 else 0 for i in range(e)
    ]
    mask = torch.tensor(counts, device="cuda", dtype=torch.int32)
    valid = torch.arange(m, device="cuda")[None, :] < mask[:, None]
    x = torch.randn(e, m, k, device="cuda", dtype=getattr(torch, args.dtype))
    if args.distribution == "row-dynamic":
        x *= torch.logspace(-2, 2, m, device="cuda").view(1, m, 1)
    x.masked_fill_(~valid[:, :, None], 0)
    config = current_nvfp4_4over6_config()
    scale_inv = make_nvfp4_global_scale(
        x[0], per_token_activation=True, nvfp4_4over6_config=config
    )
    encode = torch.stack(
        [
            make_nvfp4_global_scale(
                x[g], per_token_activation=False, nvfp4_4over6_config=config
            ).reshape(())
            if counts[g]
            else torch.ones((), device="cuda")
            for g in range(e)
        ]
    )

    def grouped():
        return scaled_fp4_grouped_quantize(
            x, mask, scale_inv, per_token_activation=True
        )

    def non_per_token():
        q, sf = scaled_fp4_grouped_quantize(x, mask, encode)
        return q, sf, None

    # CUDA takes a host scalar; converting a device tensor inside capture is illegal.
    scale_inv_host = scale_inv.item()
    paths = {
        "grouped_per_token": grouped,
        "expert_loop_per_token_cuda": lambda: reference_grouped(
            x, counts, scale_inv_host, "cuda"
        ),
        "expert_loop_per_token_cute": lambda: reference_grouped(
            x, counts, scale_inv, "cute-dsl"
        ),
        "grouped_non_per_token": non_per_token,
    }

    def time_us(fn):
        fn()
        times = bench_gpu_time(
            fn,
            enable_cupti=args.cupti,
            use_cuda_graph=True,
            cold_l2_cache=False,
            dry_run_iters=5,
            repeat_iters=args.repeat,
        )
        return {
            "median_us": statistics.median(times) * 1000,
            "min_us": min(times) * 1000,
            "max_us": max(times) * 1000,
        }

    gemm_enabled = not args.quant_only and torch.cuda.get_device_capability() in (
        (10, 0),
        (10, 3),
    )
    if gemm_enabled:
        b = torch.randn(e, n, k, device="cuda", dtype=x.dtype)
        b_encode = torch.stack(
            [
                make_nvfp4_global_scale(
                    t, per_token_activation=False, nvfp4_4over6_config=config
                ).reshape(())
                for t in b
            ]
        )
        bq, bs = scaled_fp4_grouped_quantize(
            b, torch.full((e,), n, device="cuda", dtype=torch.int32), b_encode
        )
        out = torch.empty(e, m, n, device="cuda", dtype=x.dtype).permute(1, 2, 0)
        reference = x.float() @ b.float().transpose(1, 2)
        alpha_token = b_encode.reciprocal()
        alpha_tensor = (encode * b_encode).reciprocal()

        def gemm(result):
            q, sf, rows = result
            grouped_gemm_nt_masked(
                (q, sf),
                (bq, bs),
                out,
                mask,
                ab_dtype="float4_e2m1fn",
                sf_dtype="float8_e4m3fn",
                c_dtype=args.dtype,
                sf_vec_size=16,
                a_per_token_scale=rows,
                alpha=alpha_tensor if rows is None else alpha_token,
                alpha_dtype="float32",
                mma_tiler_mn=(128, 128),
                cluster_shape_mn=(1, 1),
            )

    results = {}
    decoded_token = None
    gemm_token = None
    for name, fn in paths.items():
        result = fn()
        q, sf, rows = result
        decoded = dequantize(
            q, sf, encode.reciprocal()[:, None].expand(e, m) if rows is None else rows
        )
        assert torch.isfinite(decoded[valid]).all(), name
        metrics = {
            "quantization": time_us(fn),
            "reconstruction_error": error_metrics(decoded, x, valid),
        }
        if name == "grouped_per_token":
            decoded_token = decoded
        elif rows is not None:
            metrics["vs_grouped_per_token"] = error_metrics(
                decoded, decoded_token, valid
            )
            torch.testing.assert_close(
                decoded[valid], decoded_token[valid], rtol=1e-5, atol=1e-5
            )
        if gemm_enabled:
            gemm(result)
            if name == "grouped_per_token":
                gemm_token = out.clone()
            elif rows is not None:
                torch.testing.assert_close(
                    out.permute(2, 0, 1)[valid],
                    gemm_token.permute(2, 0, 1)[valid],
                    rtol=0,
                    atol=0,
                )
            metrics["gemm_error"] = error_metrics(
                out.permute(2, 0, 1), reference, valid
            )
            metrics["gemm"] = time_us(lambda: gemm(result))
            metrics["quantization_and_gemm"] = time_us(lambda: gemm(fn()))
        results[name] = metrics
    return {
        "shape": {"experts": e, "m": m, "k": k, "n": n},
        "valid_rows": counts,
        "gemm_enabled": gemm_enabled,
        "paths": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--m", type=int, nargs="+", default=[32, 129, 512])
    parser.add_argument("--k", type=int, nargs="+", default=[4096])
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument(
        "--distribution", choices=["normal", "row-dynamic"], default="row-dynamic"
    )
    parser.add_argument("--quant-only", action="store_true")
    parser.add_argument("--cupti", action="store_true")
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import importlib.metadata
    import flashinfer

    report = {
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": torch.cuda.get_device_capability(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cutlass_dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
            "flashinfer_source": flashinfer.__file__,
            "platform": platform.platform(),
            "quantization_env": {key: os.getenv(key) for key in NVFP4_QUANT_ENV_VARS},
        },
        "arguments": {**vars(args), "output": str(args.output)},
        "cases": [],
    }
    with torch.inference_mode():
        for e in args.experts:
            for m in args.m:
                for k in args.k:
                    case = benchmark_case(args, e, m, k, args.n)
                    report["cases"].append(case)
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    print(json.dumps(case), flush=True)


if __name__ == "__main__":
    main()
