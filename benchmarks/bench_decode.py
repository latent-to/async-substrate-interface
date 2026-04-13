"""
Benchmark bt_decode vs cyscale for realistic Bittensor query_map and query scenarios.

Record fixtures from a live node (one-time, requires network):
    python benchmarks/bench_decode.py --record benchmarks/decode_fixtures.json

Run benchmark (offline, no network required after recording):
    python benchmarks/bench_decode.py benchmarks/decode_fixtures.json
    python benchmarks/bench_decode.py benchmarks/decode_fixtures.json --iters 500
"""

import argparse
import asyncio
import json
import os
import sys
import timeit

# ---------------------------------------------------------------------------
# Path setup — can be run from either repo
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_CY_SCALE_PATH = os.path.expanduser("~/Git/cy-scale-codec")

for _p in (_REPO_ROOT, _CY_SCALE_PATH):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from bt_decode import PortableRegistry
from bt_decode import decode as bt_decode_one
from bt_decode import decode_list as bt_decode_many

import scalecodec

from scalecodec.base import RuntimeConfigurationObject
from scalecodec import ScaleBytes
from scalecodec.type_registry import load_type_registry_preset
from scalecodec.utils.ss58 import ss58_encode as _ss58_encode

print(f"scalecodec: {scalecodec.__file__}", flush=True)

# ---------------------------------------------------------------------------
# cyscale helpers
# ---------------------------------------------------------------------------


def _init_cyscale(metadata_hex: str, ss58_format: int = 42):
    """Initialize RuntimeConfigurationObject with portable registry."""
    rc = RuntimeConfigurationObject()
    rc.update_type_registry(load_type_registry_preset("core"))
    rc.update_type_registry(load_type_registry_preset("legacy"))
    meta = rc.create_scale_object("MetadataVersioned", ScaleBytes(metadata_hex))
    meta.decode()
    rc.add_portable_registry(meta)
    rc.ss58_format = ss58_format
    return rc


def cyscale_batch_decode(
    type_strings: list, rc: RuntimeConfigurationObject, data_list: list
):
    return rc.batch_decode(type_strings, data_list)


# ---------------------------------------------------------------------------
# bt_decode + SS58 post-processing
#
# bt_decode does not perform SS58 encoding. To make the comparison fair we
# apply ss58_encode to any result where the type string is "scale_info::0"
# (AccountId32) and the returned value has the shape bt_decode uses for it:
# a single-element tuple wrapping a 32-element int sequence.
# ---------------------------------------------------------------------------

_SS58_FORMAT = 42


def _apply_ss58_recursive(val):
    """
    Recursively walk a bt_decode result and SS58-encode any raw AccountId32
    values. bt_decode represents AccountId32 as a 1-element tuple wrapping a
    32-element int tuple: ((b0, b1, ..., b31),). This shape is structurally
    unambiguous for Substrate types.
    """
    if (
        isinstance(val, (tuple, list))
        and len(val) == 1
        and isinstance(val[0], (tuple, list))
        and len(val[0]) == 32
        and all(isinstance(b, int) for b in val[0])
    ):
        return _ss58_encode(bytes(val[0]), _SS58_FORMAT)
    if isinstance(val, (tuple, list)):
        return type(val)(_apply_ss58_recursive(v) for v in val)
    return val


def bt_decode_many_with_ss58(type_strings, registry, bytes_list):
    """bt_decode_many + SS58 post-processing to match cyscale output."""
    return [
        _apply_ss58_recursive(v)
        for v in bt_decode_many(type_strings, registry, bytes_list)
    ]


def bt_decode_one_with_ss58(type_string, registry, data):
    """bt_decode_one + SS58 post-processing to match cyscale output."""
    return _apply_ss58_recursive(bt_decode_one(type_string, registry, data))


# ---------------------------------------------------------------------------
# Record mode
# ---------------------------------------------------------------------------


async def _record(output_path: str):
    """Connect to a live Bittensor node, capture real decode inputs, save fixtures."""
    from tests.helpers.settings import LATENT_LITE_ENTRYPOINT
    from async_substrate_interface.async_substrate import AsyncSubstrateInterface

    output_path = os.path.join(os.path.dirname(__file__), output_path)

    # Scenarios to capture: (module, storage_fn, params, page_size, label)
    _QUERIES = [
        ("SubtensorModule", "Uids", [], 100, "Uids (u16, N=100)"),
        ("SubtensorModule", "Stake", [], 100, "Stake (u64, N=100)"),
        ("SubtensorModule", "TotalHotkeyAlpha", [], 100, "TotalHotkeyAlpha (N=100)"),
        ("SubtensorModule", "Neurons", [1], 20, "Neurons netuid=1 (struct, N=20)"),
    ]

    captured: dict[str, dict] = {}

    import async_substrate_interface.utils.decoding as _decoding_mod

    print("Connecting to node…", flush=True)
    async with AsyncSubstrateInterface(
        LATENT_LITE_ENTRYPOINT, ss58_format=42, chain_name="Bittensor"
    ) as substrate:
        block_hash = substrate.last_block_hash
        print(f"Block: {block_hash}", flush=True)

        # Fetch runtime to populate metadata_v15 + registry
        runtime = await substrate.init_runtime(block_hash=block_hash)

        # Serialize PortableRegistry for offline bt_decode init
        registry_json = runtime.registry.registry

        # Get V14 metadata hex via state_getMetadata for cyscale init
        meta_rpc = await substrate.rpc_request("state_getMetadata", [block_hash])
        metadata_hex = meta_rpc["result"]

        print("Fetching query_map scenarios…", flush=True)
        for module, storage_fn, params, page_size, label in _QUERIES:
            try:
                # Monkey-patch _async_decode_scale_list_with_runtime to capture inputs
                _scenario_capture = {}
                _orig_fn = _decoding_mod._decode_scale_list_with_runtime

                def _make_capture(lbl, store, orig):
                    async def _p(type_strings, bytes_list, rt, return_scale_obj=False):
                        if lbl not in store:
                            store[lbl] = {
                                "type_strings": list(type_strings),
                                "bytes_list": [
                                    b.hex() if isinstance(b, (bytes, bytearray)) else b
                                    for b in bytes_list
                                ],
                            }
                        return await orig(
                            type_strings, bytes_list, rt, return_scale_obj
                        )

                    return _p

                _decoding_mod._decode_scale_list_with_runtime = _make_capture(
                    label, _scenario_capture, _orig_fn
                )

                qm = await substrate.query_map(
                    module,
                    storage_fn,
                    params=params,
                    block_hash=block_hash,
                    page_size=page_size,
                    fully_exhaust=False,
                )
                async for _ in qm:
                    if label in _scenario_capture:
                        break  # one page is enough

                _decoding_mod._decode_scale_list_with_runtime = _orig_fn

                if label in _scenario_capture:
                    captured[label] = _scenario_capture[label]
                    n = len(_scenario_capture[label]["type_strings"])
                    print(f"  Captured '{label}': {n} type strings", flush=True)
                else:
                    print(f"  WARNING: no data captured for '{label}'", flush=True)

            except Exception as e:
                print(f"  SKIP '{label}': {e}", flush=True)
                _decoding_mod._decode_scale_list_with_runtime = _orig_fn

    fixture = {
        "block_hash": block_hash,
        "registry_json": registry_json,
        "metadata_hex": metadata_hex,
        "scenarios": captured,
    }
    with open(output_path, "w+") as f:
        json.dump(fixture, f)
    print(f"\nFixtures saved to {output_path}  ({len(captured)} scenarios)")


# ---------------------------------------------------------------------------
# Benchmark mode
# ---------------------------------------------------------------------------

_W = 70


def _header(title: str):
    print(f"\n{'─' * _W}")
    print(f"  {title}")
    print(f"{'─' * _W}")
    print(f"  {'Scenario':<38}  {'bt+ss58':>10}  {'cy_batch':>10}  {'speedup':>7}")
    print(f"  {'─' * 38}  {'─' * 10}  {'─' * 10}  {'─' * 7}")


def _header_wide(title: str):
    W2 = 90
    print(f"\n{'─' * W2}")
    print(f"  {title}")
    print(f"{'─' * W2}")
    print(
        f"  {'Scenario':<36}  {'bt+ss58 old':>11}  {'bt+ss58 new':>11}  {'cy old':>10}  {'cy new':>10}  {'cy gain':>8}"
    )
    print(f"  {'─' * 36}  {'─' * 10}  {'─' * 10}  {'─' * 10}  {'─' * 10}  {'─' * 8}")


def _row(label: str, bt_us: float, cy_us: float):
    speedup = bt_us / cy_us if cy_us > 0 else float("inf")
    tag = "cy" if speedup > 1.05 else ("bt" if speedup < 0.95 else "  ")
    print(f"  {label:<38}  {bt_us:>10.1f}  {cy_us:>10.1f}  {speedup:>6.1f}× {tag}")


def run(fn, iters: int) -> float:
    """Return µs/call."""
    return timeit.timeit(fn, number=iters) / iters * 1e6


def bench(fixture_path: str, iters: int):
    with open(fixture_path) as f:
        fixture = json.load(f)

    print(f"Block: {fixture['block_hash']}", flush=True)

    # --- Initialize bt_decode ---
    registry = PortableRegistry.from_json(fixture["registry_json"])

    # --- Initialize cyscale ---
    rc = _init_cyscale(fixture["metadata_hex"], ss58_format=42)

    scenarios = fixture["scenarios"]

    # -----------------------------------------------------------------------
    # Batch decode (query_map path)
    # -----------------------------------------------------------------------
    _header("Batch decode  (µs per page)")

    for label, data in scenarios.items():
        type_strings = data["type_strings"]
        bytes_list = [bytes.fromhex(h) for h in data["bytes_list"]]
        n = len(type_strings) // 2  # keys + values; show N items

        if n == 0:
            continue

        bt_us = run(
            lambda ts=type_strings, r=registry, bl=bytes_list: bt_decode_many_with_ss58(
                ts, r, bl
            ),
            iters,
        )
        cy_us = run(
            lambda ts=type_strings, r=rc, bl=bytes_list: cyscale_batch_decode(
                ts, r, bl
            ),
            iters,
        )
        _row(f"{label}", bt_us, cy_us)

    # -----------------------------------------------------------------------
    # Single-item decode (query / runtime_call path)
    # -----------------------------------------------------------------------
    _header("Single decode  (µs/call)")

    for label, data in scenarios.items():
        if not data["type_strings"]:
            continue
        ts = data["type_strings"][0]
        raw = bytes.fromhex(data["bytes_list"][0])

        bt_us = run(
            lambda t=ts, r=registry, b=raw: bt_decode_one_with_ss58(t, r, b),
            iters,
        )
        cy_us = run(
            lambda t=ts, r=rc, b=raw: rc.batch_decode([t], [b])[0],
            iters,
        )
        _row(f"{label[:38]}", bt_us, cy_us)

    # -----------------------------------------------------------------------
    # Synthetic: single-key optimization (hash prefix skip)
    # -----------------------------------------------------------------------
    # Extract type IDs used in the fixture so we don't hardcode them.
    # For the Uids scenario: key type "[u8;0], scale_info::U16, [u8;16], scale_info::AccountId"
    # We know value type scale_info::40 = u16 from the Uids value type.
    # Simulate N=100 Blake2_128Concat-hashed keys (before vs after the prefix skip).
    import re as _re

    _synth_n = 100

    # Derive type IDs from scenarios present in the fixture
    _type_ids = {}
    for _lbl, _data in scenarios.items():
        if not _data["type_strings"]:
            continue
        # value type is a bare scale_info::N
        _vt = _data["type_strings"][len(_data["type_strings"]) // 2]
        _m = _re.match(r"^scale_info::(\d+)$", _vt)
        if _m:
            _type_ids.setdefault(_lbl, _m.group(0))

    if len(_type_ids) >= 1:
        _header_wide(
            f"Single-key hash-prefix skip  (µs per {_synth_n}-item page, synthetic)"
        )

        for _lbl, _ts_bare in _type_ids.items():
            _ts_wrapped = f"([u8; 16], {_ts_bare})"

            # Determine per-item byte size from the fixture value bytes
            _raw_val_bytes = [
                bytes.fromhex(h)
                for h in scenarios[_lbl]["bytes_list"][
                    len(scenarios[_lbl]["bytes_list"]) // 2 :
                ]
            ]
            _item_size = len(_raw_val_bytes[0]) if _raw_val_bytes else 4

            # Generate synthetic random key payloads
            _rng = __import__("random")
            _rng.seed(42)
            _bare_bytes = [
                bytes(_rng.randrange(256) for _ in range(_item_size))
                for _ in range(_synth_n)
            ]
            _wrapped_bytes = [
                bytes(_rng.randrange(256) for _ in range(16)) + b for b in _bare_bytes
            ]
            _ts_wrapped_list = [_ts_wrapped] * _synth_n
            _ts_bare_list = [_ts_bare] * _synth_n

            _bt_old = run(
                lambda ts=_ts_wrapped_list,
                r=registry,
                bl=_wrapped_bytes: bt_decode_many_with_ss58(ts, r, bl),
                iters,
            )
            _bt_new = run(
                lambda ts=_ts_bare_list,
                r=registry,
                bl=_bare_bytes: bt_decode_many_with_ss58(ts, r, bl),
                iters,
            )
            _cy_old = run(
                lambda ts=_ts_wrapped_list,
                r=rc,
                bl=_wrapped_bytes: cyscale_batch_decode(ts, r, bl),
                iters,
            )
            _cy_new = run(
                lambda ts=_ts_bare_list, r=rc, bl=_bare_bytes: cyscale_batch_decode(
                    ts, r, bl
                ),
                iters,
            )
            _cy_gain = _cy_old / _cy_new if _cy_new > 0 else float("inf")
            print(
                f"  {(_lbl[:32] + f' ({_item_size}B key)'):<36}  "
                f"{_bt_old:>10.1f}  {_bt_new:>10.1f}  "
                f"{_cy_old:>10.1f}  {_cy_new:>10.1f}  "
                f"{_cy_gain:>7.0f}×"
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", nargs="?", help="Fixture JSON file for benchmarking")
    parser.add_argument(
        "--record", metavar="FILE", help="Record fixtures from live node"
    )
    parser.add_argument(
        "--iters", type=int, default=200, help="Iterations per benchmark (default: 200)"
    )
    args = parser.parse_args()

    if args.record:
        asyncio.run(_record(args.record))
    elif args.fixture:
        bench(args.fixture, args.iters)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
