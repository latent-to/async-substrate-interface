"""
Synthetic benchmark for the `query_map(fully_exhaust=True)` decode path.

Production timing model: all websocket sends go out at t=0; the node then
returns responses serialized — page 1 at ~5s, page 2 at ~10s, page 3 at ~15s,
etc. Each page's decode is short relative to the 5s gap.

  * **old**: wait for every page to arrive, then call `batch_decode` once on
    the concatenation. Wall clock = (N-1) * stagger + first_arrival + D_total.
  * **new**: decode each page the moment it arrives, inside the wait window
    for the next page. Wall clock = (N-1) * stagger + first_arrival + D_last.

So the speedup is `D_total - D_last` — essentially the cumulative decode time
minus the last batch's decode. That holds as long as `stagger > D_per_batch`,
which is overwhelmingly true in production (5000ms vs ~15ms).

We do not actually sleep through the stagger here — that would burn many
minutes for no information. Instead we measure both `D_total` (one big decode)
and the per-batch decodes, and report the predicted wall-clock under a
configurable stagger value.

Usage:
    python bench_decode_overlap.py collect   # one-shot: dump per-batch inputs to DUMP_DIR
    python bench_decode_overlap.py bench     # measure decode CPU; predict wall-clock
"""

import asyncio
import glob
import os
import pickle
import statistics
import sys
import time

from async_substrate_interface.async_substrate import AsyncSubstrateInterface
from async_substrate_interface.utils import decoding
from tests.helpers.settings import LATENT_LITE_ENTRYPOINT


# ----- knobs -----
NODE_URL = LATENT_LITE_ENTRYPOINT
PALLET = "SubtensorModule"
STORAGE_ITEM = "AlphaV2"
PAGE_SIZE = 1_000

DUMP_DIR = "/tmp/qmap_batches"

# Used only to *predict* wall-clock under a hypothetical arrival schedule;
# nothing actually sleeps for this long.
PREDICTED_FIRST_ARRIVAL_SEC = 5.0
PREDICTED_STAGGER_MS = 5_000.0
RUNS = 3


# ---------- collect ----------


async def collect() -> None:
    os.makedirs(DUMP_DIR, exist_ok=True)
    for f in glob.glob(os.path.join(DUMP_DIR, "batch_*.pkl")):
        os.remove(f)

    original = decoding._decode_scale_list_with_runtime
    counter = {"i": 0}

    def capturing(type_strings, scale_bytes_list, runtime):
        i = counter["i"]
        counter["i"] += 1
        with open(os.path.join(DUMP_DIR, f"batch_{i:04d}.pkl"), "wb") as f:
            pickle.dump(
                {
                    "type_strings": list(type_strings),
                    "scale_bytes_list": [bytes(b) for b in scale_bytes_list],
                },
                f,
            )
        return original(type_strings, scale_bytes_list, runtime)

    decoding._decode_scale_list_with_runtime = capturing
    try:
        async with AsyncSubstrateInterface(
            NODE_URL, chain_name="Bittensor"
        ) as substrate:
            block_hash = await substrate.get_block_hash(None)
            qm = await substrate.query_map(
                PALLET,
                STORAGE_ITEM,
                block_hash=block_hash,
                fully_exhaust=True,
                page_size=PAGE_SIZE,
            )
            print(
                f"Collected {counter['i']} batch dumps "
                f"({len(qm.records)} records total) into {DUMP_DIR}"
            )
    finally:
        decoding._decode_scale_list_with_runtime = original


# ---------- bench ----------


def load_batches() -> list[dict]:
    files = sorted(glob.glob(os.path.join(DUMP_DIR, "batch_*.pkl")))
    if not files:
        sys.exit(f"No dumps in {DUMP_DIR}; run `collect` first.")
    batches = []
    for path in files:
        with open(path, "rb") as fh:
            batches.append(pickle.load(fh))
    return batches


async def _open_runtime():
    substrate = AsyncSubstrateInterface(NODE_URL, chain_name="Bittensor")
    await substrate.initialize()
    block_hash = await substrate.get_block_hash(None)
    runtime = await substrate.init_runtime(block_hash=block_hash)
    return substrate, runtime.runtime_config


def time_concatenated_decode(rc, batches) -> float:
    """Time a single `batch_decode` over the concatenation of every batch's
    inputs — what the old code does after all responses arrive."""
    all_types: list[str] = []
    all_bytes: list[bytes] = []
    for b in batches:
        all_types.extend(b["type_strings"])
        all_bytes.extend(b["scale_bytes_list"])
    start = time.perf_counter()
    rc.batch_decode(all_types, all_bytes)
    return time.perf_counter() - start


def time_per_batch_decodes(rc, batches) -> list[float]:
    """Time each batch's `batch_decode` independently — what the new code
    does, one call per response."""
    per: list[float] = []
    for b in batches:
        start = time.perf_counter()
        rc.batch_decode(b["type_strings"], b["scale_bytes_list"])
        per.append(time.perf_counter() - start)
    return per


async def bench() -> None:
    batches = load_batches()
    sizes = [len(b["type_strings"]) for b in batches]
    print(
        f"Loaded {len(batches)} batches; items/batch "
        f"min={min(sizes)} max={max(sizes)} total={sum(sizes)}"
    )

    substrate, rc = await _open_runtime()
    try:
        concat_runs = [time_concatenated_decode(rc, batches) for _ in range(RUNS)]
        per_batch_runs = [time_per_batch_decodes(rc, batches) for _ in range(RUNS)]
    finally:
        await substrate.close()

    d_total = statistics.median(concat_runs)
    per_batch_medians = [statistics.median(times) for times in zip(*per_batch_runs)]
    d_per_batch_sum = sum(per_batch_medians)
    d_last = per_batch_medians[-1]
    d_max = max(per_batch_medians)

    print()
    print("--- decode CPU (medians across runs) ---")
    print(f"  single concatenated decode (old):    {d_total * 1000:.1f}ms")
    print(f"  per-batch decode total (new):        {d_per_batch_sum * 1000:.1f}ms")
    print(f"  per-batch decode max:                {d_max * 1000:.1f}ms")
    print(f"  per-batch decode last (final batch): {d_last * 1000:.1f}ms")

    # Predicted wall clock under the configured arrival schedule. The
    # tail-after-last-arrival is what differs between strategies.
    network_tail = PREDICTED_FIRST_ARRIVAL_SEC + (len(batches) - 1) * (
        PREDICTED_STAGGER_MS / 1000.0
    )
    old_wall = network_tail + d_total
    new_wall = network_tail + d_last  # assumes stagger > d_max; verified below
    print()
    print(
        f"--- predicted wall-clock @ first_arrival="
        f"{PREDICTED_FIRST_ARRIVAL_SEC:.2f}s, stagger={PREDICTED_STAGGER_MS:.0f}ms ---"
    )
    print(f"  old: {old_wall:.3f}s")
    print(f"  new: {new_wall:.3f}s")
    print(f"  speedup: {old_wall - new_wall:.3f}s")
    if PREDICTED_STAGGER_MS / 1000.0 <= d_max:
        print(
            "  WARNING: stagger <= max per-batch decode; decodes back up "
            "and the prediction above understates new's wall clock."
        )


# ---------- entry ----------

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "bench"
    if cmd == "collect":
        asyncio.run(collect())
    elif cmd == "bench":
        asyncio.run(bench())
    else:
        sys.exit("usage: bench_decode_overlap.py [collect|bench]")
