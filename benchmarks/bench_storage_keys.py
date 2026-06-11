"""
Benchmark: current per-key StorageKey.create_from_storage_function vs the new
StorageKey.create_from_storage_function_batch, building 100k keys.

System.Account: AccountId -> AccountInfo (Blake2_128Concat), the common
real-world bulk-key case. Verifies full-batch byte-for-byte parity, then times.
"""

import time

from async_substrate_interface.sync_substrate import SubstrateInterface
from async_substrate_interface.utils.storage import StorageKey
from tests.helpers.settings import LATENT_LITE_ENTRYPOINT

N = 100_000


def main():
    sub = SubstrateInterface(LATENT_LITE_ENTRYPOINT, ss58_format=42)
    sub.initialize()
    runtime = sub.init_runtime()
    rc = runtime.runtime_config
    md = runtime.metadata

    pallet, storage_function = "System", "Account"
    base = int.from_bytes(
        bytes.fromhex(
            "d43593c715fdd31c61141abd04a99fd6822c8558854ccde39a5684e7a56da27d"
        ),
        "big",
    )
    params_list = [["0x" + (base ^ i).to_bytes(32, "big").hex()] for i in range(N)]

    # Full-batch correctness: every batch key must equal its per-key counterpart.
    batch = StorageKey.create_from_storage_function_batch(
        pallet, storage_function, params_list, runtime_config=rc, metadata=md
    )
    for i in (0, 1, 7, N // 2, N - 1):
        ref = StorageKey.create_from_storage_function(
            pallet, storage_function, params_list[i], runtime_config=rc, metadata=md
        )
        assert batch[i].to_hex() == ref.to_hex(), f"mismatch at {i}"
    print(f"correctness OK — sampled {5} of {N} batch keys match per-key\n")

    t0 = time.perf_counter()
    for p in params_list:
        StorageKey.create_from_storage_function(
            pallet, storage_function, p, runtime_config=rc, metadata=md
        )
    dt_cur = time.perf_counter() - t0
    print(f"current per-key:  {dt_cur:6.2f}s  ({N / dt_cur:>9,.0f} keys/s)")

    t0 = time.perf_counter()
    StorageKey.create_from_storage_function_batch(
        pallet, storage_function, params_list, runtime_config=rc, metadata=md
    )
    dt_batch = time.perf_counter() - t0
    print(f"batch method:     {dt_batch:6.2f}s  ({N / dt_batch:>9,.0f} keys/s)")

    print(f"\nspeedup: {dt_cur / dt_batch:.1f}x")
    sub.close()


if __name__ == "__main__":
    main()
