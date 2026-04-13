import asyncio
import logging
import os.path
import time
import threading
import socket
from wsgiref.validate import assert_

import bittensor_wallet
import pytest
import pytest_asyncio
from scalecodec import ss58_encode

from async_substrate_interface.async_substrate import (
    AsyncSubstrateInterface,
    AsyncExtrinsicReceipt,
    logger,
    Websocket,
)
from tests.helpers.fixtures import MockWebsocket
from tests.helpers.settings import ARCHIVE_ENTRYPOINT, LATENT_LITE_ENTRYPOINT
from tests.helpers.proxy_server import ProxyServer


@pytest.fixture
def alice_coldkey():
    yield bittensor_wallet.Keypair.create_from_uri("//Alice")


@pytest.fixture
def bob_coldkey():
    yield bittensor_wallet.Keypair.create_from_uri("//Bob")


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def substrate():
    _sub = AsyncSubstrateInterface(
        ARCHIVE_ENTRYPOINT,
        ss58_format=42,
        chain_name="Bittensor",
        ws_shutdown_timer=None,
    )
    await _sub.initialize()
    try:
        yield _sub
    finally:
        await _sub.close()


async def get_mock_substrate(seed: str):
    sub = AsyncSubstrateInterface(
        "ws://127.0.0.1", ss58_format=42, chain_name="Bittensor", _mock=True
    )
    sub.ws = MockWebsocket(seed=seed)
    await sub.initialize()
    return sub


@pytest.mark.asyncio
async def test_legacy_decoding(substrate):
    print("Testing test_legacy_decoding")
    # roughly 4000 blocks before metadata v15 was added
    pre_metadata_v15_block = 3_010_611

    block_hash = await substrate.get_block_hash(pre_metadata_v15_block)
    events = await substrate.get_events(block_hash)
    assert isinstance(events, list)

    query_map_result = await substrate.query_map(
        module="SubtensorModule",
        storage_function="NetworksAdded",
        block_hash=block_hash,
    )
    async for key, value in query_map_result:
        assert isinstance(key, int)
        assert isinstance(value, bool)

    timestamp = await substrate.query(
        "Timestamp",
        "Now",
        block_hash=block_hash,
    )
    assert timestamp.value == 1716358476004
    print("test_legacy_decoding succeeded")


@pytest.mark.asyncio
async def test_ss58_conversion():
    print("Testing test_ss58_conversion")
    substrate = await get_mock_substrate("test_ss58_conversion")
    block_hash = await substrate.get_chain_finalised_head()

    qm = await substrate.query_map(
        "SubtensorModule",
        "OwnedHotkeys",
        block_hash=block_hash,
    )
    for key, value in qm.records:
        assert isinstance(key, str)
        assert isinstance(value, list)
        if len(value) > 0:
            for decoded_key in value:
                assert isinstance(decoded_key, str)
    print("test_ss58_conversion succeeded")


@pytest.mark.asyncio
async def test_fully_exhaust_query_map(substrate):
    print("Testing test_fully_exhaust_query_map")
    block_hash = await substrate.get_chain_finalised_head()
    non_fully_exhauster_start = time.time()
    non_fully_exhausted_qm = await substrate.query_map(
        "SubtensorModule",
        "CRV3WeightCommits",
        block_hash=block_hash,
    )
    initial_records_count = len(non_fully_exhausted_qm.records)
    assert initial_records_count <= 100  # default page size
    exhausted_records_count = 0
    async for _ in non_fully_exhausted_qm:
        exhausted_records_count += 1
    non_fully_exhausted_time = time.time() - non_fully_exhauster_start

    assert len(non_fully_exhausted_qm.records) >= initial_records_count
    fully_exhausted_start = time.time()
    fully_exhausted_qm = await substrate.query_map(
        "SubtensorModule",
        "CRV3WeightCommits",
        block_hash=block_hash,
        fully_exhaust=True,
    )

    fully_exhausted_time = time.time() - fully_exhausted_start
    initial_records_count_fully_exhaust = len(fully_exhausted_qm.records)
    assert fully_exhausted_time <= non_fully_exhausted_time, (
        f"Fully exhausted took longer than non-fully exhausted with "
        f"{len(non_fully_exhausted_qm.records)} records in non-fully exhausted "
        f"in {non_fully_exhausted_time} seconds, and {initial_records_count_fully_exhaust} in fully exhausted"
        f" in {fully_exhausted_time} seconds. This could be caused by the fact that on this specific block, "
        f"there are fewer records than take up a single page. This difference should still be small."
    )
    fully_exhausted_records_count = 0
    async for _ in fully_exhausted_qm:
        fully_exhausted_records_count += 1
    assert fully_exhausted_records_count == initial_records_count_fully_exhaust
    assert initial_records_count_fully_exhaust == exhausted_records_count
    print("test_fully_exhaust_query_map succeeded")


@pytest.mark.asyncio
async def test_get_events_proper_decoding():
    print("Testing test_get_events_proper_decoding")
    substrate = await get_mock_substrate("test_get_events_proper_decoding")
    # known block/hash pair that has the events we seek to decode
    block = 7959635
    block_hash = "0x81617dc8ede17528d8f8aab64c84285a166f73e120ff6d2acd11e3419a95abec"
    all_events = await substrate.get_events(block_hash=block_hash)
    event = all_events[1]
    assert event["attributes"] == (
        53,
        "5CsvRJXuR955WojnGMdok1hbhffZyB4N5ocrv82f3p5A2zVp",
    )
    print("test_get_events_proper_decoding succeeded")


@pytest.mark.asyncio
async def test_reconnection():
    """
    Does not use the substrate fixture because this needs to reconnect
    """
    print("Testing test_reconnection")
    async with AsyncSubstrateInterface(
        LATENT_LITE_ENTRYPOINT, ss58_format=42, retry_timeout=8.0
    ) as substrate:
        await asyncio.sleep(9)  # sleep for longer than the retry timeout
        bh = await substrate.get_chain_finalised_head()
        assert isinstance(bh, str)
        assert isinstance(await substrate.get_block_number(bh), int)
    print("test_reconnection succeeded")


@pytest.mark.asyncio
async def test_query_map_with_odd_number_of_params():
    print("Testing test_query_map_with_odd_number_of_params")
    substrate = await get_mock_substrate("test_query_map_with_odd_number_of_params")
    qm = await substrate.query_map(
        "SubtensorModule",
        "Alpha",
        ["5CoZxgtfhcJKX2HmkwnsN18KbaT9aih9eF3b6qVPTgAUbifj"],
    )
    first_record = qm.records[0]
    assert len(first_record) == 2
    assert len(first_record[0]) == 2
    print("test_query_map_with_odd_number_of_params succeeded")


@pytest.mark.skip("Weird issue with the GitHub Actions runner")
@pytest.mark.asyncio
async def test_improved_reconnection():
    def get_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))  # Bind loopback only; port 0 = OS picks free port
            s.listen(1)
            port_ = s.getsockname()[1]
        return port_

    print("Testing test_improved_reconnection")
    ws_logger_path = "/tmp/websockets-proxy-test"
    ws_logger = logging.getLogger("websockets.proxy")
    if os.path.exists(ws_logger_path):
        os.remove(ws_logger_path)
    ws_logger.setLevel(logging.INFO)
    ws_logger.addHandler(logging.FileHandler(ws_logger_path))

    asi_logger_path = "/tmp/async-substrate-interface-test"
    if os.path.exists(asi_logger_path):
        os.remove(asi_logger_path)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.FileHandler(asi_logger_path))
    port = get_free_port()
    print(f"Testing using server on port {port}")
    proxy = ProxyServer("wss://archive.sub.latent.to", 10, 20, port=port)

    server_thread = threading.Thread(target=proxy.connect_and_serve, daemon=True)
    server_thread.start()
    await asyncio.sleep(3)  # give the server start up time
    async with AsyncSubstrateInterface(
        f"ws://localhost:{port}",
        ss58_format=42,
        chain_name="Bittensor",
        retry_timeout=10.0,
        ws_shutdown_timer=None,
    ) as substrate:
        blocks_to_check = [
            5215000,
            5215001,
            5215002,
            5215003,
            5215004,
            5215005,
            5215006,
        ]
        tasks = []
        for block in blocks_to_check:
            block_hash = await substrate.get_block_hash(block_id=block)
            tasks.append(
                substrate.query_map(
                    "SubtensorModule", "TotalHotkeyShares", block_hash=block_hash
                )
            )
        records = await asyncio.gather(*tasks)
        assert len(records) == len(blocks_to_check)
        await substrate.close()
        with open(ws_logger_path, "r") as f:
            assert "Pausing" in f.read()
        with open(asi_logger_path, "r") as f:
            assert "Timeout/ConnectionClosed occurred." in f.read()
    shutdown_thread = threading.Thread(target=proxy.close, daemon=True)
    shutdown_thread.start()
    shutdown_thread.join(timeout=5)
    server_thread.join(timeout=5)
    print("test_improved_reconnection succeeded")


@pytest.mark.asyncio
async def test_get_payment_info(alice_coldkey, bob_coldkey):
    print("Testing test_get_payment_info")
    substrate = await get_mock_substrate("test_get_payment_info")
    block_hash = await substrate.get_chain_head()
    call = await substrate.compose_call(
        "Balances",
        "transfer_keep_alive",
        {"dest": bob_coldkey.ss58_address, "value": 100_000},
        block_hash,
    )
    payment_info = await substrate.get_payment_info(
        call=call,
        keypair=alice_coldkey,
    )
    partial_fee_no_era = payment_info["partial_fee"]
    assert partial_fee_no_era > 0
    payment_info_era = await substrate.get_payment_info(
        call=call, keypair=alice_coldkey, era={"period": 64}
    )
    partial_fee_era = payment_info_era["partial_fee"]
    assert partial_fee_era > partial_fee_no_era

    payment_info_all_options = await substrate.get_payment_info(
        call=call,
        keypair=alice_coldkey,
        era={"period": 64},
        nonce=await substrate.get_account_nonce(alice_coldkey.ss58_address),
        tip=5_000_000,
        tip_asset_id=64,
    )
    partial_fee_all_options = payment_info_all_options["partial_fee"]
    assert partial_fee_all_options > partial_fee_no_era
    assert partial_fee_all_options > partial_fee_era
    print("test_get_payment_info succeeded")


@pytest.mark.asyncio
async def test_concurrent_rpc_requests(substrate):
    """
    Test that multiple concurrent RPC requests on a shared connection work correctly.

    This test verifies the fix for the issue where multiple concurrent tasks
    re-initializing the WebSocket connection caused requests to hang.
    """
    print("Testing test_concurrent_rpc_requests")

    async def concurrent_task(substrate_, task_id):
        """Make multiple RPC calls from a single task."""
        for i in range(5):
            result = await substrate_.get_block_number(None)
            assert isinstance(result, int)
            assert result > 0

    # Run 5 concurrent tasks, each making 5 RPC calls (25 total)
    # This tests that the connection is properly shared without re-initialization
    tasks = [concurrent_task(substrate, i) for i in range(5)]
    await asyncio.gather(*tasks)

    print("test_concurrent_rpc_requests succeeded")


@pytest.mark.asyncio
async def test_wait_for_block(substrate):
    async def handler(_):
        return True

    current_block = await substrate.get_block_number(None)
    result = await substrate.wait_for_block(
        current_block + 3, result_handler=handler, task_return=False
    )
    assert result is True


@pytest.mark.skip("Bittensor must first be updated for this branch/version")
@pytest.mark.asyncio
async def test_old_runtime_calls():
    from bittensor import SubtensorApi

    sub = SubtensorApi(
        network=ARCHIVE_ENTRYPOINT, legacy_methods=True, async_subtensor=True
    )
    await sub.initialize()
    # will pass
    l = await sub.get_stake_info_for_coldkey(
        "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G", 4943592
    )
    # needs to use legacy
    assert await sub.get_stake_info_for_coldkey(
        "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G", 4670227
    )
    await sub.close()


@pytest.mark.asyncio
async def test_old_runtime_calls_natively(substrate):
    coldkey_ss58 = "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G"
    new_block_hash = await substrate.get_block_hash(4943592)
    result = await substrate.runtime_call(
        "StakeInfoRuntimeApi",
        "get_stake_info_for_coldkey",
        params=[coldkey_ss58],
        block_hash=new_block_hash,
    )
    assert result == [
        {
            "hotkey": "5CsvRJXuR955WojnGMdok1hbhffZyB4N5ocrv82f3p5A2zVp",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "netuid": 0,
            "stake": 2279326161672,
            "locked": 0,
            "emission": 0,
            "tao_emission": 0,
            "drain": 0,
            "is_registered": True,
        }
    ]
    old_block_hash = await substrate.get_block_hash(4670227)
    result = await substrate.runtime_call(
        "StakeInfoRuntimeApi",
        "get_stake_info_for_coldkey",
        params=[coldkey_ss58],
        block_hash=old_block_hash,
    )
    assert result == [
        {
            "netuid": 0,
            "hotkey": "5HKrFigd2VndU3Kcj6ZvoxZ8MtdX7d9vd6YzHLysPpsib9pQ",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
        {
            "netuid": 0,
            "hotkey": "5HMgj9vrpZp8c1LtJ1kjQE7EU1zwDyfLBrSx5xhBo92KWiVa",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
        {
            "netuid": 0,
            "hotkey": "5DVJf768bu38xiyNucraCif2XW5aSem7jrPkpJEaggWi5ixN",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
        {
            "netuid": 0,
            "hotkey": "5FRdKxXztAUPpBZHSku2scA4FCs9JQWu8RxPrQWTysEXCKvA",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
        {
            "netuid": 0,
            "hotkey": "5E1zzZpB88p63Q24dwmYD1X1VRCrSXcb98J1UQSJ99RSVKLi",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
        {
            "netuid": 0,
            "hotkey": "5Gn9sy6gxP1fg2gXjUGaZQVw7LGcDriBvNB8rThGrEpBfXYG",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
        {
            "netuid": 0,
            "hotkey": "5GViq4eV9ATXQQ1HZhRUvT2iHQZqJXg9B82WtDwKRVNvhc2f",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
        {
            "netuid": 0,
            "hotkey": "5G6UGtGU7KqycPMRBneUBXNbwa8M7T5Cp2BgmKGDewbrmWfA",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
        {
            "netuid": 0,
            "hotkey": "5D7F11Gq7BxdWpet5KTCDRYCADkbhhusAuYfMCPqrVUTd8vC",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
        {
            "netuid": 0,
            "hotkey": "5Dyj37kQRf7JrbaPZUYoCqCchsSZN9gZyVEw9kxeXBwpNbyx",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
        {
            "netuid": 0,
            "hotkey": "5FgsyCuszBNnR6CPHxX9bQLp3YsgaLVCKRoeRBZ7focMj2tn",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
        {
            "netuid": 0,
            "hotkey": "5FsDYzusqjMpW9bo6nxqKTo8NrwTMoS2epMbPyMWkLXqApfn",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
        {
            "netuid": 0,
            "hotkey": "5GUA8NXh3Cu8cq1w9ByzvEqJTyLgKwefYfrdAfG4DZq5Mt6Z",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
        {
            "netuid": 0,
            "hotkey": "5GHfBFLK7ZQwnsUzX7EDHokrKMdmKpRRRWd5SK7Pcgp8MmMd",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
        {
            "netuid": 0,
            "hotkey": "5CY1yz8QjxiNK4jzjvW8ueaYmdNnZzdmvJ4RVxCqxkKt6cZ2",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
        {
            "netuid": 0,
            "hotkey": "5D59sgAByRiyW8CKpriu4CH9GZ33bvxRk67Qqj4fcVZkQUry",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
        {
            "netuid": 0,
            "hotkey": "5HYgA4ZFHXdSH7j3mK9zRucPaKsZMj2CaxHD4oMPpNfs4Gn5",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
        {
            "netuid": 0,
            "hotkey": "5EABTzX2bNeYzH32XctABDcJfiydTkVZtF8JHMjLgb7GmChe",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
        {
            "netuid": 0,
            "hotkey": "5G4fPyr2NWbdhqTGiDvVoMz2xJX5hCVGoFJwKH9BWjLuspJ5",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
        {
            "netuid": 0,
            "hotkey": "5D22sqZw2YWSpRgbP2yQXtR27zdbk7mGKsMytC9f4g4GL4hb",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
        {
            "netuid": 0,
            "hotkey": "5DtPqa4WTT1bUeqXD5MnT2xmYGi6A1SZZeM8xsMaNVqsXPfd",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
        {
            "netuid": 0,
            "hotkey": "5ELa9FUP9sWNPdmSLeUoGgBhK6groESCXyNvrCZ6jKr3zjwa",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
        {
            "netuid": 0,
            "hotkey": "5CsvRJXuR955WojnGMdok1hbhffZyB4N5ocrv82f3p5A2zVp",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 2232575320215,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
        {
            "netuid": 0,
            "hotkey": "5Dz3txcvqpn64dmw19rMN2sANbzaqexQkcAisiZ1c5AfB2AZ",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
        {
            "netuid": 0,
            "hotkey": "5HKtDbnsccKtbuHiH5ijUEgnP7jTZA7ySKHuhsNBaqwwScuu",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
        {
            "netuid": 0,
            "hotkey": "5F93j61syCq6jqoorwP6Le7wYPfaFCNdWqJsv91yTYYQn6p5",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
        {
            "netuid": 0,
            "hotkey": "5GgtHAd3tzR6bNaJEDj4ufpUh7XStjLuewNmcHs2YosJhGPi",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
        {
            "netuid": 0,
            "hotkey": "5EqC5bzgBCafQncF38KZgcq35Wc6xW6R8FNu4hu3mpUV7T6D",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
        {
            "netuid": 0,
            "hotkey": "5DhgRa7c3H8fTpG3xkegbdcVsQVkVtwSJ3VknFHtUSm4muQA",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
        {
            "netuid": 0,
            "hotkey": "5Gdda7TSfvNgr6iQFU2w8aefRjdLrJsiUFYevpt8C6Dj6AhC",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
        {
            "netuid": 0,
            "hotkey": "5Dqc7MLrMso6AUCiLNpCJ8KVzj5brGDxoTEwWmsbtU5oFjMA",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
        {
            "netuid": 0,
            "hotkey": "5GRbbNm8DV3H1TgJX6gFaukMUD37pky9zR1Y6R7JSVi5ycQ6",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
        {
            "netuid": 0,
            "hotkey": "5DWoojAZSmXRyxumvY8yBbeNGY5dG2wiMUc75LbajvuYUNTj",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
        {
            "netuid": 0,
            "hotkey": "5EZdfVVLT3ortaZ1U819MfneTuTSK1786HgCRxJhRePhFqaB",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
        {
            "netuid": 0,
            "hotkey": "5FhgCHSyWRUeBzUiUiq849VJScYcktGE5pHEuE6ebqbjjrBV",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
        {
            "netuid": 0,
            "hotkey": "5EhnVSgU9UM5L6uuKnQ7NfBcXhkfAocyfjUNudVHY9WZJYfP",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
        {
            "netuid": 0,
            "hotkey": "5HKPNraukPDPWjFCAtkeChLTxQnJy9J62C7UGsoMRyZD6VJy",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
        {
            "netuid": 0,
            "hotkey": "5DvZhfeZHfMWHFTXGJHqLkxs9ZNFMY3k8vwHP4g6QyG7fF6k",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
        {
            "netuid": 0,
            "hotkey": "5G6TGnBtvxjtkUxxCxs9UZWRSJtLqpZDLHDuwmP7r85Etg1Y",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
        {
            "netuid": 0,
            "hotkey": "5Do7NqdstDfVGee9x7A8To6pon8ud2iZkHFy514LvoaVe6dR",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
        {
            "netuid": 0,
            "hotkey": "5H72qbYL3BrYADEvgm87yrUYC3U625SoCc79E4N7q4hH8cdM",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
        {
            "netuid": 0,
            "hotkey": "5GgofX3kxZ4m2Z2RtvwPtjfYqGQzhKrUU8hwso1fpGaE2VQn",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
        {
            "netuid": 0,
            "hotkey": "5Hn9Arr2QKYrBCxWNzFNo81Qn7QUQrhLZRapkZ816vAXmwGa",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
        {
            "netuid": 0,
            "hotkey": "5GU9Qqs2Dx6fK5DRnLEYnti1qBM1LFLhBXofR7kE9Wsmb2M2",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
        {
            "netuid": 0,
            "hotkey": "5Df7tTg3xRNuH859f46QHCkdadXHUXpQGVNQJ8hnM3ECw9CY",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
        {
            "netuid": 0,
            "hotkey": "5FqUNcaCXzbMWGXeWm8XarjBUWXfsTGuwNKtSiwfj5CvMqif",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
        {
            "netuid": 0,
            "hotkey": "5GHVSVMvHGQ65xPp5JfG7VpooUHCveUAfzqEqQYuHwmCBbZs",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
        {
            "netuid": 0,
            "hotkey": "5FR27aCDu4ozHEDfFcAUUJuboKXF1bCSpE6vPCQeaCnLt6iZ",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
        {
            "netuid": 0,
            "hotkey": "5H6h4CkNnNMUSGdwCbFMzgAxixarYDoe3dgywAFWjt91J5Rt",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
        {
            "netuid": 0,
            "hotkey": "5DVDV42XvfHp1BddJ5HpzQVVsfV7219AS5RwpDBTQGSjguKh",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
        {
            "netuid": 0,
            "hotkey": "5G6nYpPv11BJzwCLe3Xoj27bfWZZF2hgxEd97CzKnJEoBphs",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
        {
            "netuid": 0,
            "hotkey": "5CMVoFgq8okW6x4kscgvPSa62R6MqJikbLQvHf8QRYcXpLn5",
            "coldkey": "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G",
            "stake": 0,
            "locked": 0,
            "emission": 0,
            "drain": 0,
            "is_registered": False,
        },
    ]


async def test_bits():
    substrate = await get_mock_substrate("test_bits")
    current_sqrt_price = await substrate.query(
        module="Swap",
        storage_function="AlphaSqrtPrice",
        params=[71],
    )
    assert isinstance(current_sqrt_price.value, dict)


async def test_same_events():
    substrate = await get_mock_substrate("test_same_events")
    block_hash = await substrate.get_chain_finalised_head()
    block = await substrate.get_block_number(block_hash)
    ext_idx = 1
    events = await substrate.get_events(block_hash=block_hash)
    ext_receipt = await AsyncExtrinsicReceipt.create_from_extrinsic_identifier(
        substrate, f"{block}-{ext_idx}"
    )
    ext_events = await ext_receipt.triggered_events
    events_for_ext = [e for e in events if e["extrinsic_idx"] == ext_idx]
    assert ext_events == events_for_ext
