import asyncio
import logging
import os.path
import threading
import socket

import bittensor_wallet
import pytest


from async_substrate_interface.async_substrate import (
    AsyncSubstrateInterface,
    AsyncExtrinsicReceipt,
    logger,
)
from tests.helpers.fixtures import MockWebsocket

from tests.helpers.proxy_server import ProxyServer


@pytest.fixture
def alice_coldkey():
    yield bittensor_wallet.Keypair.create_from_uri("//Alice")


@pytest.fixture
def bob_coldkey():
    yield bittensor_wallet.Keypair.create_from_uri("//Bob")


async def get_mock_substrate(seed: str):
    sub = AsyncSubstrateInterface(
        "ws://127.0.0.1", ss58_format=42, chain_name="Bittensor", _mock=True
    )
    sub.ws = MockWebsocket(seed=seed)
    await sub.initialize()
    return sub


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
