import bittensor_wallet
import pytest

from async_substrate_interface.sync_substrate import (
    SubstrateInterface,
    ExtrinsicReceipt,
)
from tests.helpers.fixtures import FakeWebsocket


@pytest.fixture
def alice_coldkey():
    yield bittensor_wallet.Keypair.create_from_uri("//Alice")


@pytest.fixture
def bob_coldkey():
    yield bittensor_wallet.Keypair.create_from_uri("//Bob")


def get_mock_substrate(seed: str):
    sub = SubstrateInterface(
        "ws://127.0.0.1:9945", ss58_format=42, chain_name="Bittensor", _mock=True
    )
    sub.ws = FakeWebsocket(seed=seed)
    sub.initialize()
    return sub


def test_ss58_conversion():
    print("Testing test_ss58_conversion")
    substrate = get_mock_substrate("test_ss58_conversion")
    block_hash = substrate.get_chain_finalised_head()
    qm = substrate.query_map(
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


def test_get_events_proper_decoding():
    print("Testing test_get_events_proper_decoding")
    substrate = get_mock_substrate("test_get_events_proper_decoding")
    # known block/hash pair that has the events we seek to decode
    block = 7959635  # noqa
    block_hash = "0x81617dc8ede17528d8f8aab64c84285a166f73e120ff6d2acd11e3419a95abec"
    all_events = substrate.get_events(block_hash=block_hash)
    event = all_events[1]
    assert event["attributes"] == (
        53,
        "5CsvRJXuR955WojnGMdok1hbhffZyB4N5ocrv82f3p5A2zVp",
    )
    print("test_get_events_proper_decoding succeeded")


def test_query_map_with_odd_number_of_params():
    print("Testing test_query_map_with_odd_number_of_params")
    substrate = get_mock_substrate("test_query_map_with_odd_number_of_params")
    qm = substrate.query_map(
        "SubtensorModule",
        "Alpha",
        ["5CoZxgtfhcJKX2HmkwnsN18KbaT9aih9eF3b6qVPTgAUbifj"],
    )
    first_record = qm.records[0]
    assert len(first_record) == 2
    assert len(first_record[0]) == 2
    print("test_query_map_with_odd_number_of_params succeeded")


def test_get_payment_info():
    print("Testing test_get_payment_info")
    substrate = get_mock_substrate("test_get_payment_info")
    alice_coldkey = bittensor_wallet.Keypair.create_from_uri("//Alice")
    bob_coldkey = bittensor_wallet.Keypair.create_from_uri("//Bob")
    block_hash = substrate.get_chain_head()
    call = substrate.compose_call(
        "Balances",
        "transfer_keep_alive",
        {"dest": bob_coldkey.ss58_address, "value": 100_000},
        block_hash,
    )
    payment_info = substrate.get_payment_info(
        call=call,
        keypair=alice_coldkey,
    )
    partial_fee_no_era = payment_info["partial_fee"]
    assert partial_fee_no_era > 0
    payment_info_era = substrate.get_payment_info(
        call=call, keypair=alice_coldkey, era={"period": 64}
    )
    partial_fee_era = payment_info_era["partial_fee"]
    assert partial_fee_era > partial_fee_no_era

    payment_info_all_options = substrate.get_payment_info(
        call=call,
        keypair=alice_coldkey,
        era={"period": 64},
        nonce=substrate.get_account_nonce(alice_coldkey.ss58_address),
        tip=5_000_000,
        tip_asset_id=64,
    )
    partial_fee_all_options = payment_info_all_options["partial_fee"]
    assert partial_fee_all_options > partial_fee_no_era
    assert partial_fee_all_options > partial_fee_era
    print("test_get_payment_info succeeded")


def test_bits():
    substrate = get_mock_substrate("test_bits")
    current_sqrt_price = substrate.query(
        module="Swap",
        storage_function="AlphaSqrtPrice",
        params=[71],
    )
    assert isinstance(current_sqrt_price.value, dict)


def test_same_events():
    substrate = get_mock_substrate("test_same_events")
    block_hash = substrate.get_chain_finalised_head()
    block = substrate.get_block_number(block_hash)
    ext_idx = 1
    events = substrate.get_events(block_hash=block_hash)
    ext_receipt = ExtrinsicReceipt.create_from_extrinsic_identifier(
        substrate, f"{block}-{ext_idx}"
    )
    ext_events = ext_receipt.triggered_events
    events_for_ext = [e for e in events if e["extrinsic_idx"] == ext_idx]
    assert ext_events == events_for_ext
