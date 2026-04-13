import bittensor_wallet
import pytest

from async_substrate_interface.sync_substrate import (
    SubstrateInterface,
    ExtrinsicReceipt,
)
from tests.helpers.fixtures import FakeWebsocket
from tests.helpers.settings import ARCHIVE_ENTRYPOINT


@pytest.fixture
def alice_coldkey():
    yield bittensor_wallet.Keypair.create_from_uri("//Alice")


@pytest.fixture
def bob_coldkey():
    yield bittensor_wallet.Keypair.create_from_uri("//Bob")


@pytest.fixture
async def substrate():
    _sub = SubstrateInterface(
        ARCHIVE_ENTRYPOINT, ss58_format=42, chain_name="Bittensor"
    )
    _sub.initialize()
    try:
        yield _sub
    finally:
        _sub.close()


def get_mock_substrate(seed: str):
    sub = SubstrateInterface(
        ARCHIVE_ENTRYPOINT, ss58_format=42, chain_name="Bittensor", _mock=True
    )
    sub.ws = FakeWebsocket(seed=seed)
    sub.initialize()
    return sub


def test_legacy_decoding(substrate):
    print("Testing test_legacy_decoding")
    # roughly 4000 blocks before metadata v15 was added
    pre_metadata_v15_block = 3_010_611
    block_hash = substrate.get_block_hash(pre_metadata_v15_block)
    events = substrate.get_events(block_hash)
    assert isinstance(events, list)

    query_map_result = substrate.query_map(
        module="SubtensorModule",
        storage_function="NetworksAdded",
        block_hash=block_hash,
    )
    for key, value in query_map_result:
        assert isinstance(key, int)
        assert isinstance(value, bool)

    timestamp = substrate.query(
        "Timestamp",
        "Now",
        block_hash=block_hash,
    )
    assert timestamp.value == 1716358476004
    print("test_legacy_decoding succeeded")


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


def test_get_events_proper_decoding(substrate):
    print("Testing test_get_events_proper_decoding")
    # known block/hash pair that has the events we seek to decode
    block = 5846788
    block_hash = "0x0a1c45063a59b934bfee827caa25385e60d5ec1fd8566a58b5cc4affc4eec412"
    all_events = substrate.get_events(block_hash=block_hash)
    event = all_events[1]
    assert event["attributes"] == (
        "5G1NjW9YhXLadMWajvTkfcJy6up3yH2q1YzMXDTi6ijanChe",
        30,
        "0xa6b4e5c8241d60ece0c25056b19f7d21ae845269fc771ad46bf3e011865129a5",
    )
    print("test_get_events_proper_decoding succeeded")


def test_query_map_with_odd_number_of_params(substrate):
    print("Testing test_query_map_with_odd_number_of_params")
    qm = substrate.query_map(
        "SubtensorModule",
        "Alpha",
        ["5CoZxgtfhcJKX2HmkwnsN18KbaT9aih9eF3b6qVPTgAUbifj"],
    )
    first_record = qm.records[0]
    assert len(first_record) == 2
    assert len(first_record[0]) == 2
    print("test_query_map_with_odd_number_of_params succeeded")


def test_get_payment_info(substrate):
    print("Testing test_get_payment_info")
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


@pytest.mark.skip("Bittensor must first be updated for this branch/version")
def test_old_runtime_calls():
    from bittensor import SubtensorApi

    sub = SubtensorApi(
        network=ARCHIVE_ENTRYPOINT, legacy_methods=True, async_subtensor=False
    )
    # will pass
    assert sub.get_stake_info_for_coldkey(
        "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G", 4943592
    )
    # needs to use legacy
    assert sub.get_stake_info_for_coldkey(
        "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G", 4670227
    )


def test_old_runtime_calls_natively(substrate):
    coldkey_ss58 = "5CQ6dMW8JZhKCZX9kWsZRqa3kZRKmNHxbPPVFEt6FgyvGv2G"
    new_block_hash = substrate.get_block_hash(4943592)
    result = substrate.runtime_call(
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
    old_block_hash = substrate.get_block_hash(4670227)
    result = substrate.runtime_call(
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


def test_bits():
    substrate = get_mock_substrate("test_bits")
    current_sqrt_price = substrate.query(
        module="Swap",
        storage_function="AlphaSqrtPrice",
        params=[71],
    )
    assert isinstance(current_sqrt_price.value, dict)


def test_same_events(substrate: SubstrateInterface):
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
