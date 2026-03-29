import bittensor_wallet
from scalecodec import ss58_encode

from async_substrate_interface.sync_substrate import SubstrateInterface
from async_substrate_interface.types import ScaleObj
from tests.helpers.settings import ARCHIVE_ENTRYPOINT, LATENT_LITE_ENTRYPOINT


def test_legacy_decoding():
    print("Testing test_legacy_decoding")
    # roughly 4000 blocks before metadata v15 was added
    pre_metadata_v15_block = 3_010_611

    with SubstrateInterface(ARCHIVE_ENTRYPOINT) as substrate:
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
            assert isinstance(value, ScaleObj)

        timestamp = substrate.query(
            "Timestamp",
            "Now",
            block_hash=block_hash,
        )
        assert timestamp.value == 1716358476004
    print("test_legacy_decoding succeeded")


def test_ss58_conversion():
    print("Testing test_ss58_conversion")
    with SubstrateInterface(LATENT_LITE_ENTRYPOINT, ss58_format=42) as substrate:
        block_hash = substrate.get_chain_finalised_head()
        qm = substrate.query_map(
            "SubtensorModule",
            "OwnedHotkeys",
            block_hash=block_hash,
        )
        for key, value in qm.records:
            assert isinstance(key, str)
            assert isinstance(value, ScaleObj)
            assert isinstance(value.value, list)
            if len(value.value) > 0:
                for decoded_key in value.value:
                    assert isinstance(decoded_key, str)
    print("test_ss58_conversion succeeded")


def test_get_events_proper_decoding():
    print("Testing test_get_events_proper_decoding")
    # known block/hash pair that has the events we seek to decode
    block = 5846788
    block_hash = "0x0a1c45063a59b934bfee827caa25385e60d5ec1fd8566a58b5cc4affc4eec412"

    with SubstrateInterface(ARCHIVE_ENTRYPOINT) as substrate:
        all_events = substrate.get_events(block_hash=block_hash)
        event = all_events[1]
        assert event["attributes"] == (
            "5G1NjW9YhXLadMWajvTkfcJy6up3yH2q1YzMXDTi6ijanChe",
            30,
            "0xa6b4e5c8241d60ece0c25056b19f7d21ae845269fc771ad46bf3e011865129a5",
        )
    print("test_get_events_proper_decoding succeeded")


def test_query_multiple():
    print("Testing test_query_multiple")
    block = 6153277
    cks = [
        "5FH9AQM4kqbkdC9jyV5FrdEWVYt41nkhFstop7Vhyfb9ZsXt",
        "5GQxLKxjZWNZDsghmYcw7P6ahC7XJCjx1WD94WGh92quSycx",
        "5EcaPiDT1cv951SkCFsvdHDs2yAEUWhJDuRP9mHb343WnaVn",
    ]
    with SubstrateInterface(ARCHIVE_ENTRYPOINT) as substrate:
        block_hash = substrate.get_block_hash(block_id=block)
        assert substrate.query_multiple(
            params=cks,
            module="SubtensorModule",
            storage_function="OwnedHotkeys",
            block_hash=block_hash,
        )
    print("test_query_multiple succeeded")


def test_query_map_with_odd_number_of_params():
    print("Testing test_query_map_with_odd_number_of_params")
    with SubstrateInterface(LATENT_LITE_ENTRYPOINT, ss58_format=42) as substrate:
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
    alice_coldkey = bittensor_wallet.Keypair.create_from_uri("//Alice")
    bob_coldkey = bittensor_wallet.Keypair.create_from_uri("//Bob")
    with SubstrateInterface(
        LATENT_LITE_ENTRYPOINT, ss58_format=42, chain_name="Bittensor"
    ) as substrate:
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
