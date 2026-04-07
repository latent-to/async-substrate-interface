import asyncio
import tracemalloc
from unittest.mock import AsyncMock, MagicMock, ANY

import pytest
from websockets.exceptions import InvalidURI
from websockets.protocol import State

from async_substrate_interface.async_substrate import (
    AsyncExtrinsicReceipt,
    AsyncQueryMapResult,
    AsyncSubstrateInterface,
    get_async_substrate_interface,
)
from async_substrate_interface.errors import SubstrateRequestException
from async_substrate_interface.types import ScaleObj
from tests.helpers.settings import ARCHIVE_ENTRYPOINT, LATENT_LITE_ENTRYPOINT


@pytest.mark.asyncio
async def test_invalid_url_raises_exception():
    """Test that invalid URI raises an InvalidURI exception."""
    print("Testing test_invalid_url_raises_exception")
    async_substrate = AsyncSubstrateInterface("non_existent_entry_point")
    with pytest.raises(InvalidURI):
        await async_substrate.initialize()

    with pytest.raises(InvalidURI):
        async with AsyncSubstrateInterface("non_existent_entry_point") as _:
            pass
    print("test_invalid_url_raises_exception succeeded")


@pytest.mark.asyncio
async def test_runtime_call(monkeypatch):
    print("Testing test_runtime_call")
    substrate = AsyncSubstrateInterface("ws://localhost", _mock=True)

    fake_runtime = MagicMock()
    fake_metadata_v15 = MagicMock()
    fake_metadata_v15.get_metadata.return_value.value_object[1].value = {
        "apis": [
            {
                "name": "SubstrateApi",
                "methods": [
                    {
                        "name": "SubstrateMethod",
                        "inputs": [],
                        "output": "1",
                    },
                ],
            },
        ],
        "types": {
            "types": [
                {
                    "id": "1",
                    "type": {
                        "path": ["Vec"],
                        "def": {"sequence": {"type": "4"}},
                    },
                },
            ]
        },
    }
    fake_runtime.metadata_v15 = fake_metadata_v15
    substrate.init_runtime = AsyncMock(return_value=fake_runtime)

    # Patch encode_scale (should not be called in this test since no inputs)
    substrate.encode_scale = AsyncMock()

    # Patch decode_scale to produce a dummy value
    substrate.decode_scale = AsyncMock(return_value="decoded_result")

    # Patch RPC request with correct behavior
    substrate.rpc_request = AsyncMock(
        side_effect=lambda method, params, runtime: {
            "result": "0x00" if method == "state_call" else {"parentHash": "0xDEADBEEF"}
        }
    )

    # Patch get_block_runtime_info
    substrate.get_block_runtime_info = AsyncMock(return_value={"specVersion": "1"})

    # Run the call
    result = await substrate.runtime_call(
        "SubstrateApi",
        "SubstrateMethod",
    )

    # Validate the result is wrapped in ScaleObj
    assert isinstance(result, ScaleObj)
    assert result.value == "decoded_result"

    # Check decode_scale called correctly
    substrate.decode_scale.assert_called_once_with(
        "scale_info::1", b"\x00", runtime=ANY
    )

    # encode_scale should not be called since no inputs
    substrate.encode_scale.assert_not_called()

    # Check RPC request called for the state_call
    substrate.rpc_request.assert_any_call(
        "state_call", ["SubstrateApi_SubstrateMethod", "", None], runtime=ANY
    )
    print("test_runtime_call succeeded")


@pytest.mark.asyncio
async def test_websocket_shutdown_timer():
    print("Testing test_websocket_shutdown_timer")
    # using default ws shutdown timer of 5.0 seconds
    async with AsyncSubstrateInterface("wss://lite.sub.latent.to:443") as substrate:
        await substrate.get_chain_head()
        await asyncio.sleep(6)
    assert (
        substrate.ws.state is State.CLOSED
    )  # connection should have closed automatically

    # using custom ws shutdown timer of 10.0 seconds
    async with AsyncSubstrateInterface(
        "wss://lite.sub.latent.to:443", ws_shutdown_timer=10.0
    ) as substrate:
        await substrate.get_chain_head()
        await asyncio.sleep(6)  # same sleep time as before
        assert substrate.ws.state is State.OPEN  # connection should still be open
    print("test_websocket_shutdown_timer succeeded")


@pytest.mark.asyncio
async def test_runtime_switching():
    print("Testing test_runtime_switching")
    block = 6067945  # block where a runtime switch happens
    async with AsyncSubstrateInterface(
        ARCHIVE_ENTRYPOINT, ss58_format=42, chain_name="Bittensor"
    ) as substrate:
        # assures we switch between the runtimes without error
        assert await substrate.get_extrinsics(block_number=block - 20) is not None
        assert await substrate.get_extrinsics(block_number=block) is not None
        assert await substrate.get_extrinsics(block_number=block - 21) is not None
        one, two = await asyncio.gather(
            substrate.get_extrinsics(block_number=block - 22),
            substrate.get_extrinsics(block_number=block + 1),
        )
        assert one is not None
        assert two is not None
    print("test_runtime_switching succeeded")


@pytest.mark.asyncio
async def test_memory_leak():
    import gc

    # Stop any existing tracemalloc and start fresh
    tracemalloc.stop()
    tracemalloc.start()
    two_mb = 2 * 1024 * 1024

    # Warmup: populate caches before taking baseline
    for _ in range(2):
        subtensor = await get_async_substrate_interface(LATENT_LITE_ENTRYPOINT)
        await subtensor.close()

    baseline_snapshot = tracemalloc.take_snapshot()

    for i in range(5):
        subtensor = await get_async_substrate_interface(LATENT_LITE_ENTRYPOINT)
        await subtensor.close()
        gc.collect()

        snapshot = tracemalloc.take_snapshot()
        stats = snapshot.compare_to(baseline_snapshot, "lineno")
        total_diff = sum(stat.size_diff for stat in stats)
        current, peak = tracemalloc.get_traced_memory()
        # Allow cumulative growth up to 2MB per iteration from baseline
        assert total_diff < two_mb * (i + 1), (
            f"Loop {i}: diff={total_diff / 1024:.2f} KiB, current={current / 1024:.2f} KiB, "
            f"peak={peak / 1024:.2f} KiB"
        )


@pytest.mark.asyncio
async def test_async_query_map_result_retrieve_all_records():
    """Test that retrieve_all_records fetches all pages and returns the full record list."""
    page1 = [("key1", "val1"), ("key2", "val2")]
    page2 = [("key3", "val3"), ("key4", "val4")]
    page3 = [("key5", "val5")]  # partial page signals loading_complete

    mock_substrate = MagicMock()

    qm = AsyncQueryMapResult(
        records=list(page1),
        page_size=2,
        substrate=mock_substrate,
        module="TestModule",
        storage_function="TestStorage",
        last_key="key2",
    )

    # Build mock pages: first call returns page2 (full page), second returns page3 (partial)
    page2_result = AsyncQueryMapResult(
        records=list(page2),
        page_size=2,
        substrate=mock_substrate,
        last_key="key4",
    )
    page3_result = AsyncQueryMapResult(
        records=list(page3),
        page_size=2,
        substrate=mock_substrate,
        last_key="key5",
    )
    mock_substrate.query_map = AsyncMock(side_effect=[page2_result, page3_result])

    result = await qm.retrieve_all_records()

    assert result == page1 + page2 + page3
    assert qm.records == page1 + page2 + page3
    assert qm.loading_complete is True
    assert mock_substrate.query_map.call_count == 2


class TestGetBlockHash:
    @pytest.fixture
    def substrate(self):
        s = AsyncSubstrateInterface("ws://localhost", _mock=True)
        s.runtime_cache = MagicMock()
        s._cached_get_block_hash = AsyncMock(return_value="0xCACHED")
        s.get_chain_head = AsyncMock(return_value="0xHEAD")
        return s

    @pytest.mark.asyncio
    async def test_none_block_id_returns_chain_head(self, substrate):
        result = await substrate.get_block_hash(None)
        assert result == "0xHEAD"
        substrate.get_chain_head.assert_awaited_once()
        substrate._cached_get_block_hash.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_hash(self, substrate):
        substrate.runtime_cache.blocks.get.return_value = "0xFROMCACHE"
        result = await substrate.get_block_hash(42)
        assert result == "0xFROMCACHE"
        substrate.runtime_cache.blocks.get.assert_called_once_with(42)
        substrate._cached_get_block_hash.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cache_miss_fetches_and_stores(self, substrate):
        substrate.runtime_cache.blocks.get.return_value = None
        result = await substrate.get_block_hash(42)
        assert result == "0xCACHED"
        substrate._cached_get_block_hash.assert_awaited_once_with(42)
        substrate.runtime_cache.add_item.assert_called_once_with(
            block_hash="0xCACHED", block=42
        )


class TestGetBlockNumber:
    @pytest.fixture
    def substrate(self):
        s = AsyncSubstrateInterface("ws://localhost", _mock=True)
        s.runtime_cache = MagicMock()
        s._cached_get_block_number = AsyncMock(return_value=100)
        s._get_block_number = AsyncMock(return_value=99)
        return s

    @pytest.mark.asyncio
    async def test_none_block_hash_calls_get_block_number_directly(self, substrate):
        result = await substrate.get_block_number(None)
        assert result == 99
        substrate._get_block_number.assert_awaited_once_with(None)
        substrate._cached_get_block_number.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_number(self, substrate):
        substrate.runtime_cache.blocks_reverse.get.return_value = 42
        result = await substrate.get_block_number("0xABC")
        assert result == 42
        substrate.runtime_cache.blocks_reverse.get.assert_called_once_with("0xABC")
        substrate._cached_get_block_number.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cache_miss_fetches_and_stores(self, substrate):
        substrate.runtime_cache.blocks_reverse.get.return_value = None
        result = await substrate.get_block_number("0xABC")
        assert result == 100
        substrate._cached_get_block_number.assert_awaited_once_with("0xABC")
        substrate.runtime_cache.add_item.assert_called_once_with(
            block_hash="0xABC", block=100
        )


@pytest.mark.asyncio
async def test_get_account_next_index_cached_mode_uses_internal_cache():
    substrate = AsyncSubstrateInterface("ws://localhost", _mock=True)
    substrate.supports_rpc_method = AsyncMock(return_value=True)
    substrate.rpc_request = AsyncMock(return_value={"result": 5})

    first = await substrate.get_account_next_index(
        "5F3sa2TJAWMqDhXG6jhV4N8ko9NoFz5Y2s8vS8uM9f7v7mA"
    )
    second = await substrate.get_account_next_index(
        "5F3sa2TJAWMqDhXG6jhV4N8ko9NoFz5Y2s8vS8uM9f7v7mA"
    )

    assert first == 5
    assert second == 6
    substrate.rpc_request.assert_awaited_once_with(
        "account_nextIndex", ["5F3sa2TJAWMqDhXG6jhV4N8ko9NoFz5Y2s8vS8uM9f7v7mA"]
    )


@pytest.mark.asyncio
async def test_get_account_next_index_bypass_mode_does_not_create_or_mutate_cache():
    substrate = AsyncSubstrateInterface("ws://localhost", _mock=True)
    substrate.supports_rpc_method = AsyncMock(return_value=True)
    substrate.rpc_request = AsyncMock(return_value={"result": 10})

    address = "5F3sa2TJAWMqDhXG6jhV4N8ko9NoFz5Y2s8vS8uM9f7v7mA"
    assert address not in substrate._nonces

    result = await substrate.get_account_next_index(
        address,
        use_cache=False,
    )

    assert result == 10
    assert address not in substrate._nonces
    substrate.rpc_request.assert_awaited_once_with("account_nextIndex", [address])


@pytest.mark.asyncio
async def test_get_account_next_index_bypass_mode_raises_on_rpc_error():
    substrate = AsyncSubstrateInterface("ws://localhost", _mock=True)
    substrate.supports_rpc_method = AsyncMock(return_value=True)
    substrate.rpc_request = AsyncMock(
        return_value={"error": {"message": "rpc failure"}}
    )

    with pytest.raises(SubstrateRequestException, match="rpc failure"):
        await substrate.get_account_next_index(
            "5F3sa2TJAWMqDhXG6jhV4N8ko9NoFz5Y2s8vS8uM9f7v7mA",
            use_cache=False,
        )


class TestAsyncExtrinsicReceiptProcessEvents:
    def _make_event(self, module_id, event_id, attributes, extrinsic_idx=0):
        return {
            "extrinsic_idx": extrinsic_idx,
            "event": {
                "module_id": module_id,
                "event_id": event_id,
                "attributes": attributes,
            },
        }

    def _make_module_error(self, name="ModuleError", docs=None):
        module_error = MagicMock()
        module_error.name = name
        module_error.docs = docs if docs is not None else ["module error docs"]
        return module_error

    def _make_receipt(self, events):
        substrate = MagicMock()
        runtime = MagicMock()
        runtime.metadata = MagicMock()
        substrate.get_events = AsyncMock(return_value=events)
        substrate.init_runtime = AsyncMock(return_value=runtime)
        receipt = AsyncExtrinsicReceipt(
            substrate=substrate,
            extrinsic_hash="0xdeadbeef",
            block_hash="0xabc",
            extrinsic_idx=0,
        )
        return receipt, substrate, runtime

    @pytest.mark.asyncio
    async def test_extracts_dispatch_info_weight(self):
        events = [
            self._make_event(
                "System",
                "ExtrinsicSuccess",
                {"dispatch_info": {"weight": {"ref_time": 1, "proof_size": 2}}},
            )
        ]
        receipt, _, _ = self._make_receipt(events)

        assert await receipt.is_success is True
        assert await receipt.error_message is None
        assert await receipt.weight == {"ref_time": 1, "proof_size": 2}

    @pytest.mark.asyncio
    async def test_extracts_legacy_weight(self):
        events = [self._make_event("System", "ExtrinsicSuccess", {"weight": 7})]
        receipt, _, _ = self._make_receipt(events)

        assert await receipt.is_success is True
        assert await receipt.error_message is None
        assert await receipt.weight == 7

    @pytest.mark.asyncio
    async def test_prefers_transaction_fee_paid_over_deposit_fallback(self):
        events = [
            self._make_event(
                "TransactionPayment",
                "TransactionFeePaid",
                {"actual_fee": 10},
            ),
            self._make_event("Treasury", "Deposit", {"value": 99}),
            self._make_event("Balances", "Deposit", {"amount": 88}),
        ]
        receipt, _, _ = self._make_receipt(events)

        assert await receipt.total_fee_amount == 10

    @pytest.mark.asyncio
    async def test_accumulates_fallback_fee_from_deposits(self):
        events = [
            self._make_event("Treasury", "Deposit", {"value": 3}),
            self._make_event("Balances", "Deposit", {"amount": 2}),
        ]
        receipt, _, _ = self._make_receipt(events)

        assert await receipt.total_fee_amount == 5

    @pytest.mark.asyncio
    async def test_decodes_legacy_module_error_tuple(self):
        events = [
            self._make_event(
                "System",
                "ExtrinsicFailed",
                {
                    "dispatch_info": {"weight": 9},
                    "dispatch_error": {"Module": (3, 4)},
                },
            )
        ]
        receipt, substrate, runtime = self._make_receipt(events)
        runtime.metadata.get_module_error.return_value = self._make_module_error(
            name="InsufficientBalance",
            docs=["balance too low"],
        )

        assert await receipt.is_success is False
        assert await receipt.error_message == {
            "type": "Module",
            "name": "InsufficientBalance",
            "docs": ["balance too low"],
        }
        assert await receipt.weight == 9
        substrate.init_runtime.assert_awaited_once_with(block_hash="0xabc")
        runtime.metadata.get_module_error.assert_called_once_with(
            module_index=3, error_index=4
        )

    @pytest.mark.asyncio
    async def test_decodes_module_error_from_hex_error_bytes(self):
        events = [
            self._make_event(
                "System",
                "ExtrinsicFailed",
                {
                    "dispatch_info": {"weight": 9},
                    "dispatch_error": {"Module": {"index": 5, "error": "0x0a000000"}},
                },
            )
        ]
        receipt, substrate, runtime = self._make_receipt(events)
        runtime.metadata.get_module_error.return_value = self._make_module_error(
            name="DecodedHexError",
            docs=["decoded from first byte"],
        )

        assert await receipt.is_success is False
        assert await receipt.error_message == {
            "type": "Module",
            "name": "DecodedHexError",
            "docs": ["decoded from first byte"],
        }
        assert await receipt.weight == 9
        substrate.init_runtime.assert_awaited_once_with(block_hash="0xabc")
        runtime.metadata.get_module_error.assert_called_once_with(
            module_index=5, error_index=10
        )

    @pytest.mark.asyncio
    async def test_maps_bad_origin_error(self):
        events = [
            self._make_event(
                "System",
                "ExtrinsicFailed",
                {
                    "dispatch_info": {"weight": 9},
                    "dispatch_error": {"BadOrigin": None},
                },
            )
        ]
        receipt, substrate, _ = self._make_receipt(events)

        assert await receipt.is_success is False
        assert await receipt.error_message == {
            "type": "System",
            "name": "BadOrigin",
            "docs": "Bad origin",
        }
        assert await receipt.weight == 9
        substrate.init_runtime.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_maps_cannot_lookup_error(self):
        events = [
            self._make_event(
                "System",
                "ExtrinsicFailed",
                {
                    "dispatch_info": {"weight": 9},
                    "dispatch_error": {"CannotLookup": None},
                },
            )
        ]
        receipt, substrate, _ = self._make_receipt(events)

        assert await receipt.is_success is False
        assert await receipt.error_message == {
            "type": "System",
            "name": "CannotLookup",
            "docs": "Cannot lookup",
        }
        assert await receipt.weight == 9
        substrate.init_runtime.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_maps_token_error(self):
        events = [
            self._make_event(
                "System",
                "ExtrinsicFailed",
                {
                    "dispatch_info": {"weight": 9},
                    "dispatch_error": {"Token": "FundsUnavailable"},
                },
            )
        ]
        receipt, substrate, _ = self._make_receipt(events)

        assert await receipt.is_success is False
        assert await receipt.error_message == {
            "type": "System",
            "name": "Token",
            "docs": "FundsUnavailable",
        }
        assert await receipt.weight == 9
        substrate.init_runtime.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_preserves_unknown_dispatch_error_as_none(self):
        events = [
            self._make_event(
                "System",
                "ExtrinsicFailed",
                {
                    "dispatch_info": {"weight": 9},
                    "dispatch_error": {"Arithmetic": "Overflow"},
                },
            )
        ]
        receipt, substrate, _ = self._make_receipt(events)

        assert await receipt.is_success is False
        assert await receipt.error_message is None
        assert await receipt.weight == 9
        substrate.init_runtime.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_failure_takes_precedence_over_success(self):
        events = [
            self._make_event(
                "System",
                "ExtrinsicSuccess",
                {"dispatch_info": {"weight": 1}},
            ),
            self._make_event(
                "System",
                "ExtrinsicFailed",
                {
                    "dispatch_info": {"weight": 9},
                    "dispatch_error": {"Other": None},
                },
            ),
        ]
        receipt, substrate, _ = self._make_receipt(events)

        assert await receipt.is_success is False
        assert await receipt.error_message == {
            "type": "System",
            "name": "Other",
            "docs": "Unspecified error occurred",
        }
        assert await receipt.weight == 9
        substrate.init_runtime.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_maps_mevshield_decrypted_rejected_error(self):
        events = [
            self._make_event(
                "MevShield",
                "DecryptedRejected",
                {
                    "reason": {
                        "post_info": {"actual_weight": 123},
                        "error": {"Other": None},
                    }
                },
            )
        ]
        receipt, substrate, _ = self._make_receipt(events)

        assert await receipt.is_success is False
        assert await receipt.error_message == {
            "type": "System",
            "name": "Other",
            "docs": "Unspecified error occurred",
        }
        assert await receipt.weight == 123
        assert await receipt.total_fee_amount == 0
        substrate.init_runtime.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_maps_mevshield_decryption_failed_error(self):
        events = [
            self._make_event(
                "MevShield",
                "DecryptionFailed",
                {"reason": "ciphertext could not be decrypted"},
            )
        ]
        receipt, substrate, _ = self._make_receipt(events)

        assert await receipt.is_success is False
        assert await receipt.error_message == {
            "type": "MevShield",
            "name": "DecryptionFailed",
            "docs": "ciphertext could not be decrypted",
        }
        assert await receipt.total_fee_amount == 0
        assert await receipt.weight is None
        substrate.init_runtime.assert_not_awaited()
