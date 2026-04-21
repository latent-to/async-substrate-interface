from unittest.mock import MagicMock

from async_substrate_interface.sync_substrate import (
    SubstrateInterface,
    QueryMapResult,
    ExtrinsicReceipt,
)


def test_runtime_call(monkeypatch):
    print("Testing test_runtime_call")
    substrate = SubstrateInterface("ws://localhost", _mock=True)
    fake_runtime = MagicMock()
    fake_runtime.metadata_v15 = MagicMock()  # non-None so the V15 path is taken
    fake_runtime.runtime_api_map = {
        "SubstrateApi": {
            "SubstrateMethod": {"inputs": [], "output": "1"},
        }
    }
    fake_runtime.type_id_to_name = {}  # "1" not in map → not Vec<u8> → new path
    substrate.init_runtime = MagicMock(return_value=fake_runtime)

    # Patch encode_scale (should not be called in this test since no inputs)
    substrate.encode_scale = MagicMock()

    # Patch decode_scale to produce a dummy value
    mock_scale_obj = MagicMock()
    mock_scale_obj.value = "decoded_result"
    substrate.decode_scale = MagicMock(return_value=mock_scale_obj)

    # Patch RPC request with correct behavior
    substrate.rpc_request = MagicMock(
        side_effect=lambda method, params: {
            "result": "0x00" if method == "state_call" else {"parentHash": "0xDEADBEEF"}
        }
    )

    # Patch get_block_runtime_info
    substrate.get_block_runtime_info = MagicMock(return_value={"specVersion": "1"})

    # Run the call
    result = substrate.runtime_call(
        "SubstrateApi",
        "SubstrateMethod",
    )

    assert result == "decoded_result"

    # Check decode_scale called correctly
    substrate.decode_scale.assert_called_once_with("scale_info::1", b"\x00")

    # encode_scale should not be called since no inputs
    substrate.encode_scale.assert_not_called()

    # Check RPC request called for the state_call
    substrate.rpc_request.assert_any_call(
        "state_call", ["SubstrateApi_SubstrateMethod", "", None]
    )
    substrate.close()
    print("test_runtime_call succeeded")


def test_async_query_map_result_retrieve_all_records():
    """Test that retrieve_all_records fetches all pages and returns the full record list."""
    page1 = [("key1", "val1"), ("key2", "val2")]
    page2 = [("key3", "val3"), ("key4", "val4")]
    page3 = [("key5", "val5")]  # partial page signals loading_complete

    mock_substrate = MagicMock()

    qm = QueryMapResult(
        records=list(page1),
        page_size=2,
        substrate=mock_substrate,
        module="TestModule",
        storage_function="TestStorage",
        last_key="key2",
    )

    # Build mock pages: first call returns page2 (full page), second returns page3 (partial)
    page2_result = QueryMapResult(
        records=list(page2),
        page_size=2,
        substrate=mock_substrate,
        last_key="key4",
    )
    page3_result = QueryMapResult(
        records=list(page3),
        page_size=2,
        substrate=mock_substrate,
        last_key="key5",
    )
    mock_substrate.query_map = MagicMock(side_effect=[page2_result, page3_result])

    result = qm.retrieve_all_records()

    assert result == page1 + page2 + page3
    assert qm.records == page1 + page2 + page3
    assert qm.loading_complete is True
    assert mock_substrate.query_map.call_count == 2


class TestGetBlockHash:
    def _make_substrate(self):
        s = SubstrateInterface("ws://localhost", _mock=True)
        s.runtime_cache = MagicMock()
        s._get_block_hash = MagicMock(return_value="0xCACHED")
        s.get_chain_head = MagicMock(return_value="0xHEAD")
        return s

    def test_none_block_id_returns_chain_head(self):
        substrate = self._make_substrate()
        result = substrate.get_block_hash(None)
        assert result == "0xHEAD"
        substrate.get_chain_head.assert_called_once()
        substrate._get_block_hash.assert_not_called()

    def test_cache_hit_returns_cached_hash(self):
        substrate = self._make_substrate()
        substrate.runtime_cache.blocks.get.return_value = "0xFROMCACHE"
        result = substrate.get_block_hash(42)
        assert result == "0xFROMCACHE"
        substrate.runtime_cache.blocks.get.assert_called_once_with(42)
        substrate._get_block_hash.assert_not_called()

    def test_cache_miss_fetches_and_stores(self):
        substrate = self._make_substrate()
        substrate.runtime_cache.blocks.get.return_value = None
        result = substrate.get_block_hash(42)
        assert result == "0xCACHED"
        substrate._get_block_hash.assert_called_once_with(42)
        substrate.runtime_cache.add_item.assert_called_once_with(
            block_hash="0xCACHED", block=42
        )


class TestGetBlockNumber:
    def _make_substrate(self):
        s = SubstrateInterface("ws://localhost", _mock=True)
        s.runtime_cache = MagicMock()
        s._cached_get_block_number = MagicMock(return_value=100)
        s._get_block_number = MagicMock(return_value=99)
        return s

    def test_none_block_hash_calls_get_block_number_directly(self):
        substrate = self._make_substrate()
        result = substrate.get_block_number(None)
        assert result == 99
        substrate._get_block_number.assert_called_once_with(None)
        substrate._cached_get_block_number.assert_not_called()

    def test_cache_hit_returns_cached_number(self):
        substrate = self._make_substrate()
        substrate.runtime_cache.blocks_reverse.get.return_value = 42
        result = substrate.get_block_number("0xABC")
        assert result == 42
        substrate.runtime_cache.blocks_reverse.get.assert_called_once_with("0xABC")
        substrate._cached_get_block_number.assert_not_called()

    def test_cache_miss_fetches_and_stores(self):
        substrate = self._make_substrate()
        substrate.runtime_cache.blocks_reverse.get.return_value = None
        result = substrate.get_block_number("0xABC")
        assert result == 100
        substrate._cached_get_block_number.assert_called_once_with(block_hash="0xABC")
        substrate.runtime_cache.add_item.assert_called_once_with(
            block_hash="0xABC", block=100
        )


class TestExtrinsicReceiptProcessEvents:
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
        substrate.metadata = MagicMock()
        substrate.get_events = MagicMock(return_value=events)
        receipt = ExtrinsicReceipt(
            substrate=substrate,
            extrinsic_hash="0xdeadbeef",
            block_hash="0xabc",
            extrinsic_idx=0,
        )
        return receipt, substrate

    def test_extracts_dispatch_info_weight(self):
        events = [
            self._make_event(
                "System",
                "ExtrinsicSuccess",
                {"dispatch_info": {"weight": {"ref_time": 1, "proof_size": 2}}},
            )
        ]
        receipt, _ = self._make_receipt(events)

        assert receipt.is_success is True
        assert receipt.error_message is None
        assert receipt.weight == {"ref_time": 1, "proof_size": 2}

    def test_extracts_legacy_weight(self):
        events = [self._make_event("System", "ExtrinsicSuccess", {"weight": 7})]
        receipt, _ = self._make_receipt(events)

        assert receipt.is_success is True
        assert receipt.error_message is None
        assert receipt.weight == 7

    def test_prefers_transaction_fee_paid_over_deposit_fallback(self):
        events = [
            self._make_event(
                "TransactionPayment",
                "TransactionFeePaid",
                {"actual_fee": 10},
            ),
            self._make_event("Treasury", "Deposit", {"value": 99}),
            self._make_event("Balances", "Deposit", {"amount": 88}),
        ]
        receipt, _ = self._make_receipt(events)

        assert receipt.total_fee_amount == 10

    def test_accumulates_fallback_fee_from_deposits(self):
        events = [
            self._make_event("Treasury", "Deposit", {"value": 3}),
            self._make_event("Balances", "Deposit", {"amount": 2}),
        ]
        receipt, _ = self._make_receipt(events)

        assert receipt.total_fee_amount == 5

    def test_decodes_legacy_module_error_tuple(self):
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
        receipt, substrate = self._make_receipt(events)
        substrate.metadata.get_module_error.return_value = self._make_module_error(
            name="InsufficientBalance",
            docs=["balance too low"],
        )

        assert receipt.is_success is False
        assert receipt.error_message == {
            "type": "Module",
            "name": "InsufficientBalance",
            "docs": ["balance too low"],
        }
        assert receipt.weight == 9
        substrate.metadata.get_module_error.assert_called_once_with(
            module_index=3, error_index=4
        )

    def test_decodes_module_error_from_hex_error_bytes(self):
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
        receipt, substrate = self._make_receipt(events)
        substrate.metadata.get_module_error.return_value = self._make_module_error(
            name="DecodedHexError",
            docs=["decoded from first byte"],
        )

        assert receipt.is_success is False
        assert receipt.error_message == {
            "type": "Module",
            "name": "DecodedHexError",
            "docs": ["decoded from first byte"],
        }
        assert receipt.weight == 9
        substrate.metadata.get_module_error.assert_called_once_with(
            module_index=5, error_index=10
        )

    def test_maps_bad_origin_error(self):
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
        receipt, substrate = self._make_receipt(events)

        assert receipt.is_success is False
        assert receipt.error_message == {
            "type": "System",
            "name": "BadOrigin",
            "docs": "Bad origin",
        }
        assert receipt.weight == 9
        substrate.metadata.get_module_error.assert_not_called()

    def test_maps_cannot_lookup_error(self):
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
        receipt, substrate = self._make_receipt(events)

        assert receipt.is_success is False
        assert receipt.error_message == {
            "type": "System",
            "name": "CannotLookup",
            "docs": "Cannot lookup",
        }
        assert receipt.weight == 9
        substrate.metadata.get_module_error.assert_not_called()

    def test_maps_token_error(self):
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
        receipt, substrate = self._make_receipt(events)

        assert receipt.is_success is False
        assert receipt.error_message == {
            "type": "System",
            "name": "Token",
            "docs": "FundsUnavailable",
        }
        assert receipt.weight == 9
        substrate.metadata.get_module_error.assert_not_called()

    def test_preserves_unknown_dispatch_error_as_none(self):
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
        receipt, substrate = self._make_receipt(events)

        assert receipt.is_success is False
        assert receipt.error_message is None
        assert receipt.weight == 9
        substrate.metadata.get_module_error.assert_not_called()

    def test_failure_takes_precedence_over_success(self):
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
        receipt, substrate = self._make_receipt(events)

        assert receipt.is_success is False
        assert receipt.error_message == {
            "type": "System",
            "name": "Other",
            "docs": "Unspecified error occurred",
        }
        assert receipt.weight == 9
        substrate.metadata.get_module_error.assert_not_called()

    def test_maps_mevshield_decrypted_rejected_error(self):
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
        receipt, substrate = self._make_receipt(events)

        assert receipt.is_success is False
        assert receipt.error_message == {
            "type": "System",
            "name": "Other",
            "docs": "Unspecified error occurred",
        }
        assert receipt.weight == 123
        assert receipt.total_fee_amount == 0
        substrate.metadata.get_module_error.assert_not_called()

    def test_maps_mevshield_decryption_failed_error(self):
        events = [
            self._make_event(
                "MevShield",
                "DecryptionFailed",
                {"reason": "ciphertext could not be decrypted"},
            )
        ]
        receipt, substrate = self._make_receipt(events)

        assert receipt.is_success is False
        assert receipt.error_message == {
            "type": "MevShield",
            "name": "DecryptionFailed",
            "docs": "ciphertext could not be decrypted",
        }
        assert receipt.total_fee_amount == 0
        assert receipt.weight is None
        substrate.metadata.get_module_error.assert_not_called()
