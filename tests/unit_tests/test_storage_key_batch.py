"""
Parity tests for StorageKey.create_from_storage_function_batch.

The batch builder must produce byte-identical storage keys to calling
create_from_storage_function one-by-one, while resolving metadata and the
prefix hash only once. Metadata is loaded offline from the node-template
fixture, so these tests need no network.
"""

import unittest

from scalecodec import ScaleBytes

from async_substrate_interface.errors import StorageFunctionNotFound
from async_substrate_interface.sync_substrate import SubstrateInterface
from async_substrate_interface.utils.storage import StorageKey
from tests.helpers.fixtures import metadata_node_template_hex


class StorageKeyBatchTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.substrate = SubstrateInterface(
            url="dummy",
            ss58_format=42,
            type_registry_preset="substrate-node-template",
            _mock=True,
        )
        cls.runtime_config = cls.substrate.runtime_config
        metadata = cls.substrate.runtime_config.create_scale_object(
            "MetadataVersioned", ScaleBytes(metadata_node_template_hex)
        )
        metadata.decode()
        cls.metadata = metadata

    def _per_key(self, pallet, sf, params):
        return StorageKey.create_from_storage_function(
            pallet,
            sf,
            params,
            runtime_config=self.runtime_config,
            metadata=self.metadata,
        )

    def _batch(self, pallet, sf, params_list):
        return StorageKey.create_from_storage_function_batch(
            pallet,
            sf,
            params_list,
            runtime_config=self.runtime_config,
            metadata=self.metadata,
        )

    def _assert_parity(self, pallet, sf, params_list):
        batched = self._batch(pallet, sf, params_list)
        self.assertEqual(len(batched), len(params_list))
        for params, batch_key in zip(params_list, batched):
            ref = self._per_key(pallet, sf, params)
            self.assertEqual(
                batch_key.to_hex(),
                ref.to_hex(),
                msg=f"{pallet}.{sf} params={params}",
            )
            # The batch object must carry the same derived attributes the
            # per-key path sets, so downstream decoding behaves identically.
            self.assertEqual(batch_key.value_scale_type, ref.value_scale_type)
            self.assertIsNotNone(batch_key.metadata_storage_function)
            assert batch_key.metadata_storage_function is not None
            assert ref.metadata_storage_function is not None
            self.assertEqual(
                batch_key.metadata_storage_function.value,
                ref.metadata_storage_function.value,
            )

    # --- AccountId key: exercises ss58 -> 0x conversion (Blake2_128Concat) ---
    def test_account_id_ss58_params(self):
        addrs = [
            "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            "5DAAnrj7VHTznn2AWBemMuyBwZWs6FNFjdyVXUeYum3PTXFy",
            "5GNJqTPyNqANBkUVMN1LPPrxXnFouWXoe2wNSmmEoLctxiZY",
        ]
        self._assert_parity("System", "Account", [[a] for a in addrs])

    # --- AccountId key supplied as raw 0x hex (no ss58 decode) ---
    def test_account_id_hex_params(self):
        hexes = [
            "0x" + "11" * 32,
            "0x" + "ab" * 32,
            "0x" + "00" * 32,
        ]
        self._assert_parity("Balances", "Account", [[h] for h in hexes])

    # --- Integer key with a different hasher (Twox64Concat) ---
    def test_integer_key_twox64(self):
        self._assert_parity(
            "System", "BlockHash", [[n] for n in (0, 1, 42, 999, 2**31)]
        )

    # --- Storage function with no params (plain value) ---
    def test_no_param_storage_function(self):
        self._assert_parity("Timestamp", "Now", [[]])

    # --- Pre-encoded ScaleBytes param passes through unchanged ---
    def test_scalebytes_param_passthrough(self):
        obj = self.runtime_config.create_scale_object(type_string="BlockNumber")
        encoded = obj.encode(7)
        self._assert_parity("System", "BlockHash", [[encoded]])

    # --- Large batch stays correct (and is the perf path) ---
    def test_large_batch_parity_sample(self):
        params_list = [[n] for n in range(5000)]
        batched = self._batch("System", "BlockHash", params_list)
        self.assertEqual(len(batched), 5000)
        # Spot-check a sample against the per-key reference.
        for i in (0, 1, 2500, 4999):
            self.assertEqual(
                batched[i].to_hex(),
                self._per_key("System", "BlockHash", [i]).to_hex(),
            )

    # --- Empty input yields empty output ---
    def test_empty_params_list(self):
        self.assertEqual(self._batch("System", "BlockHash", []), [])

    # --- Error handling matches the per-key path ---
    def test_unknown_pallet_raises(self):
        with self.assertRaises(StorageFunctionNotFound):
            self._batch("NotAPallet", "Whatever", [[0]])

    def test_unknown_storage_function_raises(self):
        with self.assertRaises(StorageFunctionNotFound):
            self._batch("System", "NotAStorageFunction", [[0]])


if __name__ == "__main__":
    unittest.main()
