import binascii
from typing import Any, Optional

from scalecodec import ScaleBytes, GenericMetadataVersioned
from scalecodec.base import ScaleDecoder, RuntimeConfigurationObject, ScaleType
from scalecodec.utils.ss58 import ss58_decode
from async_substrate_interface.errors import StorageFunctionNotFound
from async_substrate_interface.utils.hasher import (
    blake2_256,
    two_x64_concat,
    xxh128,
    blake2_128,
    blake2_128_concat,
    identity,
)

try:
    from typing import Self
except ImportError:
    # fallback to typing_extensions if Python < 3.11
    from typing_extensions import Self

# Single source of truth mapping a metadata hasher name to its implementation.
# `None`/empty hasher defaults to "Twox128" (matches substrate behaviour).
PARAM_HASHERS = {
    "Blake2_256": blake2_256,
    "Blake2_128": blake2_128,
    "Blake2_128Concat": blake2_128_concat,
    "Twox128": xxh128,
    "Twox64Concat": two_x64_concat,
    "Identity": identity,
}


class StorageKey:
    """
    A StorageKey instance is a representation of a single state entry.

    Substrate uses a simple key-value data store implemented as a database-backed, modified Merkle tree.
    All of Substrate's higher-level storage abstractions are built on top of this simple key-value store.
    """

    def __init__(
        self,
        pallet: Optional[str],
        storage_function: Optional[str],
        params: Optional[list],
        data: Optional[bytes],
        value_scale_type: Optional[str],
        metadata: GenericMetadataVersioned,
        runtime_config: RuntimeConfigurationObject,
    ):
        self.pallet = pallet
        self.storage_function = storage_function
        self.params = params
        self.params_encoded: list[Any] = []
        self.data = data
        self.metadata = metadata
        self.runtime_config = runtime_config
        self.value_scale_type = value_scale_type
        self.metadata_storage_function: Optional[Any] = None

    @classmethod
    def create_from_data(
        cls,
        data: bytes,
        runtime_config: RuntimeConfigurationObject,
        metadata: GenericMetadataVersioned,
        value_scale_type: Optional[str] = None,
        pallet: Optional[str] = None,
        storage_function: Optional[str] = None,
    ) -> Self:
        """
        Create a StorageKey instance providing raw storage key bytes

        Args:
            data: bytes representation of the storage key
            runtime_config: RuntimeConfigurationObject
            metadata: GenericMetadataVersioned
            value_scale_type: type string of to decode result data
            pallet: name of pallet
            storage_function: name of storage function

        Returns:
            StorageKey
        """
        if not value_scale_type and pallet and storage_function:
            metadata_pallet = metadata.get_metadata_pallet(pallet)

            if not metadata_pallet:
                raise StorageFunctionNotFound(f'Pallet "{pallet}" not found')

            storage_item = metadata_pallet.get_storage_function(storage_function)

            if not storage_item:
                raise StorageFunctionNotFound(
                    f'Storage function "{pallet}.{storage_function}" not found'
                )

            # Process specific type of storage function
            value_scale_type = storage_item.get_value_type_string()

        return cls(
            pallet=None,
            storage_function=None,
            params=None,
            data=data,
            metadata=metadata,
            value_scale_type=value_scale_type,
            runtime_config=runtime_config,
        )

    @classmethod
    def create_from_storage_function(
        cls,
        pallet: str,
        storage_function: str,
        params: list,
        runtime_config: RuntimeConfigurationObject,
        metadata: GenericMetadataVersioned,
    ) -> Self:
        """
        Create a StorageKey instance providing storage function details

        Args:
            pallet: name of pallet
            storage_function: name of storage function
            params: Optional list of parameters in case of a Mapped storage function
            runtime_config: RuntimeConfigurationObject
            metadata: GenericMetadataVersioned

        Returns:
            StorageKey
        """
        storage_key_obj = cls(
            pallet=pallet,
            storage_function=storage_function,
            params=params,
            data=None,
            runtime_config=runtime_config,
            metadata=metadata,
            value_scale_type=None,
        )

        storage_key_obj.generate()

        return storage_key_obj

    @classmethod
    def create_from_storage_function_batch(
        cls,
        pallet: str,
        storage_function: str,
        params_list: list[list],
        runtime_config: RuntimeConfigurationObject,
        metadata: GenericMetadataVersioned,
    ) -> list[Self]:
        """
        Create many StorageKey instances for the same pallet/storage_function in
        one pass, one per entry in ``params_list``.

        This is much faster than calling :meth:`create_from_storage_function`
        in a loop: everything that is constant across the keys (metadata
        resolution, the pallet/storage-function prefix hash, and the scale
        objects used to encode params) is computed once and reused. For large
        batches (e.g. 100k keys) this is ~30x faster while producing
        byte-identical keys.

        Args:
            pallet: name of pallet
            storage_function: name of storage function
            params_list: list of parameter lists, one per storage key to create
            runtime_config: RuntimeConfigurationObject
            metadata: GenericMetadataVersioned

        Returns:
            list of StorageKey, in the same order as ``params_list``
        """
        # --- Resolve everything that is constant across the batch, once. ---
        metadata_pallet = metadata.get_metadata_pallet(pallet)
        if not metadata_pallet:
            raise StorageFunctionNotFound(f'Pallet "{pallet}" not found')

        metadata_storage_function = metadata_pallet.get_storage_function(
            storage_function
        )
        if not metadata_storage_function:
            raise StorageFunctionNotFound(
                f'Storage function "{pallet}.{storage_function}" not found'
            )

        value_scale_type = metadata_storage_function.get_value_type_string()
        param_types = metadata_storage_function.get_params_type_string()
        hashers = metadata_storage_function.get_param_hashers()

        # Immutable bytes: each key does `storage_hash = prefix` then `+=`, which
        # must allocate a new object rather than mutate this shared prefix. xxh128
        # returns a bytearray, so wrap it to prevent in-place accumulation.
        prefix = bytes(
            xxh128(metadata_pallet.value["storage"]["prefix"].encode())
            + xxh128(storage_function.encode())
        )

        n_params = len(param_types)

        # One reusable scale object and one resolved hasher fn per param position.
        scale_objects = [
            runtime_config.create_scale_object(type_string=param_types[idx])
            for idx in range(n_params)
        ]
        hasher_fns = []
        for idx in range(n_params):
            param_hasher = hashers[idx] if idx < len(hashers) else None
            try:
                hasher_fns.append(PARAM_HASHERS[param_hasher or "Twox128"])
            except KeyError:
                raise ValueError('Unknown storage hasher "{}"'.format(param_hasher))

        ss58_format = runtime_config.ss58_format

        # --- Per-key work only. ---
        storage_keys: list[Self] = []
        for params in params_list:
            storage_hash = prefix
            params_encoded: list[Any] = []
            for idx, param in enumerate(params):
                if type(param) is ScaleBytes:
                    # Already encoded
                    encoded = param
                    params_key = param.data
                else:
                    param = cls._convert_storage_parameter(
                        param_types[idx], param, ss58_format
                    )
                    encoded = scale_objects[idx].encode(param)
                    params_key = encoded.data
                params_encoded.append(encoded)
                storage_hash += hasher_fns[idx](params_key)

            storage_key_obj = cls(
                pallet=pallet,
                storage_function=storage_function,
                params=params,
                data=None,
                runtime_config=runtime_config,
                metadata=metadata,
                value_scale_type=value_scale_type,
            )
            # Mirror generate(): the hash is assigned onto self.data directly.
            storage_key_obj.data = storage_hash
            storage_key_obj.metadata_storage_function = metadata_storage_function
            storage_key_obj.params_encoded = params_encoded
            storage_keys.append(storage_key_obj)

        return storage_keys

    @staticmethod
    def _convert_storage_parameter(
        scale_type: str, value: Any, ss58_format: Optional[int]
    ):
        if type(value) is bytes:
            value = f"0x{value.hex()}"

        if scale_type == "AccountId":
            if value[0:2] != "0x":
                return "0x{}".format(ss58_decode(value, ss58_format))

        return value

    def convert_storage_parameter(self, scale_type: str, value: Any):
        return self._convert_storage_parameter(
            scale_type, value, self.runtime_config.ss58_format
        )

    def to_hex(self) -> Optional[str]:
        """
        Returns a Hex-string representation of current StorageKey data

        Returns:
            Hex string
        """
        if self.data:
            return f"0x{self.data.hex()}"
        return None

    def generate(self) -> bytes:
        """
        Generate a storage key for current specified pallet/function/params
        """

        # Search storage call in metadata
        assert self.pallet is not None
        assert self.storage_function is not None
        metadata_pallet = self.metadata.get_metadata_pallet(self.pallet)

        if not metadata_pallet:
            raise StorageFunctionNotFound(f'Pallet "{self.pallet}" not found')

        self.metadata_storage_function = metadata_pallet.get_storage_function(
            self.storage_function
        )

        if not self.metadata_storage_function:
            raise StorageFunctionNotFound(
                f'Storage function "{self.pallet}.{self.storage_function}" not found'
            )

        # Process specific type of storage function
        self.value_scale_type = self.metadata_storage_function.get_value_type_string()
        param_types = self.metadata_storage_function.get_params_type_string()

        hashers = self.metadata_storage_function.get_param_hashers()

        storage_hash = xxh128(
            metadata_pallet.value["storage"]["prefix"].encode()
        ) + xxh128(self.storage_function.encode())

        # Encode parameters
        self.params_encoded = []
        if self.params:
            for idx, param in enumerate(self.params):
                if type(param) is ScaleBytes:
                    # Already encoded
                    self.params_encoded.append(param)
                else:
                    param = self.convert_storage_parameter(param_types[idx], param)
                    param_obj = self.runtime_config.create_scale_object(
                        type_string=param_types[idx]
                    )
                    self.params_encoded.append(param_obj.encode(param))

            for idx, param in enumerate(self.params_encoded):
                # Get hasher associated with param
                try:
                    param_hasher = hashers[idx]
                except IndexError:
                    raise ValueError(f"No hasher found for param #{idx + 1}")

                params_key = bytes()

                # Convert param to bytes
                if type(param) is str:
                    params_key += binascii.unhexlify(param)
                elif type(param) is ScaleBytes:
                    params_key += param.data
                elif isinstance(param, ScaleDecoder):
                    assert param.data is not None
                    params_key += param.data.data

                try:
                    hasher_fn = PARAM_HASHERS[param_hasher or "Twox128"]
                except KeyError:
                    raise ValueError('Unknown storage hasher "{}"'.format(param_hasher))

                storage_hash += hasher_fn(params_key)

        self.data = storage_hash

        return self.data

    def decode_scale_value(self, data: Optional[ScaleBytes] = None) -> ScaleType:
        result_found = False
        assert self.metadata_storage_function is not None
        assert self.value_scale_type is not None

        if data is not None:
            change_scale_type = self.value_scale_type
            result_found = True
        elif self.metadata_storage_function.value["modifier"] == "Default":
            # Fallback to default value of storage function if no result
            change_scale_type = self.value_scale_type
            data = ScaleBytes(
                self.metadata_storage_function.value_object["default"].value_object
            )
        else:
            # No result is interpreted as an Option<...> result
            change_scale_type = f"Option<{self.value_scale_type}>"
            data = ScaleBytes(
                self.metadata_storage_function.value_object["default"].value_object
            )

        # Decode SCALE result data
        updated_obj = self.runtime_config.create_scale_object(
            type_string=change_scale_type, data=data, metadata=self.metadata
        )
        updated_obj.decode()
        updated_obj.meta_info = {"result_found": result_found}

        return updated_obj

    def __repr__(self):
        if self.pallet and self.storage_function:
            return f"<StorageKey(pallet={self.pallet}, storage_function={self.storage_function}, params={self.params})>"
        elif self.data:
            return f"<StorageKey(data=0x{self.data.hex()})>"
        else:
            return repr(self)
