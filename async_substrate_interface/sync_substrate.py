import functools
import logging
import os
import socket
from hashlib import blake2b
from typing import Optional, Union, Callable, Any
from unittest.mock import MagicMock

import scalecodec
from scalecodec import ScaleBytes
from scalecodec.base import ScaleType
from scalecodec.types import (
    GenericCall,
    GenericExtrinsic,
    MultiAccountId,
)
from websockets.sync.client import connect, ClientConnection
from websockets.exceptions import ConnectionClosed

from async_substrate_interface.types import (
    _decode_option_opaque_metadata,
    _decode_v15_metadata,
)

from async_substrate_interface.errors import (
    ExtrinsicNotFound,
    SubstrateRequestException,
    BlockNotFound,
    MaxRetriesExceeded,
    StateDiscardedError,
)
from async_substrate_interface.protocols import Keypair
from async_substrate_interface.types import (
    SubstrateMixin,
    RuntimeCache,
    Runtime,
    RequestManager,
    Preprocessed,
    RequestResults,
)
from async_substrate_interface.utils import (
    hex_to_bytes,
    json,
    get_next_id,
    rng as random,
)
from async_substrate_interface.utils.decoding import (
    _determine_if_old_runtime_call,
    decode_query_map,
    scale_decode,
    try_batch_decode,
)
from async_substrate_interface.utils.receipt import (
    build_system_error_message,
    extract_failure_details,
    extract_fallback_deposit_fee_amount,
    extract_success_weight,
    extract_total_fee_amount,
    is_extrinsic_failure_event,
    is_extrinsic_success_event,
    normalize_module_error,
)
from async_substrate_interface.utils.storage import StorageKey
from async_substrate_interface.type_registry import _TYPE_REGISTRY


ResultHandler = Callable[[dict, Any], tuple[dict, bool]]

logger = logging.getLogger("async_substrate_interface")
raw_websocket_logger = logging.getLogger("raw_websocket")

# env vars dictating the cache size of the cached methods
SUBSTRATE_CACHE_METHOD_SIZE = int(os.getenv("SUBSTRATE_CACHE_METHOD_SIZE", "512"))
SUBSTRATE_RUNTIME_CACHE_SIZE = int(os.getenv("SUBSTRATE_RUNTIME_CACHE_SIZE", "16"))


class ExtrinsicReceipt:
    """
    Object containing information of submitted extrinsic. Block hash where extrinsic is included is required
    when retrieving triggered events or determine if extrinsic was successful
    """

    def __init__(
        self,
        substrate: "SubstrateInterface",
        extrinsic_hash: Optional[str] = None,
        block_hash: Optional[str] = None,
        block_number: Optional[int] = None,
        extrinsic_idx: Optional[int] = None,
        finalized: bool = False,
    ):
        """
        Object containing information of submitted extrinsic. Block hash where extrinsic is included is required
        when retrieving triggered events or determine if extrinsic was successful

        Args:
            substrate: the AsyncSubstrateInterface instance
            extrinsic_hash: the hash of the extrinsic
            block_hash: the hash of the block on which this extrinsic exists
            finalized: whether the extrinsic is finalized
        """
        self.substrate = substrate
        self.extrinsic_hash = extrinsic_hash
        self.block_hash = block_hash
        self.block_number = block_number
        self.finalized = finalized

        self.__extrinsic_idx = extrinsic_idx
        self.__extrinsic = None

        self.__triggered_events: Optional[list] = None
        self.__is_success: Optional[bool] = None
        self.__error_message = None
        self.__weight = None
        self.__total_fee_amount = None

    def get_extrinsic_identifier(self) -> str:
        """
        Returns the on-chain identifier for this extrinsic in format "[block_number]-[extrinsic_idx]" e.g. 134324-2
        Returns
        -------
        str
        """
        if self.block_number is None:
            if self.block_hash is None:
                raise ValueError(
                    "Cannot create extrinsic identifier: block_hash is not set"
                )

            self.block_number = self.substrate.get_block_number(self.block_hash)

            if self.block_number is None:
                raise ValueError(
                    "Cannot create extrinsic identifier: unknown block_hash"
                )

        return f"{self.block_number}-{self.extrinsic_idx}"

    def retrieve_extrinsic(self):
        if not self.block_hash:
            raise ValueError(
                "ExtrinsicReceipt can't retrieve events because it's unknown which block_hash it is "
                "included, manually set block_hash or use `wait_for_inclusion` when sending extrinsic"
            )
        # Determine extrinsic idx

        block = self.substrate.get_block(block_hash=self.block_hash)

        extrinsics = block["extrinsics"]

        if len(extrinsics) > 0:
            if self.__extrinsic_idx is None:
                self.__extrinsic_idx = self.__get_extrinsic_index(
                    block_extrinsics=extrinsics, extrinsic_hash=self.extrinsic_hash
                )

            if self.__extrinsic_idx >= len(extrinsics):
                raise ExtrinsicNotFound()

            self.__extrinsic = extrinsics[self.__extrinsic_idx]

    @property
    def extrinsic_idx(self) -> int:
        """
        Retrieves the index of this extrinsic in containing block

        Returns
        -------
        int
        """
        if self.__extrinsic_idx is None:
            self.retrieve_extrinsic()
        return self.__extrinsic_idx

    @property
    def triggered_events(self) -> list:
        """
        Gets triggered events for submitted extrinsic. block_hash where extrinsic is included is required, manually
        set block_hash or use `wait_for_inclusion` when submitting extrinsic

        Returns
        -------
        list
        """
        if self.__triggered_events is None:
            if not self.block_hash:
                raise ValueError(
                    "ExtrinsicReceipt can't retrieve events because it's unknown which block_hash it is "
                    "included, manually set block_hash or use `wait_for_inclusion` when sending extrinsic"
                )

            if self.extrinsic_idx is None:
                self.retrieve_extrinsic()

            self.__triggered_events = []

            for event in self.substrate.get_events(block_hash=self.block_hash):
                if event["extrinsic_idx"] == self.extrinsic_idx:
                    self.__triggered_events.append(event)

        return self.__triggered_events

    @classmethod
    def create_from_extrinsic_identifier(
        cls, substrate: "SubstrateInterface", extrinsic_identifier: str
    ) -> "ExtrinsicReceipt":
        """
        Create an `AsyncExtrinsicReceipt` with on-chain identifier for this extrinsic in format
        "[block_number]-[extrinsic_idx]" e.g. 134324-2

        Args:
            substrate: SubstrateInterface
            extrinsic_identifier: "[block_number]-[extrinsic_idx]" e.g. 134324-2

        Returns:
            AsyncExtrinsicReceipt of the extrinsic
        """
        id_parts = extrinsic_identifier.split("-", maxsplit=1)
        block_number: int = int(id_parts[0])
        extrinsic_idx: int = int(id_parts[1])

        # Retrieve block hash
        block_hash = substrate.get_block_hash(block_number)

        return cls(
            substrate=substrate,
            block_hash=block_hash,
            block_number=block_number,
            extrinsic_idx=extrinsic_idx,
        )

    def process_events(self):
        if self.triggered_events:
            self.__total_fee_amount = 0

            self.__total_fee_amount, has_transaction_fee_paid_event = (
                extract_total_fee_amount(self.triggered_events)
            )

            # Process other events
            possible_success = False
            for event in self.triggered_events:
                if is_extrinsic_success_event(event):
                    possible_success = True
                    self.__weight = extract_success_weight(event)
                elif is_extrinsic_failure_event(event):
                    possible_success = False
                    self.__is_success = False
                    failure_details = extract_failure_details(event)
                    if failure_details["has_weight"]:
                        self.__weight = failure_details["weight"]
                    if failure_details["error_message"] is not None:
                        self.__error_message = failure_details["error_message"]
                        continue

                    dispatch_error = failure_details["dispatch_error"]
                    if dispatch_error is None:
                        continue

                    module_error = normalize_module_error(dispatch_error)
                    if module_error is not None:
                        self.__error_message = self._resolve_module_error_message(
                            module_index=module_error["module_index"],
                            error_index=module_error["error_index"],
                        )
                    else:
                        self.__error_message = build_system_error_message(
                            dispatch_error
                        )
                elif not has_transaction_fee_paid_event:
                    self.__total_fee_amount += extract_fallback_deposit_fee_amount(
                        event
                    )
            if possible_success is True and self.__error_message is None:
                # we delay the positive setting of the __is_success flag until we have finished iteration of the
                # events and have ensured nothing has set an error message
                self.__is_success = True

    def _resolve_module_error_message(
        self, module_index: int, error_index: int
    ) -> dict:
        module_error = self.substrate.metadata.get_module_error(
            module_index=module_index, error_index=error_index
        )
        return {
            "type": "Module",
            "name": module_error.name,
            "docs": module_error.docs,
        }

    @property
    def is_success(self) -> bool:
        """
        Returns `True` if `ExtrinsicSuccess` event is triggered, `False` in case of `ExtrinsicFailed`
        In case of False `error_message` will contain more details about the error


        Returns
        -------
        bool
        """
        if self.__is_success is None:
            self.process_events()

        return self.__is_success

    @property
    def error_message(self) -> Optional[dict]:
        """
        Returns the error message if the extrinsic failed in format e.g.:

        `{'type': 'System', 'name': 'BadOrigin', 'docs': 'Bad origin'}`

        Returns
        -------
        dict
        """
        if self.__error_message is None:
            if self.is_success:
                return None
            self.process_events()
        return self.__error_message

    @property
    def weight(self) -> Union[int, dict]:
        """
        Contains the actual weight when executing this extrinsic

        Returns
        -------
        int (WeightV1) or dict (WeightV2)
        """
        if self.__weight is None:
            self.process_events()
        return self.__weight

    @property
    def total_fee_amount(self) -> int:
        """
        Contains the total fee costs deducted when executing this extrinsic. This includes fee for the validator
            (`Balances.Deposit` event) and the fee deposited for the treasury (`Treasury.Deposit` event)

        Returns
        -------
        int
        """
        if self.__total_fee_amount is None:
            self.process_events()
        return self.__total_fee_amount

    # Helper functions
    @staticmethod
    def __get_extrinsic_index(block_extrinsics: list, extrinsic_hash: str) -> int:
        """
        Returns the index of a provided extrinsic
        """
        for idx, extrinsic in enumerate(block_extrinsics):
            if (
                extrinsic.extrinsic_hash
                and f"0x{extrinsic.extrinsic_hash.hex()}" == extrinsic_hash
            ):
                return idx
        raise ExtrinsicNotFound()

    # Backwards compatibility methods
    def __getitem__(self, item):
        return getattr(self, item)

    def __iter__(self):
        for item in self.__dict__.items():
            yield item

    def get(self, name):
        return self[name]


class QueryMapResult:
    def __init__(
        self,
        records: list,
        page_size: int,
        substrate: "SubstrateInterface",
        module: Optional[str] = None,
        storage_function: Optional[str] = None,
        params: Optional[list] = None,
        block_hash: Optional[str] = None,
        last_key: Optional[str] = None,
        max_results: Optional[int] = None,
        ignore_decoding_errors: bool = False,
    ):
        self.records = records
        self.page_size = page_size
        self.module = module
        self.storage_function = storage_function
        self.block_hash = block_hash
        self.substrate = substrate
        self.last_key = last_key
        self.max_results = max_results
        self.params = params
        self.ignore_decoding_errors = ignore_decoding_errors
        self.loading_complete = False
        self._buffer = iter(self.records)  # Initialize the buffer with initial records

    def retrieve_next_page(self, start_key) -> list:
        result = self.substrate.query_map(
            module=self.module,
            storage_function=self.storage_function,
            params=self.params,
            page_size=self.page_size,
            block_hash=self.block_hash,
            start_key=start_key,
            max_results=self.max_results,
            ignore_decoding_errors=self.ignore_decoding_errors,
        )
        if len(result.records) < self.page_size:
            self.loading_complete = True

        # Update last key from new result set to use as offset for next page
        self.last_key = result.last_key
        return result.records

    def retrieve_all_records(self) -> list[Any]:
        """
        Retrieves all records from all subsequent pages for the QueryMapResult,
        returning them as a list.

        Side effect:
            The self.records list will be populated fully after running this method.
        """
        for _ in self:
            pass
        return self.records

    def __iter__(self):
        return self

    def get_next_record(self):
        try:
            # Try to get the next record from the buffer
            record = next(self._buffer)
        except StopIteration:
            # If no more records in the buffer
            return False, None
        else:
            return True, record

    def __next__(self):
        successfully_retrieved, record = self.get_next_record()
        if successfully_retrieved:
            return record

        # If loading is already completed
        if self.loading_complete:
            raise StopIteration

        next_page = self.retrieve_next_page(self.last_key)

        # If we cannot retrieve the next page
        if not next_page:
            self.loading_complete = True
            raise StopIteration

        self.records.extend(next_page)
        # Update the buffer with the newly fetched records
        self._buffer = iter(next_page)
        return next(self._buffer)

    def __getitem__(self, item):
        return self.records[item]


class SubstrateInterface(SubstrateMixin):
    def __init__(
        self,
        url: str,
        use_remote_preset: bool = False,
        auto_discover: bool = True,
        ss58_format: Optional[int] = None,
        type_registry: Optional[dict] = None,
        type_registry_preset: Optional[str] = None,
        chain_name: str = "",
        max_retries: int = 5,
        retry_timeout: float = 60.0,
        _mock: bool = False,
        _log_raw_websockets: bool = False,
    ):
        """
        The sync compatible version of the subtensor interface commands we use in bittensor. Use this instance only
        if you are not running within an event loop, otherwise use AsyncSubstrateInterface

        Args:
            url: the URI of the chain to connect to
            use_remote_preset: whether to pull the preset from GitHub
            auto_discover: whether to automatically pull the presets based on the chain name and type registry
            ss58_format: the specific SS58 format to use
            type_registry: a dict of custom types
            type_registry_preset: preset
            chain_name: the name of the chain (the result of the rpc request for "system_chain")
            max_retries: number of times to retry RPC requests before giving up
            retry_timeout: how to long wait since the last ping to retry the RPC request
            _mock: whether to use mock version of the subtensor interface
            _log_raw_websockets: whether to log raw websocket requests during RPC requests

        """
        super().__init__(
            type_registry,
            type_registry_preset,
            use_remote_preset,
            ss58_format,
        )
        self.max_retries = max_retries
        self.retry_timeout = retry_timeout
        self.chain_endpoint = url
        self.url = url
        self._chain = chain_name
        self.config = {
            "use_remote_preset": use_remote_preset,
            "auto_discover": auto_discover,
            "rpc_methods": None,
            "strict_scale_decode": True,
        }
        self.initialized = False
        self.type_registry = type_registry
        self.type_registry_preset = type_registry_preset
        self.runtime_cache = RuntimeCache()
        self.metadata_version_hex = "0x0f000000"  # v15
        self._mock = _mock
        self.log_raw_websockets = _log_raw_websockets
        if not _mock:
            self.ws = self.connect(init=True)
            self.initialize()
        else:
            self.ws = MagicMock(spec=ClientConnection)

    def __enter__(self):
        if not self._mock:
            self.initialize()
        return self

    def __del__(self):
        try:
            self.ws.close()
        except AttributeError:
            pass
        # self.ws.protocol.fail(code=1006)  # ABNORMAL_CLOSURE

    def initialize(self):
        """
        Initialize the connection to the chain.
        """
        if not self.initialized:
            if not self._chain:
                chain = self.rpc_request("system_chain", [])
                self._chain = chain.get("result")
            self.init_runtime()
            if self.ss58_format is None:
                # Check and apply runtime constants
                ss58_prefix_constant = self.get_constant(
                    "System", "SS58Prefix", block_hash=self.last_block_hash
                )
                if ss58_prefix_constant:
                    self.ss58_format = ss58_prefix_constant.value
                    self.runtime.ss58_format = ss58_prefix_constant.value
                    self.runtime.runtime_config.ss58_format = ss58_prefix_constant.value
        self.initialized = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.ws.close()

    @property
    def metadata(self):
        if not self.runtime or self.runtime.metadata is None:
            raise AttributeError(
                "Metadata not found. This generally indicates that the AsyncSubstrateInterface object "
                "is not properly async initialized."
            )
        else:
            return self.runtime.metadata

    @property
    def properties(self):
        if self._properties is None:
            self._properties = self.rpc_request("system_properties", []).get("result")
        return self._properties

    @property
    def version(self):
        if self._version is None:
            self._version = self.rpc_request("system_version", []).get("result")
        return self._version

    @property
    def token_decimals(self):
        if self._token_decimals is None:
            self._token_decimals = self.properties.get("tokenDecimals")
        return self._token_decimals

    @property
    def token_symbol(self):
        if self._token_symbol is None:
            if self.properties:
                self._token_symbol = self.properties.get("tokenSymbol")
            else:
                self._token_symbol = "UNIT"
        return self._token_symbol

    @property
    def name(self):
        if self._name is None:
            self._name = self.rpc_request("system_name", []).get("result")
        return self._name

    def connect(self, init=False):
        if init is True:
            try:
                logger.debug(f"Websocket connecting to {self.chain_endpoint}")
                return connect(self.chain_endpoint, max_size=self.ws_max_size)
            except (ConnectionError, socket.gaierror) as e:
                raise ConnectionError(e)
        else:
            if not self.ws.close_code:
                return self.ws
            else:
                try:
                    logger.debug(f"Websocket reconnecting to {self.chain_endpoint}")
                    self.ws = connect(self.chain_endpoint, max_size=self.ws_max_size)
                    return self.ws
                except (ConnectionError, socket.gaierror) as e:
                    raise ConnectionError(e)

    def get_storage_item(
        self, module: str, storage_function: str, block_hash: Optional[str] = None
    ):
        self.init_runtime(block_hash=block_hash)
        metadata_pallet = self.runtime.metadata.get_metadata_pallet(module)
        storage_item = metadata_pallet.get_storage_function(storage_function)
        return storage_item

    def _get_current_block_hash(
        self, block_hash: Optional[str], reuse: bool
    ) -> Optional[str]:
        if block_hash:
            self.last_block_hash = block_hash
            return block_hash
        elif reuse:
            if self.last_block_hash:
                return self.last_block_hash
        return block_hash

    def _load_registry_at_block(self, block_hash: Optional[str], runtime_config=None):
        # Should be called for any block that fails decoding.
        # Possibly the metadata was different.
        try:
            metadata_rpc_result = self.rpc_request(
                "state_call",
                ["Metadata_metadata_at_version", self.metadata_version_hex],
                block_hash=block_hash,
            )
        except SubstrateRequestException as e:
            if (
                "Client error: Execution failed: Other: Exported method Metadata_metadata_at_version is not found"
                in e.args
            ):
                return None
            else:
                raise e
        metadata_option_hex_str = metadata_rpc_result["result"]
        metadata_option_bytes = bytes.fromhex(metadata_option_hex_str[2:])
        inner_bytes = _decode_option_opaque_metadata(metadata_option_bytes)
        if inner_bytes is None:
            return None
        return _decode_v15_metadata(inner_bytes, runtime_config=runtime_config)

    def decode_scale(
        self,
        type_string: str,
        scale_bytes: bytes,
    ) -> Optional[ScaleType]:
        """
        Helper function to decode arbitrary SCALE-bytes (e.g. 0x02000000) according to given RUST type_string
        (e.g. BlockNumber). The relevant versioning information of the type (if defined) will be applied if block_hash
        is set

        Args:
            type_string: the type string of the SCALE object for decoding
            scale_bytes: the bytes representation of the SCALE object to decode

        Returns:
            ScaleType object
        """
        if scale_bytes == b"":
            return None
        else:
            obj = scale_decode(type_string, scale_bytes, runtime=self.runtime)
            return obj

    def load_runtime(self, runtime):
        self.runtime = runtime

        # Update type registry
        self.runtime.reload_type_registry(use_remote_preset=False, auto_discover=True)

        self.runtime_config.set_active_spec_version_id(runtime.runtime_version)
        if self.runtime.implements_scaleinfo:
            logger.debug("Add PortableRegistry from metadata to type registry")
            self.runtime_config.add_portable_registry(runtime.metadata)
        # Set runtime compatibility flags
        try:
            _ = self.runtime_config.create_scale_object("sp_weights::weight_v2::Weight")
            self.runtime.config["is_weight_v2"] = True
            self.runtime_config.update_type_registry_types(
                {"Weight": "sp_weights::weight_v2::Weight"}
            )
        except NotImplementedError:
            self.runtime.config["is_weight_v2"] = False
            self.runtime_config.update_type_registry_types({"Weight": "WeightV1"})

    def init_runtime(
        self, block_hash: Optional[str] = None, block_id: Optional[int] = None
    ) -> Runtime:
        """
        This method is used by all other methods that deals with metadata and types defined in the type registry.
        It optionally retrieves the block_hash when block_id is given and sets the applicable metadata for that
        block_hash. Also, it applies all the versioned types at the time of the block_hash.

        Because parsing of metadata and type registry is quite heavy, the result will be cached per runtime id.
        In the future there could be support for caching backends like Redis to make this cache more persistent.

        Args:
            block_hash: optional block hash, should not be specified if block_id is
            block_id: optional block id, should not be specified if block_hash is

        Returns:
            Runtime object
        """

        if block_id and block_hash:
            raise ValueError("Cannot provide block_hash and block_id at the same time")

        if block_id is not None:
            if runtime := self.runtime_cache.retrieve(block=block_id):
                runtime.load_runtime()
                if runtime.implements_scaleinfo:
                    runtime.load_registry_type_map()
                self.runtime = runtime
                return self.runtime
            block_hash = self.get_block_hash(block_id)

        if not block_hash:
            block_hash = self.get_chain_head()
        else:
            self.last_block_hash = block_hash
            if runtime := self.runtime_cache.retrieve(block_hash=block_hash):
                runtime.load_runtime()
                if runtime.implements_scaleinfo:
                    runtime.load_registry_type_map()
                self.runtime = runtime
                return self.runtime

        runtime_version = self.get_block_runtime_version_for(block_hash)
        if runtime_version is None:
            raise SubstrateRequestException(
                f"No runtime information for block '{block_hash}'"
            )

        if self.runtime and runtime_version == self.runtime.runtime_version:
            return self.runtime

        if (
            runtime := self.runtime_cache.retrieve(runtime_version=runtime_version)
        ) is not None:
            pass
        else:
            runtime = self.get_runtime_for_version(runtime_version, block_hash)
        runtime.load_runtime()
        if runtime.implements_scaleinfo:
            runtime.load_registry_type_map()
        self.runtime = runtime
        return self.runtime

    @functools.lru_cache(maxsize=SUBSTRATE_RUNTIME_CACHE_SIZE)
    def get_runtime_for_version(
        self, runtime_version: int, block_hash: Optional[str] = None
    ) -> Runtime:
        """
        Retrieves the `Runtime` for a given runtime version at a given block hash.
        Args:
            runtime_version: version of the runtime (from `get_block_runtime_version_for`)
            block_hash: hash of the block to query

        Returns:
            Runtime object for the given runtime version
        """
        if not block_hash:
            block_hash = self.get_chain_head()
        runtime_block_hash = self.get_parent_block_hash(block_hash)
        block_number = self.get_block_number(block_hash)
        runtime_info = self.get_block_runtime_info(runtime_block_hash)

        metadata_v15 = self._load_registry_at_block(
            block_hash=runtime_block_hash, runtime_config=self.runtime_config
        )
        if metadata_v15 is not None:
            # V15 is a superset of V14; use it directly, skipping the V14 fetch
            metadata = metadata_v15
            logger.debug(
                f"Retrieved metadata v15 for {runtime_version} from Substrate node"
            )
        else:
            metadata = self.get_block_metadata(
                block_hash=runtime_block_hash, decode=True
            )
            logger.debug(
                f"Exported method Metadata_metadata_at_version is not found for {runtime_version}. "
                f"This indicates the block is quite old, decoding for this block will use legacy Python decoding."
            )
        if metadata is None:
            raise SubstrateRequestException(
                f"No metadata for block '{runtime_block_hash}'"
            )

        runtime = Runtime(
            chain=self.chain,
            runtime_config=self.runtime_config,
            metadata=metadata,
            type_registry=self.type_registry,
            metadata_v15=metadata_v15,
            runtime_info=runtime_info,
            ss58_format=self.ss58_format,
        )
        self.runtime_cache.add_item(
            block=block_number,
            block_hash=block_hash,
            runtime_version=runtime_version,
            runtime=runtime,
        )
        return runtime

    def create_storage_key(
        self,
        pallet: str,
        storage_function: str,
        params: Optional[list] = None,
        block_hash: Optional[str] = None,
    ) -> StorageKey:
        """
        Create a `StorageKey` instance providing storage function details. See `subscribe_storage()`.

        Args:
            pallet: name of pallet
            storage_function: name of storage function
            params: list of parameters in case of a Mapped storage function
            block_hash: the hash of the blockchain block whose runtime to use

        Returns:
            StorageKey
        """
        self.init_runtime(block_hash=block_hash)

        return StorageKey.create_from_storage_function(
            pallet,
            storage_function,
            params or [],
            runtime_config=self.runtime_config,
            metadata=self.runtime.metadata,
        )

    def subscribe_storage(
        self,
        storage_keys: list[StorageKey],
        subscription_handler: Callable[[StorageKey, Any, str], Any],
    ):
        """

        Subscribe to provided storage_keys and keep tracking until `subscription_handler` returns a value

        Example of a StorageKey:
        ```
        StorageKey.create_from_storage_function(
            "System", "Account", ["5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"]
        )
        ```

        Example of a subscription handler:
        ```
        def subscription_handler(storage_key, obj, subscription_id):
            if obj is not None:
                # the subscription will run until your subscription_handler returns something other than `None`
                return obj
        ```

        Args:
            storage_keys: StorageKey list of storage keys to subscribe to
            subscription_handler: function to handle value changes of subscription

        """
        self.init_runtime()

        storage_key_map = {s.to_hex(): s for s in storage_keys}

        def result_handler(
            message: dict, subscription_id: str
        ) -> tuple[bool, Optional[Any]]:
            result_found = False
            subscription_result = None
            if "params" in message:
                # Process changes
                for change_storage_key, change_data in message["params"]["result"][
                    "changes"
                ]:
                    # Check for target storage key
                    storage_key = storage_key_map[change_storage_key]

                    if change_data is not None:
                        change_scale_type = storage_key.value_scale_type
                        result_found = True
                    elif (
                        storage_key.metadata_storage_function.value["modifier"]
                        == "Default"
                    ):
                        # Fallback to default value of storage function if no result
                        change_scale_type = storage_key.value_scale_type
                        change_data = (
                            storage_key.metadata_storage_function.value_object[
                                "default"
                            ].value_object
                        )
                    else:
                        # No result is interpreted as an Option<...> result
                        change_scale_type = f"Option<{storage_key.value_scale_type}>"
                        change_data = (
                            storage_key.metadata_storage_function.value_object[
                                "default"
                            ].value_object
                        )

                    # Decode SCALE result data
                    updated_obj = self.decode_scale(
                        type_string=change_scale_type,
                        scale_bytes=hex_to_bytes(change_data),
                    )

                    subscription_result = subscription_handler(
                        storage_key, updated_obj, subscription_id
                    )

                    if subscription_result is not None:
                        # Handler returned end result: unsubscribe from further updates
                        self.rpc_request("state_unsubscribeStorage", [subscription_id])

            return result_found, subscription_result

        if not callable(subscription_handler):
            raise ValueError("Provided `subscription_handler` is not callable")

        return self.rpc_request(
            "state_subscribeStorage",
            [[s.to_hex() for s in storage_keys]],
            result_handler=result_handler,
        )

    def retrieve_pending_extrinsics(self) -> list:
        """
        Retrieves and decodes pending extrinsics from the node's transaction pool

        Returns:
            list of extrinsics
        """

        runtime = self.init_runtime()

        result_data = self.rpc_request("author_pendingExtrinsics", [])

        extrinsics = []

        for extrinsic_data in result_data["result"]:
            extrinsic = runtime.runtime_config.create_scale_object(
                "Extrinsic", metadata=runtime.metadata
            )
            extrinsic.decode(
                ScaleBytes(extrinsic_data),
                check_remaining=self.config.get("strict_scale_decode"),
            )
            extrinsics.append(extrinsic)

        return extrinsics

    def get_metadata_storage_functions(self, block_hash=None) -> list[dict[str, Any]]:
        """
        Retrieves a list of all storage functions in metadata active at given block_hash (or chaintip if block_hash is
        omitted)

        Args:
            block_hash: hash of the blockchain block whose runtime to use

        Returns:
            list of storage functions
        """
        runtime = self.init_runtime(block_hash=block_hash)
        return self._get_metadata_storage_functions(runtime=runtime)

    def get_metadata_storage_function(self, module_name, storage_name, block_hash=None):
        """
        Retrieves the details of a storage function for given module name, call function name and block_hash

        Args:
            module_name
            storage_name
            block_hash

        Returns:
            Metadata storage function
        """
        self.init_runtime(block_hash=block_hash)

        pallet = self.metadata.get_metadata_pallet(module_name)

        if pallet:
            return pallet.get_storage_function(storage_name)

    def get_metadata_errors(self, block_hash=None) -> list[dict[str, Optional[str]]]:
        """
        Retrieves a list of all errors in metadata active at given block_hash (or chaintip if block_hash is omitted)

        Args:
            block_hash: hash of the blockchain block whose metadata to use

        Returns:
            list of errors in the metadata
        """
        runtime = self.init_runtime(block_hash=block_hash)

        return self._get_metadata_errors(runtime=runtime)

    def get_metadata_error(
        self, module_name: str, error_name: str, block_hash=None
    ) -> Optional[scalecodec.GenericVariant]:
        """
        Retrieves the details of an error for given module name, call function name and block_hash

        Args:
        module_name: module name for the error lookup
        error_name: error name for the error lookup
        block_hash: hash of the blockchain block whose metadata to use

        Returns:
            error

        """
        runtime = self.init_runtime(block_hash=block_hash)
        return self._get_metadata_error(
            module_name=module_name, error_name=error_name, runtime=runtime
        )

    def get_metadata_runtime_call_functions(
        self, block_hash: Optional[str] = None
    ) -> list[scalecodec.GenericRuntimeCallDefinition]:
        """
        Get a list of available runtime API calls

        Returns:
            list of runtime call functions
        """
        runtime = self.init_runtime(block_hash=block_hash)
        return self._get_metadata_runtime_call_functions(runtime=runtime)

    def get_metadata_runtime_call_function(
        self, api: str, method: str, block_hash: Optional[str] = None
    ) -> scalecodec.GenericRuntimeCallDefinition:
        """
        Get details of a runtime API call

        Args:
            api: Name of the runtime API e.g. 'TransactionPaymentApi'
            method: Name of the method e.g. 'query_fee_details'
            block_hash: block hash whose metadata to query

        Returns:
            runtime call function
        """
        runtime = self.init_runtime(block_hash=block_hash)

        return self._get_metadata_runtime_call_function(api, method, runtime)

    def _get_block_handler(
        self,
        block_hash: str,
        ignore_decoding_errors: bool = False,
        include_author: bool = False,
        header_only: bool = False,
        finalized_only: bool = False,
        subscription_handler: Optional[Callable] = None,
    ):
        try:
            self.init_runtime(block_hash=block_hash)
        except BlockNotFound:
            return None

        def decode_block(block_data, block_data_hash=None) -> dict[str, Any]:
            if block_data:
                if block_data_hash:
                    block_data["header"]["hash"] = block_data_hash

                if isinstance(block_data["header"]["number"], str):
                    # Convert block number from hex (backwards compatibility)
                    block_data["header"]["number"] = int(
                        block_data["header"]["number"], 16
                    )

                extrinsic_cls = self.runtime_config.get_decoder_class("Extrinsic")

                if "extrinsics" in block_data:
                    for idx, extrinsic_data in enumerate(block_data["extrinsics"]):
                        try:
                            extrinsic_decoder = extrinsic_cls(
                                data=ScaleBytes(extrinsic_data),
                                metadata=self.runtime.metadata,
                                runtime_config=self.runtime_config,
                            )
                            extrinsic_decoder.decode(check_remaining=True)
                            block_data["extrinsics"][idx] = extrinsic_decoder

                        except Exception:
                            if not ignore_decoding_errors:
                                raise
                            block_data["extrinsics"][idx] = None

                for idx, log_data in enumerate(block_data["header"]["digest"]["logs"]):
                    if isinstance(log_data, str):
                        # Convert digest log from hex (backwards compatibility)
                        try:
                            log_digest_cls = self.runtime_config.get_decoder_class(
                                "sp_runtime::generic::digest::DigestItem"
                            )

                            if log_digest_cls is None:
                                raise NotImplementedError(
                                    "No decoding class found for 'DigestItem'"
                                )

                            log_digest = log_digest_cls(data=ScaleBytes(log_data))
                            log_digest.decode(
                                check_remaining=self.config.get("strict_scale_decode")
                            )

                            block_data["header"]["digest"]["logs"][idx] = log_digest

                            if include_author and "PreRuntime" in log_digest.value:
                                if self.runtime.implements_scaleinfo:
                                    engine = bytes(log_digest[1][0])
                                    # Retrieve validator set
                                    parent_hash = block_data["header"]["parentHash"]
                                    validator_set = self.query(
                                        "Session", "Validators", block_hash=parent_hash
                                    )

                                    if engine == b"BABE":
                                        babe_predigest = (
                                            self.runtime_config.create_scale_object(
                                                type_string="RawBabePreDigest",
                                                data=ScaleBytes(
                                                    bytes(log_digest[1][1])
                                                ),
                                            )
                                        )

                                        babe_predigest.decode(
                                            check_remaining=self.config.get(
                                                "strict_scale_decode"
                                            )
                                        )

                                        rank_validator = babe_predigest[1].value[
                                            "authority_index"
                                        ]

                                        block_author = validator_set[rank_validator]
                                        block_data["author"] = block_author

                                    elif engine == b"aura":
                                        aura_predigest = (
                                            self.runtime_config.create_scale_object(
                                                type_string="RawAuraPreDigest",
                                                data=ScaleBytes(
                                                    bytes(log_digest[1][1])
                                                ),
                                            )
                                        )

                                        aura_predigest.decode(check_remaining=True)

                                        rank_validator = aura_predigest.value[
                                            "slot_number"
                                        ] % len(validator_set)

                                        block_author = validator_set[rank_validator]
                                        block_data["author"] = block_author
                                    else:
                                        raise NotImplementedError(
                                            f"Cannot extract author for engine {log_digest.value['PreRuntime'][0]}"
                                        )
                                else:
                                    if (
                                        log_digest.value["PreRuntime"]["engine"]
                                        == "BABE"
                                    ):
                                        validator_set = self.query(
                                            "Session",
                                            "Validators",
                                            block_hash=block_hash,
                                        )
                                        rank_validator = log_digest.value["PreRuntime"][
                                            "data"
                                        ]["authority_index"]

                                        block_author = validator_set.elements[
                                            rank_validator
                                        ]
                                        block_data["author"] = block_author
                                    else:
                                        raise NotImplementedError(
                                            f"Cannot extract author for engine"
                                            f" {log_digest.value['PreRuntime']['engine']}"
                                        )

                        except Exception:
                            if not ignore_decoding_errors:
                                raise
                            block_data["header"]["digest"]["logs"][idx] = None

            return block_data

        if callable(subscription_handler):
            rpc_method_prefix = "Finalized" if finalized_only else "New"

            def result_handler(message: dict, subscription_id: str) -> tuple[Any, bool]:
                reached = False
                subscription_result = None
                if "params" in message:
                    new_block = decode_block({"header": message["params"]["result"]})

                    subscription_result = subscription_handler(new_block)

                    if subscription_result is not None:
                        reached = True
                        # Handler returned end result: unsubscribe from further updates
                        self.rpc_request(
                            f"chain_unsubscribe{rpc_method_prefix}Heads",
                            [subscription_id],
                        )

                return subscription_result, reached

            result = self._make_rpc_request(
                [
                    self.make_payload(
                        "_get_block_handler",
                        f"chain_subscribe{rpc_method_prefix}Heads",
                        [],
                    )
                ],
                result_handler=result_handler,
            )

            return result["_get_block_handler"][-1]

        else:
            if header_only:
                response = self.rpc_request("chain_getHeader", [block_hash])
                return decode_block(
                    {"header": response["result"]}, block_data_hash=block_hash
                )

            else:
                response = self.rpc_request("chain_getBlock", [block_hash])
                return decode_block(
                    response["result"]["block"], block_data_hash=block_hash
                )

    get_block_handler = _get_block_handler

    def get_block(
        self,
        block_hash: Optional[str] = None,
        block_number: Optional[int] = None,
        ignore_decoding_errors: bool = False,
        include_author: bool = False,
        finalized_only: bool = False,
    ) -> Optional[dict]:
        """
        Retrieves a block and decodes its containing extrinsics and log digest items. If `block_hash` and `block_number`
        is omitted the chain tip will be retrieved, or the finalized head if `finalized_only` is set to true.

        Either `block_hash` or `block_number` should be set, or both omitted.

        Args:
            block_hash: the hash of the block to be retrieved
            block_number: the block number to retrieved
            ignore_decoding_errors: When set this will catch all decoding errors, set the item to None and continue
                decoding
            include_author: This will retrieve the block author from the validator set and add to the result
            finalized_only: when no `block_hash` or `block_number` is set, this will retrieve the finalized head

        Returns:
            A dict containing the extrinsic and digest logs data
        """
        if block_hash and block_number:
            raise ValueError("Either block_hash or block_number should be set")

        if block_number is not None:
            block_hash = self.get_block_hash(block_number)

            if block_hash is None:
                return None

        if block_hash and finalized_only:
            raise ValueError(
                "finalized_only cannot be True when block_hash is provided"
            )

        if block_hash is None:
            # Retrieve block hash
            if finalized_only:
                block_hash = self.get_chain_finalised_head()
            else:
                block_hash = self.get_chain_head()

        return self._get_block_handler(
            block_hash=block_hash,
            ignore_decoding_errors=ignore_decoding_errors,
            header_only=False,
            include_author=include_author,
        )

    def get_block_header(
        self,
        block_hash: Optional[str] = None,
        block_number: Optional[int] = None,
        ignore_decoding_errors: bool = False,
        include_author: bool = False,
        finalized_only: bool = False,
    ) -> Optional[dict]:
        """
        Retrieves a block header and decodes its containing log digest items. If `block_hash` and `block_number`
        is omitted the chain tip will be retrieved, or the finalized head if `finalized_only` is set to true.

        Either `block_hash` or `block_number` should be set, or both omitted.

        See `get_block()` to also include the extrinsics in the result

        Args:
            block_hash: the hash of the block to be retrieved
            block_number: the block number to retrieved
            ignore_decoding_errors: When set this will catch all decoding errors, set the item to None and continue
                decoding
            include_author: This will retrieve the block author from the validator set and add to the result
            finalized_only: when no `block_hash` or `block_number` is set, this will retrieve the finalized head

        Returns:
            A dict containing the header and digest logs data
        """
        if block_hash and block_number:
            raise ValueError("Either block_hash or block_number should be be set")

        if block_number is not None:
            block_hash = self.get_block_hash(block_number)

            if block_hash is None:
                return None

        if block_hash and finalized_only:
            raise ValueError(
                "finalized_only cannot be True when block_hash is provided"
            )

        if block_hash is None:
            # Retrieve block hash
            if finalized_only:
                block_hash = self.get_chain_finalised_head()
            else:
                block_hash = self.get_chain_head()

        else:
            # Check conflicting scenarios
            if finalized_only:
                raise ValueError(
                    "finalized_only cannot be True when block_hash is provided"
                )

        return self._get_block_handler(
            block_hash=block_hash,
            ignore_decoding_errors=ignore_decoding_errors,
            header_only=True,
            include_author=include_author,
        )

    def subscribe_block_headers(
        self,
        subscription_handler: Callable,
        ignore_decoding_errors: bool = False,
        include_author: bool = False,
        finalized_only=False,
    ):
        """
        Subscribe to new block headers as soon as they are available. The callable `subscription_handler` will be
        executed when a new block is available and execution will block until `subscription_handler` will return
        a result other than `None`.

        Example:

        ```
        def subscription_handler(obj, update_nr, subscription_id):

            print(f"New block #{obj['header']['number']} produced by {obj['header']['author']}")

            if update_nr > 10
              return {'message': 'Subscription will cancel when a value is returned', 'updates_processed': update_nr}


        result = substrate.subscribe_block_headers(subscription_handler, include_author=True)
        ```

        Args:
            subscription_handler: the coroutine as explained above
            ignore_decoding_errors: When set this will catch all decoding errors, set the item to `None` and continue
                decoding
            include_author: This will retrieve the block author from the validator set and add to the result
            finalized_only: when no `block_hash` or `block_number` is set, this will retrieve the finalized head

        Returns:
            Value return by `subscription_handler`
        """
        # Retrieve block hash
        if finalized_only:
            block_hash = self.get_chain_finalised_head()
        else:
            block_hash = self.get_chain_head()

        return self._get_block_handler(
            block_hash,
            subscription_handler=subscription_handler,
            ignore_decoding_errors=ignore_decoding_errors,
            include_author=include_author,
            finalized_only=finalized_only,
        )

    def retrieve_extrinsic_by_identifier(
        self, extrinsic_identifier: str
    ) -> "ExtrinsicReceipt":
        """
        Retrieve an extrinsic by its identifier in format "[block_number]-[extrinsic_index]" e.g. 333456-4

        Args:
            extrinsic_identifier: "[block_number]-[extrinsic_idx]" e.g. 134324-2

        Returns:
            ExtrinsicReceiptLike object of the extrinsic
        """
        return ExtrinsicReceipt.create_from_extrinsic_identifier(
            substrate=self, extrinsic_identifier=extrinsic_identifier
        )

    def retrieve_extrinsic_by_hash(
        self, block_hash: str, extrinsic_hash: str
    ) -> "ExtrinsicReceipt":
        """
        Retrieve an extrinsic by providing the block_hash and the extrinsic hash

        Args:
            block_hash: hash of the blockchain block where the extrinsic is located
            extrinsic_hash: hash of the extrinsic

        Returns:
            ExtrinsicReceiptLike of the extrinsic
        """
        return ExtrinsicReceipt(
            substrate=self, block_hash=block_hash, extrinsic_hash=extrinsic_hash
        )

    def get_extrinsics(
        self, block_hash: str = None, block_number: Optional[int] = None
    ) -> Optional[list["ExtrinsicReceipt"]]:
        """
        Return all extrinsics for given block_hash or block_number

        Args:
            block_hash: hash of the blockchain block to retrieve extrinsics for
            block_number: block number to retrieve extrinsics for

        Returns:
            ExtrinsicReceipts of the extrinsics for the block, if any.
        """
        block = self.get_block(block_hash=block_hash, block_number=block_number)
        if block:
            return block["extrinsics"]

    def get_events(self, block_hash: Optional[str] = None) -> list[dict]:
        """
        Convenience method to get events for a certain block (storage call for module 'System' and function 'Events')

        Args:
            block_hash: the hash of the block to be retrieved

        Returns:
            list of events
        """
        if not block_hash:
            block_hash = self.get_chain_head()

        events = self.query(
            module="System",
            storage_function="Events",
            block_hash=block_hash,
        )
        return events.value

    def get_metadata(self, block_hash=None):
        """
        Returns `MetadataVersioned` object for given block_hash or chaintip if block_hash is omitted


        Args:
            block_hash

        Returns:
            MetadataVersioned
        """
        runtime = self.init_runtime(block_hash=block_hash)

        return runtime.metadata_v15

    @functools.lru_cache(maxsize=SUBSTRATE_CACHE_METHOD_SIZE)
    def get_parent_block_hash(self, block_hash):
        block_header = self.rpc_request("chain_getHeader", [block_hash])

        if block_header["result"] is None:
            raise SubstrateRequestException(f'Block not found for "{block_hash}"')
        parent_block_hash: str = block_header["result"]["parentHash"]

        if int(parent_block_hash, 16) == 0:
            # "0x0000000000000000000000000000000000000000000000000000000000000000"
            return block_hash
        return parent_block_hash

    def get_storage_by_key(self, block_hash: str, storage_key: str) -> Any:
        """
        A pass-though to existing JSONRPC method `state_getStorage`/`state_getStorageAt`

        Args:
            block_hash: hash of the block
            storage_key: storage key to query

        Returns:
            result of the query

        """

        if self.supports_rpc_method("state_getStorageAt"):
            response = self.rpc_request("state_getStorageAt", [storage_key, block_hash])
        else:
            response = self.rpc_request("state_getStorage", [storage_key, block_hash])

        if "result" in response:
            return response.get("result")
        else:
            raise SubstrateRequestException(
                "Unknown error occurred during retrieval of events"
            )

    @functools.lru_cache(maxsize=SUBSTRATE_RUNTIME_CACHE_SIZE)
    def get_block_runtime_info(self, block_hash: str) -> dict:
        """
        Retrieve the runtime info of given block_hash
        """
        response = self.rpc_request("state_getRuntimeVersion", [block_hash])
        return response.get("result")

    get_block_runtime_version = get_block_runtime_info

    @functools.lru_cache(maxsize=SUBSTRATE_CACHE_METHOD_SIZE)
    def get_block_runtime_version_for(self, block_hash: str):
        """
        Retrieve the runtime version of the parent of a given block_hash
        """
        parent_block_hash = self.get_parent_block_hash(block_hash)
        runtime_info = self.get_block_runtime_info(parent_block_hash)
        if runtime_info is None:
            return None
        return runtime_info["specVersion"]

    def get_block_metadata(
        self, block_hash: Optional[str] = None, decode: bool = True
    ) -> Optional[Union[dict, ScaleType]]:
        """
        A pass-though to existing JSONRPC method `state_getMetadata`.

        Args:
            block_hash: the hash of the block to be queried against
            decode: Whether to decode the metadata or present it raw

        Returns:
            metadata, either as a dict (not decoded) or ScaleType (decoded); None if there was no response
            from the server
        """
        params = None
        if decode and not self.runtime_config:
            raise ValueError(
                "Cannot decode runtime configuration without a supplied runtime_config"
            )

        if block_hash:
            params = [block_hash]
        response = self.rpc_request("state_getMetadata", params)

        if (result := response.get("result")) and decode:
            metadata_decoder = self.runtime_config.create_scale_object(
                "MetadataVersioned", data=ScaleBytes(result)
            )
            metadata_decoder.decode()

            return metadata_decoder
        else:
            return result

    def _preprocess(
        self,
        query_for: Optional[list],
        block_hash: Optional[str],
        storage_function: str,
        module: str,
        raw_storage_key: Optional[bytes] = None,
    ) -> Preprocessed:
        """
        Creates a Preprocessed data object for passing to `_make_rpc_request`
        """
        params = query_for if query_for else []
        # Search storage call in metadata
        metadata_pallet = self.runtime.metadata.get_metadata_pallet(module)

        if not metadata_pallet:
            raise SubstrateRequestException(f'Pallet "{module}" not found')

        storage_item = metadata_pallet.get_storage_function(storage_function)

        if not metadata_pallet or not storage_item:
            raise SubstrateRequestException(
                f'Storage function "{module}.{storage_function}" not found'
            )

        # SCALE type string of value
        param_types = storage_item.get_params_type_string()
        value_scale_type = storage_item.get_value_type_string()
        if len(params) != len(param_types):
            raise ValueError(
                f"Storage function requires {len(param_types)} parameters, {len(params)} given"
            )
        if raw_storage_key:
            storage_key = StorageKey.create_from_data(
                data=raw_storage_key,
                pallet=module,
                storage_function=storage_function,
                value_scale_type=value_scale_type,
                metadata=self.metadata,
                runtime_config=self.runtime_config,
            )
        else:
            storage_key = StorageKey.create_from_storage_function(
                module,
                storage_item.value["name"],
                params,
                runtime_config=self.runtime_config,
                metadata=self.runtime.metadata,
            )
        method = "state_getStorageAt"
        queryable = (
            str(query_for)
            if query_for is not None
            else f"{method}{random.randint(0, 7000)}"
        )
        return Preprocessed(
            queryable,
            method,
            [storage_key.to_hex(), block_hash],
            value_scale_type,
            storage_item,
        )

    def _process_response(
        self,
        response: dict,
        subscription_id: Union[int, str],
        value_scale_type: Optional[str] = None,
        storage_item: Optional[ScaleType] = None,
        result_handler: Optional[ResultHandler] = None,
        force_legacy_decode: bool = False,
    ) -> tuple[Any, bool]:
        """
        Processes the RPC call response by decoding it, returning it as is, or setting a handler for subscriptions,
        depending on the specific call.

        Args:
            response: the RPC call response
            subscription_id: the subscription id for subscriptions, used only for subscriptions with a result handler
            value_scale_type: Scale Type string used for decoding ScaleBytes results
            storage_item: The ScaleType object used for decoding ScaleBytes results
            result_handler: the result handler coroutine used for handling longer-running subscriptions
            force_legacy_decode: Whether to force legacy Metadata V14 decoding of the response

        Returns:
             (decoded response, completion)
        """
        result: Union[dict, ScaleType] = response
        if value_scale_type and isinstance(storage_item, ScaleType):
            if (response_result := response.get("result")) is not None:
                query_value = response_result
            elif storage_item.value["modifier"] == "Default":
                # Fallback to default value of storage function if no result
                query_value = storage_item.value_object["default"].value_object
            else:
                # No result is interpreted as an Option<...> result
                value_scale_type = f"Option<{value_scale_type}>"
                query_value = storage_item.value_object["default"].value_object
            if isinstance(query_value, str):
                q = bytes.fromhex(query_value[2:])
            elif isinstance(query_value, bytearray):
                q = bytes(query_value)
            else:
                q = query_value
            result = self.decode_scale(value_scale_type, q)
        if isinstance(result_handler, Callable):
            # For multipart responses as a result of subscriptions.
            message, bool_result = result_handler(result, subscription_id)
            return message, bool_result
        return result, True

    def _make_rpc_request(
        self,
        payloads: list[dict],
        value_scale_type: Optional[str] = None,
        storage_item: Optional[ScaleType] = None,
        result_handler: Optional[ResultHandler] = None,
        attempt: int = 1,
        force_legacy_decode: bool = False,
    ) -> RequestResults:
        request_manager = RequestManager(payloads)
        _received = {}

        if len(set(x["id"] for x in payloads)) != len(payloads):
            raise ValueError("Payloads must have unique ids")

        subscription_added = False

        ws = self.connect(init=False if attempt == 1 else True)
        for payload in payloads:
            item_id = get_next_id()
            to_send = json.dumps({**payload["payload"], **{"id": item_id}})
            if self.log_raw_websockets:
                raw_websocket_logger.debug(f"WEBSOCKET_SEND> {to_send}")
            ws.send(to_send)
            request_manager.add_request(item_id, payload["id"])
            # truncate to 2000 chars for debug logging
            if len(stringified_payload := str(payload)) < 2_000:
                output_payload = stringified_payload
            else:
                output_payload = f"{stringified_payload[:2_000]} (truncated)"
            logger.debug(
                f"Submitted payload ID {payload['id']} with websocket ID {item_id}: {output_payload}"
            )

        while True:
            try:
                recd = ws.recv(timeout=self.retry_timeout, decode=False)
                if self.log_raw_websockets:
                    raw_websocket_logger.debug(f"WEBSOCKET_RECEIVE> {recd.decode()}")
                response = json.loads(recd)
            except (TimeoutError, ConnectionClosed):
                if attempt >= self.max_retries:
                    logger.warning(
                        f"Timed out waiting for RPC requests {attempt} times. Exiting."
                    )
                    raise MaxRetriesExceeded("Max retries reached.")
                else:
                    return self._make_rpc_request(
                        payloads,
                        value_scale_type,
                        storage_item,
                        result_handler,
                        attempt + 1,
                        force_legacy_decode,
                    )
            if "id" in response:
                _received[response["id"]] = response
            elif "params" in response:
                _received[response["params"]["subscription"]] = response
            else:
                raise SubstrateRequestException(response)
            for item_id in request_manager.unresponded():
                if item_id not in request_manager.responses or isinstance(
                    result_handler, Callable
                ):
                    if response := _received.pop(item_id, None):
                        if (
                            isinstance(result_handler, Callable)
                            and not subscription_added
                        ):
                            # handles subscriptions, overwrites the previous mapping of {item_id : payload_id}
                            # with {subscription_id : payload_id}
                            try:
                                item_id = request_manager.overwrite_request(
                                    item_id, response["result"]
                                )
                                subscription_added = True
                            except KeyError:
                                raise SubstrateRequestException(str(response))
                                logger.error(
                                    f"Error received from subtensor for {item_id}: {response}\n"
                                    f"Currently received responses: {request_manager.get_results()}"
                                )
                        decoded_response, complete = self._process_response(
                            response,
                            item_id,
                            value_scale_type,
                            storage_item,
                            result_handler,
                            force_legacy_decode,
                        )
                        request_manager.add_response(
                            item_id, decoded_response, complete
                        )
                        # truncate to 2000 chars for debug logging
                        if len(stringified_response := str(decoded_response)) < 2_000:
                            output_response = stringified_response
                            # avoids clogging logs up needlessly (esp for Metadata stuff)
                        else:
                            output_response = (
                                f"{stringified_response[:2_000]} (truncated)"
                            )
                        logger.debug(
                            f"Received response for item ID {item_id}:\n{output_response}\n"
                            f"Complete: {complete}"
                        )

            if request_manager.is_complete:
                break

        return request_manager.get_results()

    @functools.lru_cache(maxsize=SUBSTRATE_CACHE_METHOD_SIZE)
    def supports_rpc_method(self, name: str) -> bool:
        """
        Check if substrate RPC supports given method
        Parameters
        ----------
        name: name of method to check

        Returns
        -------
        bool
        """
        result = self.rpc_request("rpc_methods", []).get("result")
        if result:
            self.config["rpc_methods"] = result.get("methods", [])

        return name in self.config["rpc_methods"]

    def rpc_request(
        self,
        method: str,
        params: Optional[list],
        result_handler: Optional[Callable] = None,
        block_hash: Optional[str] = None,
        reuse_block_hash: bool = False,
    ) -> Any:
        """
        Makes an RPC request to the subtensor. Use this only if `self.query` and `self.query_multiple` and
        `self.query_map` do not meet your needs.

        Args:
            method: str the method in the RPC request
            params: list of the params in the RPC request
            result_handler: Callback function that processes the result received from the node
            block_hash: the hash of the block — only supply this if not supplying the block
                hash in the params, and not reusing the block hash
            reuse_block_hash: whether to reuse the block hash in the params — only mark as True
                if not supplying the block hash in the params, or via the `block_hash` parameter

        Returns:
            the response from the RPC request
        """
        block_hash = self._get_current_block_hash(block_hash, reuse_block_hash)
        params = params or []
        payload_id = f"{method}{random.randint(0, 7000)}"
        payloads = [
            self.make_payload(
                payload_id,
                method,
                params + [block_hash] if block_hash else params,
            )
        ]
        result = self._make_rpc_request(payloads, result_handler=result_handler)
        if "error" in result[payload_id][0]:
            if "Failed to get runtime version" in (
                err_msg := result[payload_id][0]["error"]["message"]
            ):
                logger.warning(
                    "Failed to get runtime. Re-fetching from chain, and retrying."
                )
                self.init_runtime(block_hash=block_hash)
                return self.rpc_request(
                    method, params, result_handler, block_hash, reuse_block_hash
                )
            elif (
                "Client error: Api called for an unknown Block: State already discarded"
                in err_msg
            ):
                bh = err_msg.split("State already discarded for ")[1].strip()
                raise StateDiscardedError(bh)
            else:
                raise SubstrateRequestException(err_msg)
        if "result" in result[payload_id][0]:
            return result[payload_id][0]
        else:
            raise SubstrateRequestException(result[payload_id][0])

    def get_block_hash(self, block_id: Optional[int]) -> str:
        """
        Retrieves the block hash for a given block number, or the chaintip hash if None
        """
        if block_id is None:
            return self.get_chain_head()
        else:
            if (block_hash := self.runtime_cache.blocks.get(block_id)) is not None:
                return block_hash
            block_hash = self._get_block_hash(block_id)
            self.runtime_cache.add_item(block_hash=block_hash, block=block_id)
            return block_hash

    @functools.lru_cache(maxsize=SUBSTRATE_CACHE_METHOD_SIZE)
    def _get_block_hash(self, block_id: int) -> str:
        return self.rpc_request("chain_getBlockHash", [block_id])["result"]

    def get_chain_head(self) -> str:
        response = self._make_rpc_request(
            [
                self.make_payload(
                    "rpc_request",
                    "chain_getHead",
                    [],
                )
            ]
        )
        result = response["rpc_request"][0]
        if "error" in result:
            raise SubstrateRequestException(result["error"]["message"])
        self.last_block_hash = result["result"]
        return result["result"]

    def compose_call(
        self,
        call_module: str,
        call_function: str,
        call_params: Optional[dict] = None,
        block_hash: Optional[str] = None,
    ) -> GenericCall:
        """
        Composes a call payload which can be used in an extrinsic.

        Args:
            call_module: Name of the runtime module e.g. Balances
            call_function: Name of the call function e.g. transfer
            call_params: This is a dict containing the params of the call. e.g.
                `{'dest': 'EaG2CRhJWPb7qmdcJvy3LiWdh26Jreu9Dx6R1rXxPmYXoDk', 'value': 1000000000000}`
            block_hash: Use metadata at given block_hash to compose call

        Returns:
            A composed call
        """
        if call_params is None:
            call_params = {}

        self.init_runtime(block_hash=block_hash)

        call = self.runtime_config.create_scale_object(
            type_string="Call", metadata=self.runtime.metadata
        )

        call.encode(
            {
                "call_module": call_module,
                "call_function": call_function,
                "call_args": call_params,
            }
        )

        return call

    def query_multiple(
        self,
        params: list,
        storage_function: str,
        module: str,
        block_hash: Optional[str] = None,
        reuse_block_hash: bool = False,
    ) -> dict[str, ScaleType]:
        """
        Queries the subtensor. Only use this when making multiple queries, else use ``self.query``
        """
        block_hash = self._get_current_block_hash(block_hash, reuse_block_hash)
        if block_hash:
            self.last_block_hash = block_hash
        self.init_runtime(block_hash=block_hash)

        preprocessed: list[Preprocessed] = [
            self._preprocess([x], block_hash, storage_function, module) for x in params
        ]
        all_info = [
            self.make_payload(item.queryable, item.method, item.params)
            for item in preprocessed
        ]
        # These will always be the same throughout the preprocessed list, so we just grab the first one
        value_scale_type = preprocessed[0].value_scale_type
        storage_item = preprocessed[0].storage_item

        responses = self._make_rpc_request(all_info, value_scale_type, storage_item)
        return {
            param: responses[p.queryable][0] for (param, p) in zip(params, preprocessed)
        }

    def query_multi(
        self, storage_keys: list[StorageKey], block_hash: Optional[str] = None
    ) -> list[tuple[StorageKey, Any]]:
        """
        Query multiple storage keys in one request.

        Example:

        ```
        storage_keys = [
            substrate.create_storage_key(
                "System", "Account", ["F4xQKRUagnSGjFqafyhajLs94e7Vvzvr8ebwYJceKpr8R7T"]
            ),
            substrate.create_storage_key(
                "System", "Account", ["GSEX8kR4Kz5UZGhvRUCJG93D5hhTAoVZ5tAe6Zne7V42DSi"]
            )
        ]

        result = substrate.query_multi(storage_keys)
        ```

        Args:
            storage_keys: list of StorageKey objects
            block_hash: hash of the block to query against

        Returns:
            list of `(storage_key, scale_obj)` tuples
        """
        runtime = self.init_runtime(block_hash=block_hash)

        # Retrieve corresponding value
        response = self.rpc_request(
            "state_queryStorageAt", [[s.to_hex() for s in storage_keys], block_hash]
        )

        storage_key_map = {s.to_hex(): s for s in storage_keys}

        pairs = []
        for result_group in response["result"]:
            for change_storage_key, change_data in result_group["changes"]:
                storage_key = storage_key_map[change_storage_key]
                pairs.append(
                    (
                        storage_key,
                        ScaleBytes(change_data) if change_data is not None else None,
                    )
                )

        decoded = try_batch_decode(pairs, runtime)
        return list(zip([sk for sk, _ in pairs], decoded))

    def create_scale_object(
        self,
        type_string: str,
        data: Optional[ScaleBytes] = None,
        block_hash: Optional[str] = None,
        **kwargs,
    ) -> "ScaleType":
        """
        Convenience method to create a SCALE object of type `type_string`, this will initialize the runtime
        automatically at moment of `block_hash`, or chain tip if omitted.

        Args:
            type_string: Name of SCALE type to create
            data: ScaleBytes: ScaleBytes to decode
            block_hash: block hash for moment of decoding, when omitted the chain tip will be used
            kwargs: keyword args for the Scale Type constructor

        Returns:
             The created Scale Type object
        """
        self.init_runtime(block_hash=block_hash)

        if "metadata" not in kwargs:
            kwargs["metadata"] = self.runtime.metadata

        return self.runtime.runtime_config.create_scale_object(
            type_string, data=data, **kwargs
        )

    def generate_signature_payload(
        self,
        call: GenericCall,
        era=None,
        nonce: int = 0,
        tip: int = 0,
        tip_asset_id: Optional[int] = None,
        include_call_length: bool = False,
    ) -> ScaleBytes:
        # Retrieve genesis hash
        genesis_hash = self.get_block_hash(0)

        if not era:
            era = "00"

        if era == "00":
            # Immortal extrinsic
            block_hash = genesis_hash
        else:
            # Determine mortality of extrinsic
            era_obj = self.runtime_config.create_scale_object("Era")

            if isinstance(era, dict) and "current" not in era and "phase" not in era:
                raise ValueError(
                    'The era dict must contain either "current" or "phase" element to encode a valid era'
                )

            era_obj.encode(era)
            block_hash = self.get_block_hash(block_id=era_obj.birth(era.get("current")))

        # Create signature payload
        signature_payload = self.runtime_config.create_scale_object(
            "ExtrinsicPayloadValue"
        )

        # Process signed extensions in metadata
        if "signed_extensions" in self.runtime.metadata[1][1]["extrinsic"]:
            # Base signature payload
            signature_payload.type_mapping = [["call", "CallBytes"]]

            # Add signed extensions to payload
            signed_extensions = self.runtime.metadata.get_signed_extensions()

            if "CheckMortality" in signed_extensions:
                signature_payload.type_mapping.append(
                    ["era", signed_extensions["CheckMortality"]["extrinsic"]]
                )

            if "CheckEra" in signed_extensions:
                signature_payload.type_mapping.append(
                    ["era", signed_extensions["CheckEra"]["extrinsic"]]
                )

            if "CheckNonce" in signed_extensions:
                signature_payload.type_mapping.append(
                    ["nonce", signed_extensions["CheckNonce"]["extrinsic"]]
                )

            if "ChargeTransactionPayment" in signed_extensions:
                signature_payload.type_mapping.append(
                    ["tip", signed_extensions["ChargeTransactionPayment"]["extrinsic"]]
                )

            if "ChargeAssetTxPayment" in signed_extensions:
                signature_payload.type_mapping.append(
                    ["asset_id", signed_extensions["ChargeAssetTxPayment"]["extrinsic"]]
                )

            if "CheckMetadataHash" in signed_extensions:
                signature_payload.type_mapping.append(
                    ["mode", signed_extensions["CheckMetadataHash"]["extrinsic"]]
                )

            if "CheckSpecVersion" in signed_extensions:
                signature_payload.type_mapping.append(
                    [
                        "spec_version",
                        signed_extensions["CheckSpecVersion"]["additional_signed"],
                    ]
                )

            if "CheckTxVersion" in signed_extensions:
                signature_payload.type_mapping.append(
                    [
                        "transaction_version",
                        signed_extensions["CheckTxVersion"]["additional_signed"],
                    ]
                )

            if "CheckGenesis" in signed_extensions:
                signature_payload.type_mapping.append(
                    [
                        "genesis_hash",
                        signed_extensions["CheckGenesis"]["additional_signed"],
                    ]
                )

            if "CheckMortality" in signed_extensions:
                signature_payload.type_mapping.append(
                    [
                        "block_hash",
                        signed_extensions["CheckMortality"]["additional_signed"],
                    ]
                )

            if "CheckEra" in signed_extensions:
                signature_payload.type_mapping.append(
                    ["block_hash", signed_extensions["CheckEra"]["additional_signed"]]
                )

            if "CheckMetadataHash" in signed_extensions:
                signature_payload.type_mapping.append(
                    [
                        "metadata_hash",
                        signed_extensions["CheckMetadataHash"]["additional_signed"],
                    ]
                )

        if include_call_length:
            length_obj = self.runtime_config.create_scale_object("Bytes")
            call_data = str(length_obj.encode(str(call.data)))

        else:
            call_data = str(call.data)

        payload_dict = {
            "call": call_data,
            "era": era,
            "nonce": nonce,
            "tip": tip,
            "spec_version": self.runtime.runtime_version,
            "genesis_hash": genesis_hash,
            "block_hash": block_hash,
            "transaction_version": self.runtime.transaction_version,
            "asset_id": {"tip": tip, "asset_id": tip_asset_id},
            "metadata_hash": None,
            "mode": "Disabled",
        }

        signature_payload.encode(payload_dict)

        if signature_payload.data.length > 256:
            return ScaleBytes(
                data=blake2b(signature_payload.data.data, digest_size=32).digest()
            )

        return signature_payload.data

    def create_signed_extrinsic(
        self,
        call: GenericCall,
        keypair: Keypair,
        era: Optional[Union[dict, str]] = None,
        nonce: Optional[int] = None,
        tip: int = 0,
        tip_asset_id: Optional[int] = None,
        signature: Optional[Union[bytes, str]] = None,
    ) -> "GenericExtrinsic":
        """
        Creates an extrinsic signed by given account details

        Args:
            call: GenericCall to create extrinsic for
            keypair: Keypair used to sign the extrinsic
            era: Specify mortality in blocks in follow format:
                {'period': [amount_blocks]} If omitted the extrinsic is immortal
            nonce: nonce to include in extrinsics, if omitted the current nonce is retrieved on-chain
            tip: The tip for the block author to gain priority during network congestion
            tip_asset_id: Optional asset ID with which to pay the tip
            signature: Optionally provide signature if externally signed

        Returns:
             The signed Extrinsic
        """
        # only support creating extrinsics for current block
        self.init_runtime(block_id=self.get_block_number())

        # Check requirements
        if not isinstance(call, GenericCall):
            raise TypeError("'call' must be of type Call")

        # Check if extrinsic version is supported
        if self.runtime.metadata[1][1]["extrinsic"]["version"] != 4:  # type: ignore
            raise NotImplementedError(
                f"Extrinsic version {self.runtime.metadata[1][1]['extrinsic']['version']} not supported"  # type: ignore
            )

        # Retrieve nonce
        if nonce is None:
            nonce = self.get_account_nonce(keypair.ss58_address) or 0

        # Process era
        if era is None:
            era = "00"
        else:
            if isinstance(era, dict) and "current" not in era and "phase" not in era:
                # Retrieve current block id
                era["current"] = self.get_block_number(self.get_chain_finalised_head())

        if signature is not None:
            if isinstance(signature, str) and signature[0:2] == "0x":
                signature = bytes.fromhex(signature[2:])

            # Check if signature is a MultiSignature and contains signature version
            if len(signature) == 65:
                signature_version = signature[0]
                signature = signature[1:]
            else:
                signature_version = keypair.crypto_type

        else:
            # Create signature payload
            signature_payload = self.generate_signature_payload(
                call=call, era=era, nonce=nonce, tip=tip, tip_asset_id=tip_asset_id
            )

            # Set Signature version to crypto type of keypair
            signature_version = keypair.crypto_type

            # Sign payload
            signature = keypair.sign(signature_payload.data)

        # Create extrinsic
        extrinsic = self.runtime_config.create_scale_object(
            type_string="Extrinsic", metadata=self.runtime.metadata
        )

        value = {
            "account_id": f"0x{keypair.public_key.hex()}",
            "signature": f"0x{signature.hex()}",
            "call_function": call.value["call_function"],
            "call_module": call.value["call_module"],
            "call_args": call.value["call_args"],
            "nonce": nonce,
            "era": era,
            "tip": tip,
            "asset_id": {"tip": tip, "asset_id": tip_asset_id},
            "mode": "Disabled",
        }

        # Check if ExtrinsicSignature is MultiSignature, otherwise omit signature_version
        signature_cls = self.runtime_config.get_decoder_class("ExtrinsicSignature")
        if issubclass(signature_cls, self.runtime_config.get_decoder_class("Enum")):
            value["signature_version"] = signature_version

        extrinsic.encode(value)

        return extrinsic

    def create_unsigned_extrinsic(self, call: GenericCall) -> GenericExtrinsic:
        """
        Create unsigned extrinsic for given `Call`

        Args:
            call: GenericCall the call the extrinsic should contain

        Returns:
            GenericExtrinsic
        """

        runtime = self.init_runtime()

        # Create extrinsic
        extrinsic = self.runtime_config.create_scale_object(
            type_string="Extrinsic", metadata=runtime.metadata
        )

        extrinsic.encode(
            {
                "call_function": call.value["call_function"],
                "call_module": call.value["call_module"],
                "call_args": call.value["call_args"],
            }
        )

        return extrinsic

    def get_chain_finalised_head(self):
        """
        A pass-though to existing JSONRPC method `chain_getFinalizedHead`

        Returns
        -------

        """
        response = self.rpc_request("chain_getFinalizedHead", [])
        return response["result"]

    def _do_runtime_call_old(
        self,
        api: str,
        method: str,
        params: Optional[Union[list, dict]] = None,
        block_hash: Optional[str] = None,
    ) -> ScaleType[Any]:
        logger.debug(
            f"Decoding old runtime call: {api}.{method} with params: {params} at block hash: {block_hash}"
        )
        runtime_call_def = _TYPE_REGISTRY["runtime_api"][api]["methods"][method]

        # Encode params
        param_data: Union[ScaleBytes, bytes] = b""

        runtime = self.init_runtime(block_hash=block_hash)

        if "encoder" in runtime_call_def:
            param_data = runtime_call_def["encoder"](params, runtime)
            param_hex = param_data.hex()
        else:
            param_data = self._encode_scale_legacy(runtime_call_def, params, runtime)
            param_hex = param_data.to_hex()

        # RPC request
        result_data = self.rpc_request(
            "state_call", [f"{api}_{method}", param_hex, block_hash]
        )
        result_vec_u8_bytes = hex_to_bytes(result_data["result"])
        result_bytes = self.decode_scale("Vec<u8>", result_vec_u8_bytes)

        # Decode result
        # Get correct type
        if isinstance(result_bytes, str):
            raw_bytes = hex_to_bytes(result_bytes)
        else:
            raw_bytes = bytes(result_bytes)
        result_decoded = runtime_call_def["decoder"](raw_bytes, runtime)
        return result_decoded

    def runtime_call(
        self,
        api: str,
        method: str,
        params: Optional[Union[list, dict]] = None,
        block_hash: Optional[str] = None,
    ) -> ScaleType[Any]:
        """
        Calls a runtime API method

        Args:
            api: Name of the runtime API e.g. 'TransactionPaymentApi'
            method: Name of the method e.g. 'query_fee_details'
            params: List of parameters needed to call the runtime API
            block_hash: Hash of the block at which to make the runtime API call

        Returns:
             ScaleType from the runtime call
        """
        runtime = self.init_runtime(block_hash=block_hash)

        if params is None:
            params = {}

        try:
            if runtime.metadata_v15 is None:
                _ = self.runtime_config.type_registry["runtime_api"][api]["methods"][
                    method
                ]
                runtime_api_types = self.runtime_config.type_registry["runtime_api"][
                    api
                ].get("types", {})
                runtime.runtime_config.update_type_registry_types(runtime_api_types)
                return self._do_runtime_call_old(api, method, params, block_hash)
            else:
                runtime_call_def = runtime.runtime_api_map[api][method]
                if _determine_if_old_runtime_call(runtime_call_def, runtime):
                    return self._do_runtime_call_old(api, method, params, block_hash)

        except KeyError:
            raise ValueError(f"Runtime API Call '{api}.{method}' not found in registry")

        if isinstance(params, list) and len(params) != len(runtime_call_def["inputs"]):
            raise ValueError(
                f"Number of parameter provided ({len(params)}) does not "
                f"match definition {len(runtime_call_def['inputs'])}"
            )

        # Encode params
        param_data = b""
        for idx, param in enumerate(runtime_call_def["inputs"]):
            param_type_string = f"scale_info::{param['ty']}"
            if isinstance(params, list):
                param_data += self.encode_scale(
                    param_type_string, params[idx], runtime=runtime
                )
            else:
                if param["name"] not in params:
                    raise ValueError(f"Runtime Call param '{param['name']}' is missing")

                param_data += self.encode_scale(
                    param_type_string, params[param["name"]], runtime=runtime
                )

        # RPC request
        result_data = self.rpc_request(
            "state_call", [f"{api}_{method}", param_data.hex(), block_hash]
        )
        output_type_string = f"scale_info::{runtime_call_def['output']}"

        # Decode result
        result_bytes = hex_to_bytes(result_data["result"])
        return self.decode_scale(output_type_string, result_bytes)

    def get_account_nonce(self, account_address: str) -> int:
        """
        Returns current nonce for given account address

        Args:
            account_address: SS58 formatted address

        Returns:
            Nonce for given account address
        """
        if self.supports_rpc_method("state_call"):
            nonce_obj = self.runtime_call(
                "AccountNonceApi", "account_nonce", [account_address]
            )
            return getattr(nonce_obj, "value", nonce_obj)
        else:
            response = self.query(
                module="System", storage_function="Account", params=[account_address]
            )
            return response["nonce"]

    def get_account_next_index(self, account_address: str) -> int:
        """
        Returns next index for the given account address, taking into account the transaction pool.

        Args:
            account_address: SS58 formatted address

        Returns:
            Next index for the given account address
        """
        if not self.supports_rpc_method("account_nextIndex"):
            # Unlikely to happen, this is a common RPC method
            raise Exception("account_nextIndex not supported")

        nonce_obj = self.rpc_request("account_nextIndex", [account_address])
        return nonce_obj["result"]

    def get_metadata_constants(self, block_hash=None) -> list[dict]:
        """
        Retrieves a list of all constants in metadata active at given block_hash (or chaintip if block_hash is omitted)

        Args:
            block_hash: hash of the block

        Returns:
            list of constants
        """

        runtime = self.init_runtime(block_hash=block_hash)

        return self._get_metadata_constants(runtime)

    def get_metadata_constant(
        self, module_name, constant_name, block_hash=None
    ) -> Optional[scalecodec.ScaleInfoModuleConstantMetadata]:
        """
        Retrieves the details of a constant for given module name, call function name and block_hash
        (or chaintip if block_hash is omitted)

        Args:
            module_name: name of the module you are querying
            constant_name: name of the constant you are querying
            block_hash: hash of the block at which to make the runtime API call

        Returns:
            MetadataModuleConstants
        """
        runtime = self.init_runtime(block_hash=block_hash)
        return self._get_metadata_constant(module_name, constant_name, runtime)

    def get_constant(
        self,
        module_name: str,
        constant_name: str,
        block_hash: Optional[str] = None,
        reuse_block_hash: bool = False,
    ) -> Optional[ScaleType]:
        """
        Returns the decoded `ScaleType` object of the constant for given module name, call function name and block_hash
        (or chaintip if block_hash is omitted)

        Args:
            module_name: Name of the module to query
            constant_name: Name of the constant to query
            block_hash: Hash of the block at which to make the runtime API call
            reuse_block_hash: Reuse last-used block hash if set to true

        Returns:
             ScaleType from the runtime call
        """
        block_hash = self._get_current_block_hash(block_hash, reuse_block_hash)
        constant = self.get_metadata_constant(
            module_name, constant_name, block_hash=block_hash
        )
        if constant:
            # Decode to ScaleType
            return self.decode_scale(constant.type, bytes(constant.constant_value))
        else:
            return None

    def get_payment_info(
        self,
        call: GenericCall,
        keypair: Keypair,
        era: Optional[Union[dict, str]] = None,
        nonce: Optional[int] = None,
        tip: int = 0,
        tip_asset_id: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Retrieves fee estimation via RPC for given extrinsic

        Args:
            call: Call object to estimate fees for
            keypair: Keypair of the sender, does not have to include private key because no valid signature is
                     required
            era: Specify mortality in blocks in follow format:
                {'period': [amount_blocks]} If omitted the extrinsic is immortal
            nonce: nonce to include in extrinsics, if omitted the current nonce is retrieved on-chain
            tip: The tip for the block author to gain priority during network congestion
            tip_asset_id: Optional asset ID with which to pay the tip

        Returns:
            Dict with payment info
            E.g. `{'class': 'normal', 'partialFee': 151000000, 'weight': {'ref_time': 143322000}}`

        """

        # Check requirements
        if not isinstance(call, GenericCall):
            raise TypeError("'call' must be of type Call")

        if not isinstance(keypair, Keypair):
            raise TypeError("'keypair' must be of type Keypair")

        # No valid signature is required for fee estimation
        signature = "0x" + "00" * 64

        # Create extrinsic
        extrinsic = self.create_signed_extrinsic(
            call=call,
            keypair=keypair,
            era=era,
            nonce=nonce,
            tip=tip,
            tip_asset_id=tip_asset_id,
            signature=signature,
        )
        extrinsic_len = len(extrinsic.data)

        result = self.runtime_call(
            "TransactionPaymentApi", "query_info", [extrinsic, extrinsic_len]
        )

        return result.value

    def get_type_registry(
        self, block_hash: Optional[str] = None, max_recursion: int = 4
    ) -> dict:
        """
        Generates an exhaustive list of which RUST types exist in the runtime specified at given block_hash (or
        chaintip if block_hash is omitted)

        MetadataV14 or higher is required.

        Args:
            block_hash: Chaintip will be used if block_hash is omitted
            max_recursion: Increasing recursion will provide more detail but also has impact on performance

        Returns:
            dict mapping the type strings to the type decompositions
        """
        self.init_runtime(block_hash=block_hash)

        if not self.runtime.implements_scaleinfo:
            raise NotImplementedError("MetadataV14 or higher runtimes is required")

        type_registry = {}

        for scale_info_type in self.metadata.portable_registry["types"]:
            if (
                "path" in scale_info_type.value["type"]
                and len(scale_info_type.value["type"]["path"]) > 0
            ):
                type_string = "::".join(scale_info_type.value["type"]["path"])
            else:
                type_string = f"scale_info::{scale_info_type.value['id']}"

            scale_cls = self.runtime_config.get_decoder_class(type_string)
            type_registry[type_string] = scale_cls.generate_type_decomposition(
                max_recursion=max_recursion
            )

        return type_registry

    def get_type_definition(
        self, type_string: str, block_hash: Optional[str] = None
    ) -> str:
        """
        Retrieves SCALE encoding specifications of given type_string

        Args:
            type_string: RUST variable type, e.g. Vec<Address> or scale_info::0
            block_hash: hash of the blockchain block

        Returns:
            type decomposition
        """
        scale_obj = self.create_scale_object(type_string, block_hash=block_hash)
        return scale_obj.generate_type_decomposition()

    def get_metadata_modules(self, block_hash=None) -> list[dict[str, Any]]:
        """
        Retrieves a list of modules in metadata for given block_hash (or chaintip if block_hash is omitted)

        Args:
            block_hash: hash of the blockchain block

        Returns:
            List of metadata modules
        """
        runtime = self.init_runtime(block_hash=block_hash)
        return self._get_metadata_modules(runtime)

    def get_metadata_module(self, name, block_hash=None) -> ScaleType:
        """
        Retrieves modules in metadata by name for given block_hash (or chaintip if block_hash is omitted)

        Args:
            name: Name of the module
            block_hash: hash of the blockchain block

        Returns:
            MetadataModule
        """
        self.init_runtime(block_hash=block_hash)

        return self.metadata.get_metadata_pallet(name)

    def query(
        self,
        module: str,
        storage_function: str,
        params: Optional[list] = None,
        block_hash: Optional[str] = None,
        raw_storage_key: Optional[bytes] = None,
        subscription_handler=None,
        reuse_block_hash: bool = False,
    ) -> Optional[ScaleType[Any]]:
        """
        Queries substrate. This should only be used when making a single request. For multiple requests,
        you should use ``self.query_multiple``
        """
        block_hash = self._get_current_block_hash(block_hash, reuse_block_hash)
        if block_hash:
            self.last_block_hash = block_hash
        self.init_runtime(block_hash=block_hash)
        preprocessed: Preprocessed = self._preprocess(
            params, block_hash, storage_function, module, raw_storage_key
        )
        payload = [
            self.make_payload(
                preprocessed.queryable, preprocessed.method, preprocessed.params
            )
        ]
        value_scale_type = preprocessed.value_scale_type
        storage_item = preprocessed.storage_item

        responses = self._make_rpc_request(
            payload,
            value_scale_type,
            storage_item,
            result_handler=subscription_handler,
        )
        result = responses[preprocessed.queryable][0]
        return result

    def query_map(
        self,
        module: str,
        storage_function: str,
        params: Optional[list] = None,
        block_hash: Optional[str] = None,
        max_results: Optional[int] = None,
        start_key: Optional[str] = None,
        page_size: int = 100,
        ignore_decoding_errors: bool = False,
        reuse_block_hash: bool = False,
    ) -> QueryMapResult:
        """
        Iterates over all key-pairs located at the given module and storage_function. The storage
        item must be a map.

        Example:

        ```
        result = substrate.query_map('System', 'Account', max_results=100)

        for account, account_info in result:
            print(f"Free balance of account '{account.value}': {account_info.value['data']['free']}")
        ```

        Note: it is important that you do not use `for x in result.records`, as this will sidestep possible
        pagination. You must do `for x in result`.

        Args:
            module: The module name in the metadata, e.g. System or Balances.
            storage_function: The storage function name, e.g. Account or Locks.
            params: The input parameters in case of for example a `DoubleMap` storage function
            block_hash: Optional block hash for result at given block, when left to None the chain tip will be used.
            max_results: the maximum of results required, if set the query will stop fetching results when number is
                reached
            start_key: The storage key used as offset for the results, for pagination purposes
            page_size: The results are fetched from the node RPC in chunks of this size
            ignore_decoding_errors: When set this will catch all decoding errors, set the item to None and continue
                decoding
            reuse_block_hash: use True if you wish to make the query using the last-used block hash. Do not mark True
                              if supplying a block_hash

        Returns:
             QueryMapResult object
        """
        params = params or []
        block_hash = self._get_current_block_hash(block_hash, reuse_block_hash)
        if block_hash:
            self.last_block_hash = block_hash
        runtime = self.init_runtime(block_hash=block_hash)

        metadata_pallet = self.runtime.metadata.get_metadata_pallet(module)
        if not metadata_pallet:
            raise ValueError(f'Pallet "{module}" not found')
        storage_item = metadata_pallet.get_storage_function(storage_function)

        if not metadata_pallet or not storage_item:
            raise ValueError(
                f'Storage function "{module}.{storage_function}" not found'
            )

        value_type = storage_item.get_value_type_string()
        param_types = storage_item.get_params_type_string()
        key_hashers = storage_item.get_param_hashers()

        # Check MapType conditions
        if len(param_types) == 0:
            raise ValueError("Given storage function is not a map")
        if len(params) > len(param_types) - 1:
            raise ValueError(
                f"Storage function map can accept max {len(param_types) - 1} parameters, {len(params)} given"
            )

        # Generate storage key prefix
        # TODO should this use raw storage keys if necessary?
        storage_key = StorageKey.create_from_storage_function(
            module,
            storage_item.value["name"],
            params,
            runtime_config=self.runtime_config,
            metadata=self.runtime.metadata,
        )
        prefix = storage_key.to_hex()

        if not start_key:
            start_key = prefix

        # Make sure if the max result is smaller than the page size, adjust the page size
        if max_results is not None and max_results < page_size:
            page_size = max_results

        # Retrieve storage keys
        response = self.rpc_request(
            method="state_getKeysPaged",
            params=[prefix, page_size, start_key, block_hash],
        )

        result_keys = response.get("result")

        result = []
        last_key = None

        if len(result_keys) > 0:
            last_key = result_keys[-1]

            # Retrieve corresponding value
            response = self.rpc_request(
                method="state_queryStorageAt", params=[result_keys, block_hash]
            )

            for result_group in response["result"]:
                result = decode_query_map(
                    result_group["changes"],
                    prefix,
                    runtime,
                    param_types,
                    params,
                    value_type,
                    key_hashers,
                    ignore_decoding_errors,
                )
        return QueryMapResult(
            records=result,
            page_size=page_size,
            module=module,
            storage_function=storage_function,
            params=params,
            block_hash=block_hash,
            substrate=self,
            last_key=last_key,
            max_results=max_results,
            ignore_decoding_errors=ignore_decoding_errors,
        )

    def create_multisig_extrinsic(
        self,
        call: GenericCall,
        keypair: Keypair,
        multisig_account: MultiAccountId,
        max_weight: Optional[Union[dict, int]] = None,
        era: Optional[dict] = None,
        nonce: Optional[int] = None,
        tip: int = 0,
        tip_asset_id: Optional[int] = None,
        signature: Optional[Union[bytes, str]] = None,
    ) -> GenericExtrinsic:
        """
        Create a Multisig extrinsic that will be signed by one of the signatories. Checks on-chain if the threshold
        of the multisig account is reached and try to execute the call accordingly.

        Args:
            call: GenericCall to create extrinsic for
            keypair: Keypair of the signatory to approve given call
            multisig_account: MultiAccountId to use of origin of the extrinsic (see `generate_multisig_account()`)
            max_weight: Maximum allowed weight to execute the call ( Uses `get_payment_info()` by default)
            era: Specify mortality in blocks in follow format: {'period': [amount_blocks]} If omitted the extrinsic is
                immortal
            nonce: nonce to include in extrinsics, if omitted the current nonce is retrieved on-chain
            tip: The tip for the block author to gain priority during network congestion
            tip_asset_id: Optional asset ID with which to pay the tip
            signature: Optionally provide signature if externally signed

        Returns:
            GenericExtrinsic
        """
        if max_weight is None:
            payment_info = self.get_payment_info(call, keypair)
            max_weight = payment_info["weight"]

        # Check if call has existing approvals
        multisig_details = self.query(
            "Multisig", "Multisigs", [multisig_account.value, call.call_hash]
        )

        if multisig_details.value:
            maybe_timepoint = multisig_details.value["when"]
        else:
            maybe_timepoint = None

        # Compose 'as_multi' when final, 'approve_as_multi' otherwise
        if (
            multisig_details.value
            and len(multisig_details.value["approvals"]) + 1
            == multisig_account.threshold
        ):
            multi_sig_call = self.compose_call(
                "Multisig",
                "as_multi",
                {
                    "other_signatories": [
                        s
                        for s in multisig_account.signatories
                        if s != f"0x{keypair.public_key.hex()}"
                    ],
                    "threshold": multisig_account.threshold,
                    "maybe_timepoint": maybe_timepoint,
                    "call": call,
                    "store_call": False,
                    "max_weight": max_weight,
                },
            )
        else:
            multi_sig_call = self.compose_call(
                "Multisig",
                "approve_as_multi",
                {
                    "other_signatories": [
                        s
                        for s in multisig_account.signatories
                        if s != f"0x{keypair.public_key.hex()}"
                    ],
                    "threshold": multisig_account.threshold,
                    "maybe_timepoint": maybe_timepoint,
                    "call_hash": call.call_hash,
                    "max_weight": max_weight,
                },
            )

        return self.create_signed_extrinsic(
            multi_sig_call,
            keypair,
            era=era,
            nonce=nonce,
            tip=tip,
            tip_asset_id=tip_asset_id,
            signature=signature,
        )

    def submit_extrinsic(
        self,
        extrinsic: GenericExtrinsic,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = False,
    ) -> "ExtrinsicReceipt":
        """
        Submit an extrinsic to the connected node, with the possibility to wait until the extrinsic is included
         in a block and/or the block is finalized. The receipt returned provided information about the block and
         triggered events

        Args:
            extrinsic: Extrinsic The extrinsic to be sent to the network
            wait_for_inclusion: wait until extrinsic is included in a block (only works for websocket connections)
            wait_for_finalization: wait until extrinsic is finalized (only works for websocket connections)

        Returns:
            ExtrinsicReceipt object of your submitted extrinsic
        """

        # Check requirements
        if not isinstance(extrinsic, GenericExtrinsic):
            raise TypeError("'extrinsic' must be of type Extrinsics")

        def result_handler(message: dict, subscription_id) -> tuple[dict, bool]:
            """
            Result handler function passed as an arg to _make_rpc_request as the result_handler
            to handle the results of the extrinsic rpc call, which are multipart, and require
            subscribing to the message

            Args:
                message: message received from the rpc call
                subscription_id: subscription id received from the initial rpc call for the subscription

            Returns:
                tuple containing the dict of the block info for the subscription, and bool for whether
                the subscription is completed.
            """
            # Check if extrinsic is included and finalized
            if "params" in message and isinstance(message["params"]["result"], dict):
                # Convert result enum to lower for backwards compatibility
                message_result = {
                    k.lower(): v for k, v in message["params"]["result"].items()
                }

                # check for any subscription indicators of failure
                failure_message = None
                if "usurped" in message_result:
                    failure_message = (
                        f"Subscription {subscription_id} usurped: {message_result}"
                    )
                if "retracted" in message_result:
                    failure_message = (
                        f"Subscription {subscription_id} retracted: {message_result}"
                    )
                if "finalitytimeout" in message_result:
                    failure_message = f"Subscription {subscription_id} finalityTimeout: {message_result}"
                if "dropped" in message_result:
                    failure_message = (
                        f"Subscription {subscription_id} dropped: {message_result}"
                    )
                if "invalid" in message_result:
                    failure_message = (
                        f"Subscription {subscription_id} invalid: {message_result}"
                    )

                if failure_message is not None:
                    self.rpc_request("author_unwatchExtrinsic", [subscription_id])
                    logger.error(failure_message)
                    raise SubstrateRequestException(failure_message)

                if "finalized" in message_result and wait_for_finalization:
                    # Created as a task because we don't actually care about the result
                    # TODO change this logic
                    self.rpc_request("author_unwatchExtrinsic", [subscription_id])
                    return {
                        "block_hash": message_result["finalized"],
                        "extrinsic_hash": "0x{}".format(extrinsic.extrinsic_hash.hex()),
                        "finalized": True,
                    }, True
                elif (
                    "inblock" in message_result
                    and wait_for_inclusion
                    and not wait_for_finalization
                ):
                    self.rpc_request("author_unwatchExtrinsic", [subscription_id])
                    return {
                        "block_hash": message_result["inblock"],
                        "extrinsic_hash": "0x{}".format(extrinsic.extrinsic_hash.hex()),
                        "finalized": False,
                    }, True

            elif "params" in message and message["params"].get("result") == "invalid":
                failure_message = f"Subscription {subscription_id} invalid: {message}"
                self.rpc_request("author_unwatchExtrinsic", [subscription_id])
                logger.error(failure_message)
                raise SubstrateRequestException(failure_message)

            return message, False

        if wait_for_inclusion or wait_for_finalization:
            responses = (
                self._make_rpc_request(
                    [
                        self.make_payload(
                            "rpc_request",
                            "author_submitAndWatchExtrinsic",
                            [str(extrinsic.data)],
                        )
                    ],
                    result_handler=result_handler,
                )
            )["rpc_request"]
            response = next(
                (r for r in responses if "block_hash" in r and "extrinsic_hash" in r),
                None,
            )

            if not response:
                raise SubstrateRequestException(responses)

            # Also, this will be a multipart response, so maybe should change to everything after the first response?
            # The following code implies this will be a single response after the initial subscription id.
            result = ExtrinsicReceipt(
                substrate=self,
                extrinsic_hash=response["extrinsic_hash"],
                block_hash=response["block_hash"],
                finalized=response["finalized"],
            )

        else:
            response = self.rpc_request("author_submitExtrinsic", [str(extrinsic.data)])

            if "result" not in response:
                raise SubstrateRequestException(response.get("error"))

            result = ExtrinsicReceipt(substrate=self, extrinsic_hash=response["result"])

        return result

    def get_metadata_call_functions(
        self, block_hash: Optional[str] = None
    ) -> dict[str, dict[str, dict[str, dict[str, Union[str, int, list]]]]]:
        """
        Retrieves calls functions for the metadata at the specified block_hash. If not specified, the metadata at
        chaintip is used.

        Args:
            block_hash: block hash to retrieve metadata for

        Returns:
            dict mapping {pallet name: {call name: {param name: param definition}}}
            e.g.
            {
                "Sudo":{
                    "sudo": {
                        "_docs": "Authenticates the sudo key and dispatches a function call with `Root` origin.",
                        "call": {
                            "name": "call",
                            "type": 227,
                            "typeName": "Box<<T as Config>::RuntimeCall>",
                            "index": 0,
                            "_docs": ""
                        }
                    },
                    ...
                },
                ...
            }
        """
        runtime = self.init_runtime(block_hash=block_hash)
        return self._get_metadata_call_functions(runtime)

    def get_metadata_call_function(
        self,
        module_name: str,
        call_function_name: str,
        block_hash: Optional[str] = None,
    ) -> Optional[scalecodec.GenericVariant]:
        """
        Retrieves specified call from the metadata at the block specified, or the chain tip if omitted.

        Args:
            module_name: name of the module
            call_function_name: name of the call function
            block_hash: optional block hash

        Returns:
            The dict-like call definition, if found. None otherwise.
        """
        runtime = self.init_runtime(block_hash=block_hash)

        return self._get_metadata_call_function(
            module_name, call_function_name, runtime
        )

    def get_metadata_events(self, block_hash=None) -> list[dict]:
        """
        Retrieves a list of all events in metadata active for given block_hash (or chaintip if block_hash is omitted)

        Args:
            block_hash

        Returns:
            list of module events
        """

        runtime = self.init_runtime(block_hash=block_hash)
        return self._get_metadata_events(runtime)

    def get_metadata_event(
        self, module_name: str, event_name: str, block_hash=None
    ) -> Optional[Any]:
        """
        Retrieves the details of an event for given module name, call function name and block_hash
        (or chaintip if block_hash is omitted)

        Args:
            module_name: name of the module to call
            event_name: name of the event
            block_hash: hash of the block

        Returns:
            Metadata event

        """

        runtime = self.init_runtime(block_hash=block_hash)
        return self._get_metadata_event(module_name, event_name, runtime)

    def get_block_number(self, block_hash: Optional[str] = None) -> int:
        """
        Retrieves the block number for a given block hash or chaintip.
        """
        if block_hash is None:
            return self._get_block_number(None)
        else:
            if (
                block_number := self.runtime_cache.blocks_reverse.get(block_hash)
            ) is not None:
                return block_number
            block_number = self._cached_get_block_number(block_hash=block_hash)
            self.runtime_cache.add_item(block_hash=block_hash, block=block_number)
            return block_number

    @functools.lru_cache(maxsize=SUBSTRATE_CACHE_METHOD_SIZE)
    def _cached_get_block_number(self, block_hash: Optional[str]) -> int:
        return self._get_block_number(block_hash=block_hash)

    def _get_block_number(self, block_hash: Optional[str]) -> int:
        response = self.rpc_request("chain_getHeader", [block_hash])
        return int(response["result"]["number"], 16)

    def close(self):
        """
        Closes the substrate connection, and the websocket connection.
        """
        try:
            self.ws.close()
        except AttributeError:
            pass
        # Clear lru_cache on instance methods to allow garbage collection
        self.get_runtime_for_version.cache_clear()
        self.get_parent_block_hash.cache_clear()
        self.get_block_runtime_info.cache_clear()
        self.get_block_runtime_version_for.cache_clear()
        self.supports_rpc_method.cache_clear()
        self._get_block_hash.cache_clear()
        self._cached_get_block_number.cache_clear()

    encode_scale = SubstrateMixin._encode_scale
