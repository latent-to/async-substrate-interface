"""
A number of "plugins" for SubstrateInterface (and AsyncSubstrateInterface). At initial creation, it contains only
Retry (sync and async versions).
"""

import asyncio
import logging
import socket
from functools import partial
from itertools import cycle
from typing import Optional

from websockets.exceptions import ConnectionClosed

from async_substrate_interface.async_substrate import AsyncSubstrateInterface, Websocket
from async_substrate_interface.const import SS58_FORMAT
from async_substrate_interface.errors import MaxRetriesExceeded, StateDiscardedError
from async_substrate_interface.sync_substrate import SubstrateInterface

logger = logging.getLogger("async_substrate_interface")


RETRY_METHODS = [
    "_get_block_handler",
    "close",
    "compose_call",
    "create_scale_object",
    "create_signed_extrinsic",
    "create_storage_key",
    "decode_scale",
    "encode_scale",
    "generate_signature_payload",
    "get_account_next_index",
    "get_account_nonce",
    "get_block",
    "get_block_hash",
    "get_block_header",
    "get_block_metadata",
    "get_block_number",
    "get_block_runtime_info",
    "get_block_runtime_version_for",
    "get_chain_finalised_head",
    "get_chain_head",
    "get_constant",
    "get_events",
    "get_extrinsics",
    "get_metadata_call_function",
    "get_metadata_constant",
    "get_metadata_error",
    "get_metadata_errors",
    "get_metadata_module",
    "get_metadata_modules",
    "get_metadata_runtime_call_function",
    "get_metadata_runtime_call_functions",
    "get_metadata_storage_function",
    "get_metadata_storage_functions",
    "get_parent_block_hash",
    "get_payment_info",
    "get_storage_item",
    "get_type_definition",
    "get_type_registry",
    "init_runtime",
    "initialize",
    "query",
    "query_map",
    "query_multi",
    "retrieve_extrinsic_by_identifier",
    "rpc_request",
    "runtime_call",
    "submit_extrinsic",
    "subscribe_block_headers",
    "supports_rpc_method",
]

RETRY_PROPS = ["properties", "version", "token_decimals", "token_symbol", "name"]


class RetrySyncSubstrate(SubstrateInterface):
    """
    A subclass of SubstrateInterface that allows for handling chain failures by using backup chains. If a sustained
    network failure is encountered on a chain endpoint, the object will initialize a new connection on the next chain in
    the `fallback_chains` list. If the `retry_forever` flag is set, upon reaching the last chain in `fallback_chains`,
    the connection will attempt to iterate over the list (starting with `url`) again.

    E.g.
    ```
    substrate = RetrySyncSubstrate(
        "wss://entrypoint-finney.opentensor.ai:443",
        fallback_chains=["ws://127.0.0.1:9946"]
    )
    ```
    In this case, if there is a failure on entrypoint-finney, the connection will next attempt to hit localhost. If this
    also fails, a `MaxRetriesExceeded` exception will be raised.

    ```
    substrate = RetrySyncSubstrate(
        "wss://entrypoint-finney.opentensor.ai:443",
        fallback_chains=["ws://127.0.0.1:9946"],
        retry_forever=True
    )
    ```
    In this case, rather than a MaxRetriesExceeded exception being raised upon failure of the second chain (localhost),
    the object will again being to initialize a new connection on entrypoint-finney, and then localhost, and so on and
    so forth.
    """

    def __init__(
        self,
        url: str,
        *,
        use_remote_preset: bool = False,
        fallback_chains: Optional[list[str]] = None,
        retry_forever: bool = False,
        ss58_format: Optional[int] = SS58_FORMAT,
        type_registry: Optional[dict] = None,
        type_registry_preset: Optional[str] = None,
        chain_name: str = "",
        max_retries: int = 5,
        retry_timeout: float = 60.0,
        _mock: bool = False,
        _log_raw_websockets: bool = False,
        archive_nodes: Optional[list[str]] = None,
    ):
        fallback_chains = fallback_chains or []
        archive_nodes = archive_nodes or []
        self.fallback_chains = (
            iter(fallback_chains)
            if not retry_forever
            else cycle(fallback_chains + [url])
        )
        self.archive_nodes = (
            iter(archive_nodes) if not retry_forever else cycle(archive_nodes)
        )
        self.use_remote_preset = use_remote_preset
        self.chain_name = chain_name
        self._mock = _mock
        self.retry_timeout = retry_timeout
        self.max_retries = max_retries
        self.chain_endpoint = url
        self.url = url
        initialized = False
        for chain_url in [url] + fallback_chains:
            try:
                self.chain_endpoint = chain_url
                self.url = chain_url
                super().__init__(
                    url=chain_url,
                    ss58_format=ss58_format,
                    type_registry=type_registry,
                    use_remote_preset=use_remote_preset,
                    type_registry_preset=type_registry_preset,
                    chain_name=chain_name,
                    _mock=_mock,
                    retry_timeout=retry_timeout,
                    max_retries=max_retries,
                    _log_raw_websockets=_log_raw_websockets,
                )
                initialized = True
                logger.info(f"Connected to {chain_url}")
                break
            except ConnectionError:
                logger.warning(f"Unable to connect to {chain_url}")
        if not initialized:
            raise ConnectionError(
                f"Unable to connect at any chains specified: {[url] + fallback_chains}"
            )
        # "connect" is only used by SubstrateInterface, not AsyncSubstrateInterface
        retry_methods = ["connect"] + RETRY_METHODS
        self._original_methods = {
            method: getattr(self, method) for method in retry_methods
        }
        for method in retry_methods:
            setattr(self, method, partial(self._retry, method))

    def _retry(self, method_name, *args, **kwargs):
        method_ = self._original_methods[method_name]
        try:
            return method_(*args, **kwargs)
        except (
            MaxRetriesExceeded,
            ConnectionError,
            EOFError,
            ConnectionClosed,
            TimeoutError,
            socket.gaierror,
            StateDiscardedError,
        ) as e:
            use_archive = isinstance(e, StateDiscardedError)
            try:
                self._reinstantiate_substrate(e, use_archive=use_archive)
                return method_(*args, **kwargs)
            except StopIteration:
                logger.error(
                    f"Max retries exceeded with {self.url}. No more fallback chains."
                )
                raise MaxRetriesExceeded

    def _reinstantiate_substrate(
        self, e: Optional[Exception] = None, use_archive: bool = False
    ) -> None:
        if use_archive:
            bh = getattr(e, "block_hash", "Unknown Block Hash")
            logger.info(
                f"Attempt made to {bh} failed for state discarded. Attempting to switch to archive node."
            )
            next_network = next(self.archive_nodes)
        else:
            next_network = next(self.fallback_chains)
        self.ws.close()
        if isinstance(e, MaxRetriesExceeded):
            logger.error(
                f"Max retries exceeded with {self.url}. Retrying with {next_network}."
            )
        else:
            logger.error(f"Connection error. Trying again with {next_network}")
        self.url = next_network
        self.chain_endpoint = next_network
        self.initialized = False
        self.ws = self.connect(init=True)
        if not self._mock:
            self.initialize()


class RetryAsyncSubstrate(AsyncSubstrateInterface):
    """
    A subclass of AsyncSubstrateInterface that allows for handling chain failures by using backup chains. If a
    sustained network failure is encountered on a chain endpoint, the object will initialize a new connection on
    the next chain in the `fallback_chains` list. If the `retry_forever` flag is set, upon reaching the last chain
    in `fallback_chains`, the connection will attempt to iterate over the list (starting with `url`) again.

    E.g.
    ```
    substrate = RetryAsyncSubstrate(
        "wss://entrypoint-finney.opentensor.ai:443",
        fallback_chains=["ws://127.0.0.1:9946"]
    )
    ```
    In this case, if there is a failure on entrypoint-finney, the connection will next attempt to hit localhost. If this
    also fails, a `MaxRetriesExceeded` exception will be raised.

    ```
    substrate = RetryAsyncSubstrate(
        "wss://entrypoint-finney.opentensor.ai:443",
        fallback_chains=["ws://127.0.0.1:9946"],
        retry_forever=True
    )
    ```
    In this case, rather than a MaxRetriesExceeded exception being raised upon failure of the second chain (localhost),
    the object will again being to initialize a new connection on entrypoint-finney, and then localhost, and so on and
    so forth.
    """

    def __init__(
        self,
        url: str,
        *,
        use_remote_preset: bool = False,
        fallback_chains: Optional[list[str]] = None,
        retry_forever: bool = False,
        ss58_format: Optional[int] = SS58_FORMAT,
        type_registry: Optional[dict] = None,
        type_registry_preset: Optional[str] = None,
        chain_name: str = "",
        max_retries: int = 5,
        retry_timeout: float = 60.0,
        _mock: bool = False,
        _log_raw_websockets: bool = False,
        archive_nodes: Optional[list[str]] = None,
        ws_shutdown_timer: Optional[float] = 5.0,
    ):
        fallback_chains = fallback_chains or []
        archive_nodes = archive_nodes or []
        self.fallback_chains = (
            iter(fallback_chains)
            if not retry_forever
            else cycle(fallback_chains + [url])
        )
        self.archive_nodes = (
            iter(archive_nodes) if not retry_forever else cycle(archive_nodes)
        )
        self.use_remote_preset = use_remote_preset
        self.chain_name = chain_name
        self._mock = _mock
        self.retry_timeout = retry_timeout
        self.max_retries = max_retries
        super().__init__(
            url=url,
            ss58_format=ss58_format,
            type_registry=type_registry,
            use_remote_preset=use_remote_preset,
            type_registry_preset=type_registry_preset,
            chain_name=chain_name,
            _mock=_mock,
            retry_timeout=retry_timeout,
            max_retries=max_retries,
            _log_raw_websockets=_log_raw_websockets,
            ws_shutdown_timer=ws_shutdown_timer,
        )
        self._original_methods = {
            method: getattr(self, method) for method in RETRY_METHODS
        }
        for method in RETRY_METHODS:
            setattr(self, method, partial(self._retry, method))

    async def _reinstantiate_substrate(
        self, e: Optional[Exception] = None, use_archive: bool = False
    ) -> None:
        if use_archive:
            bh = getattr(e, "block_hash", "Unknown Block Hash")
            logger.info(
                f"Attempt made to {bh} failed for state discarded. Attempting to switch to archive node."
            )
            next_network = next(self.archive_nodes)
        else:
            next_network = next(self.fallback_chains)
        if isinstance(e, MaxRetriesExceeded):
            logger.error(
                f"Max retries exceeded with {self.url}. Retrying with {next_network}."
            )
        else:
            logger.error(f"Connection error. Trying again with {next_network}")
        if (
            self.startup_runtime_task is not None
            and not self.startup_runtime_task.done()
        ):
            self.startup_runtime_task.cancel()
            try:
                await self.startup_runtime_task
            except asyncio.CancelledError:
                pass
        try:
            await self.ws.shutdown()
        except AttributeError:
            pass
        _forgettable_task: asyncio.Task
        for _forgettable_task in self._forgettable_tasks:
            _forgettable_task.cancel()
            try:
                await _forgettable_task
            except asyncio.CancelledError:
                pass
        self.chain_endpoint = next_network
        self.url = next_network
        self.ws = Websocket(
            next_network,
            options={
                "max_size": self.ws_max_size,
                "write_limit": 2**16,
            },
        )
        self._initialized = False
        self._initializing = False
        await self.initialize()

    async def _retry(self, method_name, *args, **kwargs):
        method_ = self._original_methods[method_name]
        try:
            return await method_(*args, **kwargs)
        except (
            MaxRetriesExceeded,
            ConnectionError,
            EOFError,
            ConnectionClosed,
            TimeoutError,
            socket.gaierror,
            StateDiscardedError,
        ) as e:
            use_archive = isinstance(e, StateDiscardedError)
            try:
                await self._reinstantiate_substrate(e, use_archive=use_archive)
                return await method_(*args, **kwargs)
            except StopIteration:
                logger.error(
                    f"Max retries exceeded with {self.url}. No more fallback chains."
                )
                raise MaxRetriesExceeded
