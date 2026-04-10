"""
This library comprises the asyncio-compatible version of the subtensor interface commands we use in bittensor, as
well as its helper functions and classes. The docstring for the `AsyncSubstrateInterface` class goes more in-depth in
regard to how to instantiate and use it.
"""

import asyncio
import inspect
import logging
import os
import socket
import ssl
import time
import warnings
from contextlib import suppress
from unittest.mock import AsyncMock
from hashlib import blake2b
from typing import (
    Optional,
    Any,
    Callable,
    Awaitable,
    cast,
)

import scalecodec
from scalecodec import ScaleBytes
from scalecodec.base import ScaleType, RuntimeConfigurationObject
from scalecodec.type_registry import load_type_registry_preset
from scalecodec.types import (
    GenericCall,
    GenericExtrinsic,
    MultiAccountId,
)
from websockets import CloseCode
from websockets.asyncio.client import connect, ClientConnection
import websockets.exceptions
from websockets.exceptions import (
    ConnectionClosed,
    InvalidURI,
)
from websockets.protocol import State

from async_substrate_interface.errors import (
    SubstrateRequestException,
    ExtrinsicNotFound,
    BlockNotFound,
    StateDiscardedError,
)
from async_substrate_interface.protocols import Keypair
from async_substrate_interface.types import (
    RequestManager,
    Runtime,
    RuntimeCache,
    SubstrateMixin,
    Preprocessed,
    RequestResults,
)
from async_substrate_interface.utils import (
    hex_to_bytes,
    json,
    get_next_id,
    rng as random,
)
from async_substrate_interface.utils.cache import (
    async_sql_lru_cache,
    cached_fetcher,
    AsyncSqliteDB,
)
from async_substrate_interface.utils.decoding import (
    _determine_if_old_runtime_call,
    scale_decode,
    decode_query_map,
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
from async_substrate_interface.types import (
    _decode_option_opaque_metadata,
    _decode_v15_metadata,
)
from async_substrate_interface.utils.storage import StorageKey
from async_substrate_interface.type_registry import _TYPE_REGISTRY

ResultHandler = Callable[[dict, Any], Awaitable[tuple[dict, bool]]]

logger = logging.getLogger("async_substrate_interface")
raw_websocket_logger = logging.getLogger("raw_websocket")

# env vars dictating the cache size of the cached methods
SUBSTRATE_CACHE_METHOD_SIZE = int(os.getenv("SUBSTRATE_CACHE_METHOD_SIZE", "512"))
SUBSTRATE_RUNTIME_CACHE_SIZE = int(os.getenv("SUBSTRATE_RUNTIME_CACHE_SIZE", "16"))
SSL_SESSION_TTL = int(os.getenv("SUBSTRATE_SSL_SESSION_TTL", "300"))


class AsyncExtrinsicReceipt:
    """
    Object containing information of submitted extrinsic. Block hash where extrinsic is included is required
    when retrieving triggered events or determine if extrinsic was successful
    """

    def __init__(
        self,
        substrate: "AsyncSubstrateInterface",
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

    def __str__(self):
        return (
            f"AsyncExtrinsicReceipt({self.extrinsic_hash}), "
            f"block_hash={self.block_hash}, block_number={self.block_number}), "
            f"finalized={self.finalized})"
        )

    def __repr__(self):
        return self.__str__()

    async def get_extrinsic_identifier(self) -> str:
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

            self.block_number = await self.substrate.get_block_number(self.block_hash)

            if self.block_number is None:
                raise ValueError(
                    "Cannot create extrinsic identifier: unknown block_hash"
                )

        return f"{self.block_number}-{await self.extrinsic_idx}"

    async def retrieve_extrinsic(self):
        if not self.block_hash:
            raise ValueError(
                "ExtrinsicReceipt can't retrieve events because it's unknown which block_hash it is "
                "included, manually set block_hash or use `wait_for_inclusion` when sending extrinsic"
            )
        # Determine extrinsic idx

        block = await self.substrate.get_block(block_hash=self.block_hash)

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
    async def extrinsic_idx(self) -> int:
        """
        Retrieves the index of this extrinsic in containing block

        Returns
        -------
        int
        """
        if self.__extrinsic_idx is None:
            await self.retrieve_extrinsic()
        return self.__extrinsic_idx

    @property
    async def triggered_events(self) -> list:
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

            if await self.extrinsic_idx is None:
                await self.retrieve_extrinsic()

            self.__triggered_events = []

            for event in await self.substrate.get_events(block_hash=self.block_hash):
                if event["extrinsic_idx"] == await self.extrinsic_idx:
                    self.__triggered_events.append(event)

        return self.__triggered_events

    @classmethod
    async def create_from_extrinsic_identifier(
        cls, substrate: "AsyncSubstrateInterface", extrinsic_identifier: str
    ) -> "AsyncExtrinsicReceipt":
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
        block_hash = await substrate.get_block_hash(block_number)

        return cls(
            substrate=substrate,
            block_hash=block_hash,
            block_number=block_number,
            extrinsic_idx=extrinsic_idx,
        )

    async def process_events(self):
        if await self.triggered_events:
            self.__total_fee_amount = 0

            events = await self.triggered_events
            self.__total_fee_amount, has_transaction_fee_paid_event = (
                extract_total_fee_amount(events)
            )

            # Process other events
            possible_success = False
            for event in events:
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
                        self.__error_message = await self._resolve_module_error_message(
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

    async def _resolve_module_error_message(
        self, module_index: int, error_index: int
    ) -> dict:
        if self.block_hash:
            runtime = await self.substrate.init_runtime(block_hash=self.block_hash)
        else:
            runtime = await self.substrate.init_runtime(block_id=self.block_number)

        module_error = runtime.metadata.get_module_error(
            module_index=module_index, error_index=error_index
        )
        return {
            "type": "Module",
            "name": module_error.name,
            "docs": module_error.docs,
        }

    @property
    async def is_success(self) -> bool:
        """
        Returns `True` if `ExtrinsicSuccess` event is triggered, `False` in case of `ExtrinsicFailed`
        In case of False `error_message` will contain more details about the error


        Returns
        -------
        bool
        """
        if self.__is_success is None:
            await self.process_events()

        return cast(bool, self.__is_success)

    @property
    async def error_message(self) -> Optional[dict]:
        """
        Returns the error message if the extrinsic failed in format e.g.:

        `{'type': 'System', 'name': 'BadOrigin', 'docs': 'Bad origin'}`

        Returns
        -------
        dict
        """
        if self.__error_message is None:
            if await self.is_success:
                return None
            await self.process_events()
        return self.__error_message

    @property
    async def weight(self) -> int | dict:
        """
        Contains the actual weight when executing this extrinsic

        Returns
        -------
        int (WeightV1) or dict (WeightV2)
        """
        if self.__weight is None:
            await self.process_events()
        return self.__weight

    @property
    async def total_fee_amount(self) -> int:
        """
        Contains the total fee costs deducted when executing this extrinsic. This includes fee for the validator
            (`Balances.Deposit` event) and the fee deposited for the treasury (`Treasury.Deposit` event)

        Returns
        -------
        int
        """
        if self.__total_fee_amount is None:
            await self.process_events()
        return cast(int, self.__total_fee_amount)

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


class AsyncQueryMapResult:
    def __init__(
        self,
        records: list,
        page_size: int,
        substrate: "AsyncSubstrateInterface",
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

    async def retrieve_next_page(self, start_key) -> list:
        result = await self.substrate.query_map(
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

    async def retrieve_all_records(self) -> list[Any]:
        """
        Retrieves all records from all subsequent pages for the AsyncQueryMapResult,
        returning them as a list.

        Side effect:
            The self.records list will be populated fully after running this method.
        """
        async for _ in self:
            pass
        return self.records

    def __aiter__(self):
        return self

    def __iter__(self):
        return self

    async def get_next_record(self):
        try:
            # Try to get the next record from the buffer
            record = next(self._buffer)
        except StopIteration:
            # If no more records in the buffer
            return False, None
        else:
            return True, record

    async def __anext__(self):
        successfully_retrieved, record = await self.get_next_record()
        if successfully_retrieved:
            return record

        # If loading is already completed
        if self.loading_complete:
            raise StopAsyncIteration

        next_page = await self.retrieve_next_page(self.last_key)

        # If we cannot retrieve the next page
        if not next_page:
            self.loading_complete = True
            raise StopAsyncIteration

        self.records.extend(next_page)
        # Update the buffer with the newly fetched records
        self._buffer = iter(next_page)
        return next(self._buffer)

    def __getitem__(self, item):
        return self.records[item]


class _SessionResumingSSLContext(ssl.SSLContext):
    """
    An SSL context that saves the last TLS session and attempts to resume it on
    reconnection, as long as it is still within its TTL.

    Session resumption avoids a full TLS handshake on reconnect, reducing
    latency. The effective TTL is the minimum of ``session_ttl`` and the
    server-advertised session timeout.
    """

    def __new__(cls, protocol: int = ssl.PROTOCOL_TLS_CLIENT, **_kwargs):
        return ssl.SSLContext.__new__(cls, protocol)

    def __init__(
        self,
        protocol: int = ssl.PROTOCOL_TLS_CLIENT,
        *,
        session_ttl: int = SSL_SESSION_TTL,
    ):
        self._saved_session: Optional[ssl.SSLSession] = None
        self._session_established_at: Optional[float] = None
        self._session_ttl = session_ttl

    def save_session(self, session: ssl.SSLSession) -> None:
        self._saved_session = session
        self._session_established_at = time.monotonic()

    def _session_is_valid(self) -> bool:
        if self._saved_session is None or self._session_established_at is None:
            return False
        elapsed = time.monotonic() - self._session_established_at
        effective_ttl = min(self._session_ttl, self._saved_session.timeout)
        return elapsed < effective_ttl

    def wrap_bio(
        self, incoming, outgoing, server_side=False, server_hostname=None, session=None
    ):
        if not server_side and session is None and self._session_is_valid():
            session = self._saved_session
            logger.debug("Attempting TLS session resumption")
        return super().wrap_bio(
            incoming,
            outgoing,
            server_side=server_side,
            server_hostname=server_hostname,
            session=session,
        )


class Websocket:
    def __init__(
        self,
        ws_url: str,
        max_subscriptions: int = 1024,
        max_connections: int = 100,
        shutdown_timer: Optional[float] = 5.0,
        options: Optional[dict] = None,
        _log_raw_websockets: bool = False,
        retry_timeout: float = 60.0,
        max_retries: int = 5,
        ssl_context: Optional[_SessionResumingSSLContext] = None,
        dns_ttl: int = 300,
    ):
        """
        Websocket manager object. Allows for the use of a single websocket connection by multiple
        calls.

        Args:
            ws_url: Websocket URL to connect to
            max_subscriptions: Maximum number of subscriptions per websocket connection
            max_connections: Maximum number of connections total
            shutdown_timer: Number of seconds to shut down websocket connection after last use. If set to `None`, the
                connection will never be automatically shut down. Use this for very long-running processes, where you
                will manually shut down the connection if ever you intend to close it.
            options: Options to pass to the websocket connection
            _log_raw_websockets: Whether to log raw websockets in the "raw_websocket" logger
            retry_timeout: Timeout in seconds to retry websocket connection
            max_retries: Maximum number of retries following a timeout
            ssl_context: Optional session-resuming SSL context for wss:// connections.
                When provided, the context's saved TLS session is reused on reconnection
                to avoid a full handshake.
            dns_ttl: Seconds to cache DNS results. Set to 0 to disable caching.
        """
        # TODO allow setting max concurrent connections and rpc subscriptions per connection
        self.ws_url = ws_url
        self.ws: Optional[ClientConnection] = None
        self.max_subscriptions = asyncio.Semaphore(max_subscriptions)
        self.max_connections = max_connections
        self.shutdown_timer = shutdown_timer
        self.retry_timeout = retry_timeout
        self._received: dict[str, asyncio.Future] = {}
        self._received_subscriptions: dict[str, asyncio.Queue] = {}
        self._sending: Optional[asyncio.Queue] = None
        self._send_recv_task: Optional[asyncio.Task] = None
        self._inflight: dict[str, str] = {}
        self._attempts = 0
        self._lock = asyncio.Lock()
        self._exit_task = None
        self._options = options if options else {}
        self._log_raw_websockets = _log_raw_websockets
        self._in_use_ids = set()
        self._max_retries = max_retries
        self._last_activity = asyncio.Event()
        self._last_activity.set()
        self._waiting_for_response = 0
        self._ssl_context = ssl_context
        if ssl_context is not None and ws_url.startswith("wss://"):
            self._options["ssl"] = ssl_context
        self._dns_ttl = dns_ttl
        self._dns_cache: Optional[tuple[list, float]] = None

    @property
    def state(self):
        if self.ws is None:
            return State.CLOSED
        else:
            return self.ws.state

    async def __aenter__(self):
        if self.state not in (State.CONNECTING, State.OPEN):
            await self.connect()
        return self

    async def mark_waiting_for_response(self):
        """
        Mark that a response is expected. This will cause the websocket to not automatically close.

        Note: you must mark as response received once you have received the response.
        """
        async with self._lock:
            self._waiting_for_response += 1

    async def mark_response_received(self):
        """
        Mark that the expected response has been received. Automatic shutdown of websocket will proceed normally.

        Note: only do this if you have previously marked as waiting for response
        """
        async with self._lock:
            self._waiting_for_response -= 1

    @staticmethod
    async def loop_time() -> float:
        return asyncio.get_running_loop().time()

    async def _reset_activity_timer(self):
        """Reset the shared activity timeout"""
        # Create a NEW event instead of reusing the same one
        old_event = self._last_activity
        self._last_activity = asyncio.Event()
        self._last_activity.clear()  # Start fresh
        old_event.set()  # Wake up anyone waiting on the old event

    async def _wait_with_activity_timeout(self, coro, timeout: float):
        """
        Wait for a coroutine with a shared activity timeout.
        Returns the result or raises TimeoutError if no activity for timeout seconds.
        """
        activity_task = asyncio.create_task(self._last_activity.wait())

        if isinstance(coro, asyncio.Task):
            main_task = coro
        else:
            main_task = asyncio.create_task(coro)

        try:
            done, pending = await asyncio.wait(
                [main_task, activity_task],
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if not done:
                logger.debug(f"Activity timeout after {timeout}s, no activity detected")
                for task in pending:
                    task.cancel()
                raise TimeoutError()

            if main_task in done:
                activity_task.cancel()

                exc = main_task.exception()
                if exc is not None:
                    raise exc
                else:
                    return main_task.result()
            else:
                logger.debug("Activity detected, resetting timeout")
                return await self._wait_with_activity_timeout(main_task, timeout)

        except asyncio.CancelledError:
            main_task.cancel()
            activity_task.cancel()
            raise

    async def _cancel(self):
        try:
            logger.debug("Cancelling send/recv tasks")
            if self._send_recv_task is not None:
                self._send_recv_task.cancel()
                try:
                    await self._send_recv_task
                except asyncio.CancelledError:
                    pass
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(
                f"{e} encountered while trying to close websocket connection."
            )
        try:
            logger.debug("Closing websocket connection")
            if self.ws is not None:
                await self.ws.close()
        except Exception as e:
            logger.warning(
                f"{e} encountered while trying to close websocket connection."
            )

    async def _resolve_host(self) -> tuple:
        """
        Resolve the websocket hostname to a (family, type, proto, canonname, sockaddr) tuple,
        using a cached result if it is still within ``dns_ttl`` seconds.

        Invalidate the cache by setting ``_dns_cache = None`` before calling.
        """
        from urllib.parse import urlparse

        parsed = urlparse(self.ws_url)
        if parsed.scheme not in ("ws", "wss"):
            raise InvalidURI(self.ws_url, f"Invalid URI scheme: {parsed.scheme!r}")
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "wss" else 80)

        now = time.monotonic()
        if self._dns_cache is not None and self._dns_ttl > 0:
            infos, resolved_at = self._dns_cache
            if now - resolved_at < self._dns_ttl:
                logger.debug(f"DNS cache hit for {host} (age={now - resolved_at:.0f}s)")
                return infos[0]

        logger.debug(f"Resolving DNS for {host}:{port}")
        loop = asyncio.get_running_loop()
        infos = await loop.getaddrinfo(
            host, port, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM
        )
        self._dns_cache = (infos, now)
        logger.debug(f"DNS resolved {host} -> {infos[0][4][0]}")
        return infos[0]

    async def connect(self, force=False):
        if not force:
            async with self._lock:
                return await self._connect_internal(force)
        else:
            logger.debug("Proceeding without acquiring lock.")
            return await self._connect_internal(force)

    async def _connect_internal(self, force):
        # Check state again after acquiring lock to avoid duplicate connections
        if not force and self.state in (State.OPEN, State.CONNECTING):
            return None

        logger.debug(f"Websocket connecting to {self.ws_url}")
        if self._sending is None or self._sending.empty():
            self._sending = asyncio.Queue()
        if self._exit_task:
            self._exit_task.cancel()
        logger.debug(f"self.state={self.state}")
        if force and self.state == State.OPEN:
            logger.debug(f"Attempting to reconnect while already connected.")
            if self.ws is not None:
                self.ws.protocol.fail(CloseCode.SERVICE_RESTART)
            logger.debug(f"Open connection cancelled.")
            await asyncio.sleep(1)
        if self.state not in (State.OPEN, State.CONNECTING) or force:
            if not force:
                try:
                    logger.debug("Attempting cancellation")
                    await asyncio.wait_for(self._cancel(), timeout=10.0)
                except asyncio.TimeoutError:
                    logger.debug(f"Timed out waiting for cancellation")
                    pass
            logger.debug("Attempting connection")
            loop = asyncio.get_running_loop()
            try:
                family, type_, proto, _, sockaddr = await self._resolve_host()
                tcp_sock = socket.socket(family, type_, proto)
                tcp_sock.setblocking(False)
                try:
                    await asyncio.wait_for(
                        loop.sock_connect(tcp_sock, sockaddr), timeout=10.0
                    )
                except Exception:
                    tcp_sock.close()
                    self._dns_cache = None  # invalidate on TCP failure
                    raise
                connection = await asyncio.wait_for(
                    connect(self.ws_url, sock=tcp_sock, **self._options), timeout=10.0
                )
            except socket.gaierror:
                logger.debug(f"Hostname not known (this is just for testing")
                await asyncio.sleep(10)
                return await self.connect(force=force)
            logger.debug("Connection established")
            self.ws = connection
            if self._ssl_context is not None:
                try:
                    ssl_obj = connection.transport.get_extra_info("ssl_object")
                    if ssl_obj is not None and ssl_obj.session is not None:
                        self._ssl_context.save_session(ssl_obj.session)
                        logger.debug(
                            f"Saved TLS session "
                            f"(reused={ssl_obj.session_reused}, "
                            f"timeout={ssl_obj.session.timeout}s)"
                        )
                except Exception as e:
                    logger.debug(f"Could not save TLS session: {e}")
            if self._send_recv_task is None or self._send_recv_task.done():
                self._send_recv_task = loop.create_task(self._handler(self.ws))
        return None

    async def _handler(self, ws: ClientConnection) -> Optional[Exception]:
        logger.debug("WS handler attached")
        recv_task = asyncio.create_task(self._start_receiving(ws))
        send_task = asyncio.create_task(self._start_sending(ws))
        try:
            done, pending = await asyncio.wait(
                [recv_task, send_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
        except asyncio.CancelledError:
            # Handler was cancelled, clean up child tasks
            for task in [recv_task, send_task]:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            raise
        loop = asyncio.get_running_loop()
        should_reconnect = False
        is_retry = False

        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        for task in done:
            task_res = task.result()

            # If ConnectionClosedOK, graceful shutdown - don't reconnect
            if (
                isinstance(task_res, websockets.exceptions.ConnectionClosedOK)
                and self._waiting_for_response <= 0
            ):
                logger.debug("Graceful shutdown detected, not reconnecting")
                return None  # Clean exit

            # Check for timeout/connection errors that should trigger reconnect
            if isinstance(
                task_res, (asyncio.TimeoutError, TimeoutError, ConnectionClosed)
            ):
                should_reconnect = True
                logger.debug(f"Reconnection triggered by: {type(task_res).__name__}")

            if isinstance(task_res, (asyncio.TimeoutError, TimeoutError)):
                self._attempts += 1
                is_retry = True

        if should_reconnect is True:
            if len(self._received_subscriptions) > 0:
                return SubstrateRequestException(
                    f"Unable to reconnect because there are currently open subscriptions."
                )

            if is_retry:
                if self._attempts >= self._max_retries:
                    logger.error("Max retries exceeded.")
                    return TimeoutError("Max retries exceeded.")
                logger.info(
                    f"Timeout occurred. Reconnecting. Attempt {self._attempts} of {self._max_retries}"
                )

            async with self._lock:
                for original_id in list(self._inflight.keys()):
                    payload = self._inflight.pop(original_id)
                    self._received[original_id] = loop.create_future()
                    to_send = json.loads(payload)
                    logger.debug(f"Resubmitting {to_send['id']}")
                    await self._sending.put(to_send)

            logger.debug("Attempting reconnection...")
            await self.connect(True)
            logger.debug(f"Reconnected. Send queue size: {self._sending.qsize()}")
            # Recursively call handler
            return await self._handler(self.ws)
        elif isinstance(e := recv_task.result(), Exception):
            return e
        elif isinstance(e := send_task.result(), Exception):
            return e
        elif len(self._received_subscriptions) > 0:
            return SubstrateRequestException(
                f"Currently open subscriptions while disconnecting. "
                f"Ensure these are unsubscribed from before closing in the future."
            )
        return None

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.shutdown_timer is not None:
            if (
                self.state != State.CONNECTING
                and self._sending.qsize() == 0
                and not self._received_subscriptions
                and self._waiting_for_response <= 0
            ):
                if self._exit_task is not None:
                    self._exit_task.cancel()
                    try:
                        await self._exit_task
                    except asyncio.CancelledError:
                        pass
                if self.ws is not None:
                    self._exit_task = asyncio.create_task(self._exit_with_timer())
        self._attempts = 0

    async def _exit_with_timer(self):
        """
        Allows for graceful shutdown of websocket connection after specified number of seconds, allowing
        for reuse of the websocket connection.
        """
        try:
            if self.shutdown_timer is not None:
                logger.debug("Exiting with timer")
                await asyncio.sleep(self.shutdown_timer)
            if (
                self.state != State.CONNECTING
                and self._sending.qsize() == 0
                and not self._received_subscriptions
                and self._waiting_for_response <= 0
            ):
                await self.shutdown()
        except asyncio.CancelledError:
            pass

    async def shutdown(self):
        logger.debug("Shutdown requested")
        # Cancel the exit timer task if it exists
        if self._exit_task is not None:
            self._exit_task.cancel()
            try:
                await self._exit_task
            except asyncio.CancelledError:
                pass
            self._exit_task = None
        try:
            await asyncio.wait_for(self._cancel(), timeout=10.0)
        except asyncio.TimeoutError:
            pass
        self.ws = None
        self._send_recv_task = None

    async def _recv(self, recd: bytes) -> None:
        if self._log_raw_websockets:
            raw_websocket_logger.debug(f"WEBSOCKET_RECEIVE> {recd.decode()}")
        response = json.loads(recd)
        if "id" in response:
            async with self._lock:
                inflight_item = self._inflight.pop(response["id"], None)
                if inflight_item is not None:
                    logger.debug(f"Popped {response['id']} from inflight")
                else:
                    logger.debug(
                        f"Received response for {response['id']} which is no longer inflight (likely reconnection)"
                    )
            if self._received.get(response["id"]) is not None:
                self._received[response["id"]].set_result(response)
            self._in_use_ids.discard(response["id"])
        elif "params" in response:
            sub_id = response["params"]["subscription"]
            if sub_id not in self._received_subscriptions:
                self._received_subscriptions[sub_id] = asyncio.Queue()
            await self._received_subscriptions[sub_id].put(response)
        else:
            raise KeyError(response)

    async def _start_receiving(self, ws: ClientConnection) -> Optional[Exception]:
        logger.debug("Starting receiving task")
        try:
            while True:
                try:
                    recd = await self._wait_with_activity_timeout(
                        ws.recv(decode=False), self.retry_timeout
                    )
                    await self._reset_activity_timer()
                    self._attempts = 0
                    await self._recv(recd)
                except TimeoutError:
                    if (
                        self._waiting_for_response <= 0
                        or self._sending.qsize() == 0
                        or len(self._inflight) == 0
                        or len(self._received_subscriptions) == 0
                    ):
                        # if there's nothing in a queue, we really have no reason to have this, so we continue to wait
                        continue
        except websockets.exceptions.ConnectionClosedOK as e:
            logger.debug("ConnectionClosedOK")
            return e
        except Exception as e:
            if isinstance(e, ssl.SSLError):
                e = ConnectionClosed
            if not isinstance(
                e, (asyncio.TimeoutError, TimeoutError, ConnectionClosed)
            ):
                logger.exception("Websocket receiving exception", exc_info=e)
                for fut in self._received.values():
                    if not fut.done():
                        fut.set_exception(e)
                        fut.cancel()
            else:
                logger.debug(f"Timeout/ConnectionClosed occurred.")
            return e

    async def _start_sending(self, ws) -> Exception:
        logger.debug("Starting sending task")
        to_send = None
        try:
            while True:
                logger.debug(f"_sending, {self._sending.qsize()}")
                to_send_ = await self._sending.get()
                logger.debug("Retrieved item from sending queue")
                self._sending.task_done()
                send_id = to_send_["id"]
                to_send = json.dumps(to_send_)
                async with self._lock:
                    self._inflight[send_id] = to_send
                if self._log_raw_websockets:
                    raw_websocket_logger.debug(f"WEBSOCKET_SEND> {to_send}")
                await ws.send(to_send)
                logger.debug("Sent to websocket")
                await self._reset_activity_timer()
        except Exception as e:
            if isinstance(e, ssl.SSLError):
                e = ConnectionClosed
            if not isinstance(
                e, (asyncio.TimeoutError, TimeoutError, ConnectionClosed)
            ):
                logger.exception(
                    f"Websocket sending exception; "
                    f"sending: {self._sending.qsize()}; "
                    f"waiting_for_response: {self._waiting_for_response}; "
                    f"inflight: {len(self._inflight)}; "
                    f"subscriptions: {len(self._received_subscriptions)};",
                    exc_info=e,
                )
                if to_send is not None:
                    to_send_ = json.loads(to_send)
                    if to_send_["id"] in self._received:
                        self._received[to_send_["id"]].set_exception(e)
                        self._received[to_send_["id"]].cancel()
                else:
                    for i in self._received.keys():
                        self._received[i].set_exception(e)
                        self._received[i].cancel()
            elif isinstance(e, websockets.exceptions.ConnectionClosedOK):
                logger.debug("Websocket connection closed.")
            else:
                logger.debug("Timeout occurred.")
            return e

    async def send(self, payload: dict) -> str:
        """
        Sends a payload to the websocket connection.

        Args:
            payload: payload, generate a payload with the AsyncSubstrateInterface.make_payload method

        Returns:
            id: the internal ID of the request (incremented int)
        """
        await self.max_subscriptions.acquire()
        async with self._lock:
            original_id = get_next_id()
            while original_id in self._in_use_ids:
                original_id = get_next_id()
            self._in_use_ids.add(original_id)
            self._received[original_id] = asyncio.get_running_loop().create_future()
        to_send = {**payload, **{"id": original_id}}
        await self._sending.put(to_send)
        return original_id

    async def unsubscribe(
        self, subscription_id: str, method: str = "author_unwatchExtrinsic"
    ) -> None:
        """
        Unwatches a watched extrinsic subscription.

        Args:
            subscription_id: the internal ID of the subscription (typically a hex string)
            method: Typically "author_unwatchExtrinsic" for extrinsics, but can have different unsubscribe
                methods for things like watching chain head ("chain_unsubscribeFinalizedHeads" or
                "chain_unsubscribeNewHeads")
        """
        async with self._lock:
            original_id = get_next_id()
            while original_id in self._in_use_ids:
                original_id = get_next_id()
            logger.debug(f"Unwatched extrinsic subscription {subscription_id}")
            self._received_subscriptions.pop(subscription_id, None)

        to_send = {
            "jsonrpc": "2.0",
            "id": original_id,
            "method": method,
            "params": [subscription_id],
        }
        await self._sending.put(to_send)

    async def retrieve(self, item_id: str) -> Optional[dict]:
        """
        Retrieves a single item from received responses dict queue

        Args:
            item_id: id of the item to retrieve

        Returns:
             retrieved item
        """
        item: Optional[asyncio.Future] = self._received.get(item_id)
        if item is not None:
            if item.done():
                self.max_subscriptions.release()
                res = item.result()
                del self._received[item_id]
                return res
        else:
            try:
                subscription = self._received_subscriptions[item_id].get_nowait()
                self._received_subscriptions[item_id].task_done()
                return subscription
            except asyncio.QueueEmpty:
                pass
            except KeyError:
                logger.debug(
                    f"Received item {item_id} not in received subscriptions. "
                    f"This indicates the response of the subscription was inflight when sending "
                    f"the unsubscribe request."
                )
        if self._send_recv_task is not None and self._send_recv_task.done():
            if not self._send_recv_task.cancelled():
                if isinstance((e := self._send_recv_task.exception()), Exception):
                    logger.exception(f"Websocket sending exception: {e}")
                    raise e
                elif isinstance((e := self._send_recv_task.result()), Exception):
                    logger.exception(f"Websocket sending exception: {e}")
                    raise e
        return None


class AsyncSubstrateInterface(SubstrateMixin):
    ws: "Websocket"

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
        ws_shutdown_timer: Optional[float] = 5.0,
        _ssl_context: Optional[_SessionResumingSSLContext] = None,
        dns_ttl: int = 300,
    ):
        """
        The asyncio-compatible version of the subtensor interface commands we use in bittensor. It is important to
        initialise this class asynchronously in an async context manager using `async with AsyncSubstrateInterface()`.
        Otherwise, some (most) methods will not work properly, and may raise exceptions.

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
            ws_shutdown_timer: how long after the last connection your websocket should close
            _ssl_context: optional session-resuming SSL context; used internally by
                DiskCachedAsyncSubstrateInterface to enable TLS session reuse.
            dns_ttl: seconds to cache DNS results for the websocket URL (default 300). Set to 0
                to disable caching.

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
        self._log_raw_websockets = _log_raw_websockets
        if not _mock:
            self.ws = Websocket(
                url,
                _log_raw_websockets=_log_raw_websockets,
                options={
                    "max_size": self.ws_max_size,
                    "write_limit": 2**16,
                },
                shutdown_timer=ws_shutdown_timer,
                retry_timeout=self.retry_timeout,
                max_retries=max_retries,
                ssl_context=_ssl_context,
                dns_ttl=dns_ttl,
            )
        else:
            self.ws = AsyncMock(spec=Websocket)

        self._lock = asyncio.Lock()
        self.config = {
            "use_remote_preset": use_remote_preset,
            "auto_discover": auto_discover,
            "rpc_methods": None,
            "strict_scale_decode": True,
        }
        self.initialized = False
        self._forgettable_tasks = set()
        self.type_registry = type_registry
        self.type_registry_preset = type_registry_preset
        self.runtime_cache = RuntimeCache()
        self._nonces = {}
        self.metadata_version_hex = "0x0f000000"  # v15
        self._initializing = False
        self._mock = _mock
        self.startup_runtime_task: Optional[asyncio.Task] = None
        self.startup_block_hash: Optional[str] = None

    async def __aenter__(self):
        if not self._mock:
            await self.initialize()
        return self

    async def initialize(self) -> None:
        await self._initialize()

    async def _initialize(self) -> None:
        """
        Initialize the connection to the chain.
        """
        self._initializing = True
        if not self.initialized:
            await self.ws.connect()
            if not self._chain:
                chain = await self.rpc_request("system_chain", [])
                self._chain = chain.get("result")
            self.startup_block_hash = block_hash = await self.get_chain_head()
            self.startup_runtime_task = asyncio.create_task(
                self.init_runtime(block_hash=block_hash, init=True)
            )
            if self.ss58_format is None:
                runtime = await self.init_runtime(block_hash)
                # Check and apply runtime constants
                ss58_prefix_constant = await self.get_constant(
                    "System", "SS58Prefix", runtime=runtime
                )

                if ss58_prefix_constant:
                    self.ss58_format = ss58_prefix_constant.value
                    runtime.ss58_format = ss58_prefix_constant.value
                    runtime.runtime_config.ss58_format = ss58_prefix_constant.value
        self.initialized = True
        self._initializing = False

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    @property
    def metadata(self):
        warnings.warn(
            "Calling AsyncSubstrateInterface.metadata is deprecated, as metadata is runtime-dependent, and it"
            "can be unclear which for runtime you seek the metadata. You should instead use the specific runtime's "
            "metadata. For now, the most recently used runtime will be given.",
            category=DeprecationWarning,
        )
        runtime = self.runtime_cache.last_used
        if not runtime or runtime.metadata is None:
            raise AttributeError(
                "Metadata not found. This generally indicates that the AsyncSubstrateInterface object "
                "is not properly async initialized."
            )
        else:
            return runtime.metadata

    @property
    def implements_scaleinfo(self) -> Optional[bool]:
        """
        Returns True if most-recently-used runtime implements a `PortableRegistry` (`MetadataV14` and higher). Returns
        `None` if no runtime has been loaded.
        """
        runtime = self.runtime_cache.last_used
        if runtime is not None:
            return runtime.implements_scaleinfo
        else:
            return None

    @property
    async def properties(self):
        if self._properties is None:
            self._properties = (await self.rpc_request("system_properties", [])).get(
                "result"
            )
        return self._properties

    @property
    async def version(self):
        if self._version is None:
            self._version = (await self.rpc_request("system_version", [])).get("result")
        return self._version

    @property
    async def token_decimals(self):
        if self._token_decimals is None:
            self._token_decimals = (await self.properties).get("tokenDecimals")
        return self._token_decimals

    @property
    async def token_symbol(self):
        if self._token_symbol is None:
            if self.properties:
                self._token_symbol = (await self.properties).get("tokenSymbol")
            else:
                self._token_symbol = "UNIT"
        return self._token_symbol

    @property
    async def name(self):
        if self._name is None:
            self._name = (await self.rpc_request("system_name", [])).get("result")
        return self._name

    async def get_storage_item(
        self, module: str, storage_function: str, block_hash: Optional[str] = None
    ):
        runtime = await self.init_runtime(block_hash=block_hash)
        metadata_pallet = runtime.metadata.get_metadata_pallet(module)
        storage_item = metadata_pallet.get_storage_function(storage_function)
        return storage_item

    async def _get_current_block_hash(
        self, block_hash: Optional[str], reuse: bool
    ) -> Optional[str]:
        if block_hash:
            self.last_block_hash = block_hash
            return block_hash
        elif reuse:
            if self.last_block_hash:
                return self.last_block_hash
        return block_hash

    async def _load_registry_at_block(
        self, block_hash: Optional[str], runtime_config=None
    ):
        # Should be called for any block that fails decoding.
        # Possibly the metadata was different.
        try:
            metadata_rpc_result = await self.rpc_request(
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
        with open(f"/tmp/{block_hash}.bin", "wb+") as f:
            f.write(inner_bytes)
        if inner_bytes is None:
            return None
        return _decode_v15_metadata(inner_bytes, runtime_config=runtime_config)

    async def encode_scale(
        self,
        type_string,
        value: Any,
        block_hash: Optional[str] = None,
        runtime: Optional[Runtime] = None,
    ) -> bytes:
        """
        Helper function to encode arbitrary data into SCALE-bytes for given RUST type_string. If neither `block_hash`
        nor `runtime` are supplied, the runtime of the current block will be used.

        Args:
            type_string: the type string of the SCALE object for decoding
            value: value to encode
            block_hash: hash of the block where the desired runtime is located. Ignored if supplying `runtime`
            runtime: the runtime to use for the scale encoding. If supplied, `block_hash` is ignored

        Returns:
            encoded bytes
        """
        if runtime is None:
            runtime = await self.init_runtime(block_hash=block_hash)
        return self._encode_scale(type_string, value, runtime=runtime)

    async def decode_scale(
        self,
        type_string: str,
        scale_bytes: bytes,
        block_hash: Optional[str] = None,
        runtime: Optional[Runtime] = None,
    ) -> Optional[ScaleType[Any] | Any]:
        """
        Helper function to decode arbitrary SCALE-bytes (e.g. 0x02000000) according to given RUST type_string
        (e.g. BlockNumber). The relevant versioning information of the type (if defined) will be applied if block_hash
        is set

        Args:
            type_string: the type string of the SCALE object for decoding
            scale_bytes: the bytes representation of the SCALE object to decode
            block_hash: Hash of the block where the desired runtime is located. Ignored if supplying `runtime`
            runtime: Optional Runtime object whose registry to use for decoding. If not specified, runtime will be
                loaded based on the block hash specified (or latest block if no block_hash is specified)

        Returns:
            ScaleType object
        """
        if scale_bytes == b"":
            return None
        else:
            if runtime is None:
                runtime = await self.init_runtime(block_hash=block_hash)
            obj = scale_decode(type_string, scale_bytes, runtime=runtime)
            return obj

    async def init_runtime(
        self,
        block_hash: Optional[str] = None,
        block_id: Optional[int] = None,
        init: bool = False,
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
        if (
            not init
            and self.startup_runtime_task is not None
            and block_hash == self.startup_block_hash
        ):
            await self.startup_runtime_task
            self.startup_runtime_task = None

        if block_id and block_hash:
            raise ValueError("Cannot provide block_hash and block_id at the same time")

        if block_id is not None:
            if runtime := self.runtime_cache.retrieve(block=block_id):
                return runtime
            block_hash = await self.get_block_hash(block_id)

        if not block_hash:
            block_hash = await self.get_chain_head()
        else:
            self.last_block_hash = block_hash
            if runtime := self.runtime_cache.retrieve(block_hash=block_hash):
                return runtime

        runtime_version = await self.get_block_runtime_version_for(block_hash)

        if runtime_version is None:
            raise SubstrateRequestException(
                f"No runtime information for block '{block_hash}'"
            )

        if runtime := self.runtime_cache.retrieve(runtime_version=runtime_version):
            return runtime
        else:
            return await self.get_runtime_for_version(runtime_version, block_hash)

    @cached_fetcher(max_size=SUBSTRATE_RUNTIME_CACHE_SIZE, cache_key_index=0)
    async def get_runtime_for_version(
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
        return await self._get_runtime_for_version(runtime_version, block_hash)

    async def _get_runtime_for_version(
        self, runtime_version: int, block_hash: Optional[str] = None
    ) -> Runtime:
        runtime_config = RuntimeConfigurationObject(ss58_format=self.ss58_format)
        runtime_config.clear_type_registry()
        runtime_config.update_type_registry(load_type_registry_preset(name="core"))

        if not block_hash:
            block_hash, runtime_block_hash, block_number = await asyncio.gather(
                self.get_chain_head(),
                self.get_parent_block_hash(block_hash),
                self.get_block_number(block_hash),
            )
        else:
            runtime_block_hash, block_number = await asyncio.gather(
                self.get_parent_block_hash(block_hash),
                self.get_block_number(block_hash),
            )
        runtime_info, metadata_v15 = await asyncio.gather(
            self.get_block_runtime_info(runtime_block_hash),
            self._load_registry_at_block(
                block_hash=runtime_block_hash, runtime_config=runtime_config
            ),
        )
        if metadata_v15 is not None:
            # V15 is a superset of V14; use it directly, skipping the V14 fetch
            metadata = metadata_v15
            logger.debug(
                f"Retrieved metadata v15 for {runtime_version} from Substrate node"
            )
        else:
            metadata = await self.get_block_metadata(
                block_hash=runtime_block_hash, runtime_config=runtime_config
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
            metadata=metadata,
            type_registry=self.type_registry,
            runtime_config=runtime_config,
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

    async def create_storage_key(
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
        runtime = await self.init_runtime(block_hash=block_hash)
        params = params or []
        return StorageKey.create_from_storage_function(
            pallet,
            storage_function,
            params,
            runtime_config=runtime.runtime_config,
            metadata=runtime.metadata,
        )

    async def subscribe_storage(
        self,
        storage_keys: list[StorageKey],
        subscription_handler: Callable[[StorageKey, Any, str], Awaitable[Any]],
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
        async def subscription_handler(storage_key, obj, subscription_id):
            if obj is not None:
                # the subscription will run until your subscription_handler returns something other than `None`
                return obj
        ```

        Args:
            storage_keys: StorageKey list of storage keys to subscribe to
            subscription_handler: coroutine function to handle value changes of subscription

        """
        runtime = await self.init_runtime()

        storage_key_map = {s.to_hex(): s for s in storage_keys}

        async def result_handler(
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
                    updated_obj = await self.decode_scale(
                        type_string=change_scale_type,
                        scale_bytes=hex_to_bytes(change_data),
                        runtime=runtime,
                    )

                    subscription_result = await subscription_handler(
                        storage_key, updated_obj, subscription_id
                    )

                    if subscription_result is not None:
                        # Handler returned end result: unsubscribe from further updates
                        unsub_task = asyncio.create_task(
                            self.rpc_request(
                                "state_unsubscribeStorage", [subscription_id]
                            )
                        )
                        self._forgettable_tasks.add(unsub_task)
                        unsub_task.add_done_callback(self._forgettable_tasks.discard)

            return result_found, subscription_result

        if not callable(subscription_handler):
            raise ValueError("Provided `subscription_handler` is not callable")

        return await self.rpc_request(
            "state_subscribeStorage",
            [[s.to_hex() for s in storage_keys]],
            result_handler=result_handler,
        )

    async def retrieve_pending_extrinsics(self) -> list:
        """
        Retrieves and decodes pending extrinsics from the node's transaction pool

        Returns:
            list of extrinsics
        """

        runtime = await self.init_runtime()

        result_data = await self.rpc_request("author_pendingExtrinsics", [])
        if "error" in result_data:
            logger.error(
                f"Error in retrieving pending extrinsics: {result_data['error']}"
            )
            raise SubstrateRequestException(result_data["error"]["message"])
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

    async def get_metadata_storage_functions(
        self, block_hash: Optional[str] = None, runtime: Optional[Runtime] = None
    ) -> list[dict[str, Any]]:
        """
        Retrieves a list of all storage functions in metadata active at given block_hash (or chaintip if
        block_hash and runtime are omitted)

        Args:
            block_hash: hash of the blockchain block whose runtime to use
            runtime: Optional `Runtime` whose metadata to use

        Returns:
            list of storage functions
        """
        if runtime is None:
            runtime = await self.init_runtime(block_hash=block_hash)

        return self._get_metadata_storage_functions(runtime=runtime)

    async def get_metadata_storage_function(
        self,
        module_name,
        storage_name,
        block_hash=None,
        runtime: Optional[Runtime] = None,
    ):
        """
        Retrieves the details of a storage function for given module name, call function name and block_hash

        Args:
            module_name
            storage_name
            block_hash
            runtime: Optional `Runtime` whose metadata to use

        Returns:
            Metadata storage function
        """
        if runtime is None:
            runtime = await self.init_runtime(block_hash=block_hash)

        pallet = runtime.metadata.get_metadata_pallet(module_name)

        if pallet:
            return pallet.get_storage_function(storage_name)

    async def get_metadata_errors(
        self, block_hash=None, runtime: Optional[Runtime] = None
    ) -> list[dict[str, Optional[str]]]:
        """
        Retrieves a list of all errors in metadata active at given block_hash (or chaintip if block_hash is omitted)

        Args:
            block_hash: hash of the blockchain block whose metadata to use
            runtime: Optional `Runtime` whose metadata to use

        Returns:
            list of errors in the metadata
        """
        if runtime is None:
            runtime = await self.init_runtime(block_hash=block_hash)

        return self._get_metadata_errors(runtime=runtime)

    async def get_metadata_error(
        self,
        module_name: str,
        error_name: str,
        block_hash: Optional[str] = None,
        runtime: Optional[Runtime] = None,
    ) -> Optional[scalecodec.GenericVariant]:
        """
        Retrieves the details of an error for given module name, call function name and block_hash

        Args:
        module_name: module name for the error lookup
        error_name: error name for the error lookup
        block_hash: hash of the blockchain block whose metadata to use
        runtime: Optional `Runtime` whose metadata to use

        Returns:
            error

        """
        if runtime is None:
            runtime = await self.init_runtime(block_hash=block_hash)
        return self._get_metadata_error(
            module_name=module_name, error_name=error_name, runtime=runtime
        )

    async def get_metadata_runtime_call_functions(
        self, block_hash: Optional[str] = None, runtime: Optional[Runtime] = None
    ) -> list[scalecodec.GenericRuntimeCallDefinition]:
        """
        Get a list of available runtime API calls

        Returns:
            list of runtime call functions
        """
        if runtime is None:
            runtime = await self.init_runtime(block_hash=block_hash)
        return self._get_metadata_runtime_call_functions(runtime=runtime)

    async def get_metadata_runtime_call_function(
        self,
        api: str,
        method: str,
        block_hash: Optional[str] = None,
        runtime: Optional[Runtime] = None,
    ) -> scalecodec.GenericRuntimeCallDefinition:
        """
        Get details of a runtime API call. If not supplying `block_hash` or `runtime`, the runtime of the current block
        will be used.

        Args:
            api: Name of the runtime API e.g. 'TransactionPaymentApi'
            method: Name of the method e.g. 'query_fee_details'
            block_hash: Hash of the block whose runtime to use, if not specifying `runtime`
            runtime: The `Runtime` object whose metadata to use.

        Returns:
            GenericRuntimeCallDefinition
        """
        if runtime is None:
            runtime = await self.init_runtime(block_hash=block_hash)
        return self._get_metadata_runtime_call_function(api, method, runtime)

    async def _get_block_handler(
        self,
        block_hash: str,
        ignore_decoding_errors: bool = False,
        include_author: bool = False,
        header_only: bool = False,
        finalized_only: bool = False,
        subscription_handler: Optional[Callable[[dict], Awaitable[Any]]] = None,
    ):
        try:
            runtime = await self.init_runtime(block_hash=block_hash)
        except BlockNotFound:
            return None

        async def decode_block(block_data, block_data_hash=None) -> dict[str, Any]:
            if block_data:
                if block_data_hash:
                    block_data["header"]["hash"] = block_data_hash

                if isinstance(block_data["header"]["number"], str):
                    # Convert block number from hex (backwards compatibility)
                    block_data["header"]["number"] = int(
                        block_data["header"]["number"], 16
                    )

                extrinsic_cls = runtime.runtime_config.get_decoder_class("Extrinsic")

                if "extrinsics" in block_data:
                    for idx, extrinsic_data in enumerate(block_data["extrinsics"]):
                        try:
                            extrinsic_decoder = extrinsic_cls(
                                data=ScaleBytes(extrinsic_data),
                                metadata=runtime.metadata,
                                runtime_config=runtime.runtime_config,
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
                            log_digest_cls = runtime.runtime_config.get_decoder_class(
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
                                if runtime.implements_scaleinfo:
                                    engine = bytes(log_digest[1][0])
                                    # Retrieve validator set
                                    parent_hash = block_data["header"]["parentHash"]
                                    validator_set = await self.query(
                                        "Session",
                                        "Validators",
                                        block_hash=parent_hash,
                                        runtime=runtime,
                                    )

                                    if engine == b"BABE":
                                        babe_predigest = (
                                            runtime.runtime_config.create_scale_object(
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
                                            runtime.runtime_config.create_scale_object(
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
                                        validator_set = await self.query(
                                            "Session",
                                            "Validators",
                                            block_hash=block_hash,
                                            runtime=runtime,
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

            async def result_handler(
                message: dict, subscription_id: str
            ) -> tuple[Any, bool]:
                reached = False
                subscription_result = None
                if "params" in message:
                    new_block = await decode_block(
                        {"header": message["params"]["result"]}
                    )

                    subscription_result = await subscription_handler(new_block)

                    if subscription_result is not None:
                        reached = True
                        # Handler returned end result: unsubscribe from further updates
                        async with self.ws as ws:
                            await ws.unsubscribe(
                                subscription_id,
                                method=f"chain_unsubscribe{rpc_method_prefix}Heads",
                            )

                return subscription_result, reached

            result = await self._make_rpc_request(
                [
                    self.make_payload(
                        "_get_block_handler",
                        f"chain_subscribe{rpc_method_prefix}Heads",
                        [],
                    )
                ],
                result_handler=result_handler,
                runtime=runtime,
            )

            return result["_get_block_handler"][-1]

        else:
            if header_only:
                response = await self.rpc_request(
                    "chain_getHeader", [block_hash], runtime=runtime
                )
                return await decode_block(
                    {"header": response["result"]}, block_data_hash=block_hash
                )

            else:
                response = await self.rpc_request(
                    "chain_getBlock", [block_hash], runtime=runtime
                )
                return await decode_block(
                    response["result"]["block"], block_data_hash=block_hash
                )

    get_block_handler = _get_block_handler

    async def get_block(
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
            block_hash = await self.get_block_hash(block_number)

            if block_hash is None:
                return

        if block_hash and finalized_only:
            raise ValueError(
                "finalized_only cannot be True when block_hash is provided"
            )

        if block_hash is None:
            # Retrieve block hash
            if finalized_only:
                block_hash = await self.get_chain_finalised_head()
            else:
                block_hash = await self.get_chain_head()

        return await self._get_block_handler(
            block_hash=block_hash,
            ignore_decoding_errors=ignore_decoding_errors,
            header_only=False,
            include_author=include_author,
        )

    async def get_block_header(
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
            block_hash = await self.get_block_hash(block_number)

            if block_hash is None:
                return None

        if block_hash and finalized_only:
            raise ValueError(
                "finalized_only cannot be True when block_hash is provided"
            )

        if block_hash is None:
            # Retrieve block hash
            if finalized_only:
                block_hash = await self.get_chain_finalised_head()
            else:
                block_hash = await self.get_chain_head()

        else:
            # Check conflicting scenarios
            if finalized_only:
                raise ValueError(
                    "finalized_only cannot be True when block_hash is provided"
                )

        return await self._get_block_handler(
            block_hash=block_hash,
            ignore_decoding_errors=ignore_decoding_errors,
            header_only=True,
            include_author=include_author,
        )

    async def subscribe_block_headers(
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
        async def subscription_handler(obj, update_nr, subscription_id):

            print(f"New block #{obj['header']['number']} produced by {obj['header']['author']}")

            if update_nr > 10
              return {'message': 'Subscription will cancel when a value is returned', 'updates_processed': update_nr}


        result = await substrate.subscribe_block_headers(subscription_handler, include_author=True)
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
            block_hash = await self.get_chain_finalised_head()
        else:
            block_hash = await self.get_chain_head()

        return await self._get_block_handler(
            block_hash,
            subscription_handler=subscription_handler,
            ignore_decoding_errors=ignore_decoding_errors,
            include_author=include_author,
            finalized_only=finalized_only,
        )

    async def retrieve_extrinsic_by_identifier(
        self, extrinsic_identifier: str
    ) -> "AsyncExtrinsicReceipt":
        """
        Retrieve an extrinsic by its identifier in format "[block_number]-[extrinsic_index]" e.g. 333456-4

        Args:
            extrinsic_identifier: "[block_number]-[extrinsic_idx]" e.g. 134324-2

        Returns:
            ExtrinsicReceiptLike object of the extrinsic
        """
        return await AsyncExtrinsicReceipt.create_from_extrinsic_identifier(
            substrate=self, extrinsic_identifier=extrinsic_identifier
        )

    def retrieve_extrinsic_by_hash(
        self, block_hash: str, extrinsic_hash: str
    ) -> "AsyncExtrinsicReceipt":
        """
        Retrieve an extrinsic by providing the block_hash and the extrinsic hash

        Args:
            block_hash: hash of the blockchain block where the extrinsic is located
            extrinsic_hash: hash of the extrinsic

        Returns:
            ExtrinsicReceiptLike of the extrinsic
        """
        return AsyncExtrinsicReceipt(
            substrate=self, block_hash=block_hash, extrinsic_hash=extrinsic_hash
        )

    async def get_extrinsics(
        self, block_hash: Optional[str] = None, block_number: Optional[int] = None
    ) -> Optional[list["AsyncExtrinsicReceipt"]]:
        """
        Return all extrinsics for given block_hash or block_number

        Args:
            block_hash: hash of the blockchain block to retrieve extrinsics for
            block_number: block number to retrieve extrinsics for

        Returns:
            ExtrinsicReceipts of the extrinsics for the block, if any.
        """
        block = await self.get_block(block_hash=block_hash, block_number=block_number)
        if block:
            return block["extrinsics"]

    async def get_events(self, block_hash: Optional[str] = None) -> list[dict]:
        """
        Convenience method to get events for a certain block (storage call for module 'System' and function 'Events')

        Args:
            block_hash: the hash of the block to be retrieved

        Returns:
            list of events
        """

        if not block_hash:
            block_hash = await self.get_chain_head()

        events = await self.query(
            module="System",
            storage_function="Events",
            block_hash=block_hash,
        )
        return events.value

    async def get_metadata(self, block_hash=None):
        """
        Returns `MetadataVersioned` object for given block_hash or chaintip if block_hash is omitted


        Args:
            block_hash

        Returns:
            MetadataVersioned
        """
        runtime = await self.init_runtime(block_hash=block_hash)

        return runtime.metadata_v15

    @cached_fetcher(max_size=SUBSTRATE_CACHE_METHOD_SIZE)
    async def get_parent_block_hash(self, block_hash) -> str:
        """
        Retrieves the block hash of the parent of the given block hash
        Args:
            block_hash: hash of the block to query

        Returns:
            Hash of the parent block hash, or the original block hash (if it has not parent)
        """
        return await self._get_parent_block_hash(block_hash)

    async def _get_parent_block_hash(self, block_hash) -> str:
        block_header = await self.rpc_request("chain_getHeader", [block_hash])
        if "error" in block_header:
            raise SubstrateRequestException(block_header["error"]["message"])

        if block_header["result"] is None:
            raise SubstrateRequestException(f'Block not found for "{block_hash}"')
        parent_block_hash: str = block_header["result"]["parentHash"]

        if int(parent_block_hash, 16) == 0:
            # "0x0000000000000000000000000000000000000000000000000000000000000000"
            return block_hash
        return parent_block_hash

    async def get_storage_by_key(self, block_hash: str, storage_key: str) -> Any:
        """
        A pass-though to existing JSONRPC method `state_getStorage`/`state_getStorageAt`

        Args:
            block_hash: hash of the block
            storage_key: storage key to query

        Returns:
            result of the query

        """

        if await self.supports_rpc_method("state_getStorageAt"):
            response = await self.rpc_request(
                "state_getStorageAt", [storage_key, block_hash]
            )
        else:
            response = await self.rpc_request(
                "state_getStorage", [storage_key, block_hash]
            )
        return response.get("result")

    @cached_fetcher(max_size=SUBSTRATE_RUNTIME_CACHE_SIZE)
    async def get_block_runtime_info(self, block_hash: str) -> dict:
        """
        Retrieve the runtime info of given block_hash
        """
        return await self._get_block_runtime_info(block_hash)

    get_block_runtime_version = get_block_runtime_info

    async def _get_block_runtime_info(self, block_hash: str) -> dict:
        response = await self.rpc_request("state_getRuntimeVersion", [block_hash])
        return response.get("result")

    @cached_fetcher(max_size=SUBSTRATE_CACHE_METHOD_SIZE)
    async def get_block_runtime_version_for(self, block_hash: str):
        """
        Retrieve the runtime version of the parent of a given block_hash
        """
        return await self._get_block_runtime_version_for(block_hash)

    async def _get_block_runtime_version_for(self, block_hash: str):
        parent_block_hash = await self.get_parent_block_hash(block_hash)
        runtime_info = await self.get_block_runtime_info(parent_block_hash)
        if runtime_info is None:
            return None
        return runtime_info["specVersion"]

    async def get_block_metadata(
        self,
        block_hash: Optional[str] = None,
        runtime_config: Optional[RuntimeConfigurationObject] = None,
    ) -> Optional[scalecodec.types.GenericMetadataVersioned]:
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
        if not runtime_config:
            raise ValueError(
                "Cannot decode runtime configuration without a supplied runtime_config"
            )

        if block_hash:
            params = [block_hash]
        response = await self.rpc_request("state_getMetadata", params)

        if result := response.get("result"):
            metadata_decoder = runtime_config.create_scale_object(
                "MetadataVersioned", data=ScaleBytes(result)
            )
            metadata_decoder.decode()
            return metadata_decoder
        return None

    async def _preprocess(
        self,
        query_for: Optional[list],
        block_hash: Optional[str],
        storage_function: str,
        module: str,
        raw_storage_key: Optional[bytes] = None,
        runtime: Optional[Runtime] = None,
    ) -> Preprocessed:
        """
        Creates a Preprocessed data object for passing to `_make_rpc_request`
        """
        params = query_for if query_for else []
        # Search storage call in metadata
        if runtime is None:
            runtime = self.runtime
        metadata_pallet = runtime.metadata.get_metadata_pallet(module)

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
                metadata=runtime.metadata,
                runtime_config=runtime.runtime_config,
            )
        else:
            storage_key = StorageKey.create_from_storage_function(
                module,
                storage_item.value["name"],
                params,
                runtime_config=runtime.runtime_config,
                metadata=runtime.metadata,
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

    async def _process_response(
        self,
        response: dict,
        subscription_id: int | str,
        value_scale_type: Optional[str] = None,
        storage_item: Optional[ScaleType] = None,
        result_handler: Optional[ResultHandler] = None,
        runtime: Optional[Runtime] = None,
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
            runtime: Optional Runtime to use for decoding. If not specified, the currently-loaded `self.runtime` is used
            force_legacy_decode: Whether to force the use of the legacy Metadata V14 decoder

        Returns:
             (decoded response, completion)
        """
        result: dict | ScaleType = response
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
            result = await self.decode_scale(value_scale_type, q, runtime=runtime)
        if asyncio.iscoroutinefunction(result_handler):
            # For multipart responses as a result of subscriptions.
            message, bool_result = await result_handler(result, subscription_id)
            return message, bool_result
        return result, True

    async def _make_rpc_request(
        self,
        payloads: list[dict],
        value_scale_type: Optional[str] = None,
        storage_item: Optional[ScaleType] = None,
        result_handler: Optional[ResultHandler] = None,
        attempt: int = 1,
        runtime: Optional[Runtime] = None,
        force_legacy_decode: bool = False,
    ) -> RequestResults:
        request_manager = RequestManager(payloads)

        if len(set(x["id"] for x in payloads)) != len(payloads):
            raise ValueError("Payloads must have unique ids")

        subscription_added = False

        async with self.ws as ws:
            await ws.mark_waiting_for_response()
            for payload in payloads:
                item_id = await ws.send(payload["payload"])
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
                for item_id in request_manager.unresponded():
                    if (
                        item_id not in request_manager.responses
                        or asyncio.iscoroutinefunction(result_handler)
                    ):
                        if response := await ws.retrieve(item_id):
                            if (
                                asyncio.iscoroutinefunction(result_handler)
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
                                    logger.error(
                                        f"Error received from subtensor for {item_id}: {response}\n"
                                        f"Currently received responses: {request_manager.get_results()}"
                                    )
                                    raise SubstrateRequestException(str(response))
                            (
                                decoded_response,
                                complete,
                            ) = await self._process_response(
                                response,
                                item_id,
                                value_scale_type,
                                storage_item,
                                result_handler,
                                runtime=runtime,
                                force_legacy_decode=force_legacy_decode,
                            )
                            request_manager.add_response(
                                item_id, decoded_response, complete
                            )
                            # truncate to 2000 chars for debug logging
                            if (
                                len(stringified_response := str(decoded_response))
                                < 2_000
                            ):
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
                    await ws.mark_response_received()
                    break
                else:
                    await asyncio.sleep(0.01)

        return request_manager.get_results()

    async def supports_rpc_method(self, name: str) -> bool:
        """
        Check if substrate RPC supports given method
        Parameters
        ----------
        name: name of method to check

        Returns
        -------
        bool
        """
        result = (await self.rpc_request("rpc_methods", [])).get("result")
        if result:
            self.config["rpc_methods"] = result.get("methods", [])

        return name in self.config["rpc_methods"]

    async def rpc_request(
        self,
        method: str,
        params: Optional[list],
        result_handler: Optional[ResultHandler] = None,
        block_hash: Optional[str] = None,
        reuse_block_hash: bool = False,
        runtime: Optional[Runtime] = None,
    ) -> Any:
        """
        Makes an RPC request to the subtensor. Use this only if `self.query` and `self.query_multi` and
        `self.query_map` do not meet your needs.

        Args:
            method: str the method in the RPC request
            params: list of the params in the RPC request
            result_handler: ResultHandler
            block_hash: the hash of the block — only supply this if not supplying the block
                hash in the params, and not reusing the block hash
            reuse_block_hash: whether to reuse the block hash in the params — only mark as True
                if not supplying the block hash in the params, or via the `block_hash` parameter
            runtime: Optional runtime to be used for decoding results of the request. If not specified, the
                currently-loaded `self.runtime` is used.

        Returns:
            the response from the RPC request
        """
        block_hash = await self._get_current_block_hash(block_hash, reuse_block_hash)
        params = params or []
        payload_id = f"{method}{random.randint(0, 7000)}"
        payloads = [
            self.make_payload(
                payload_id,
                method,
                params + [block_hash] if block_hash else params,
            )
        ]
        result = await self._make_rpc_request(
            payloads, result_handler=result_handler, runtime=runtime
        )
        if "error" in result[payload_id][0]:
            if "Failed to get runtime version" in (
                err_msg := result[payload_id][0]["error"]["message"]
            ):
                logger.warning(
                    "Failed to get runtime. Re-fetching from chain, and retrying."
                )
                runtime = await self.init_runtime(block_hash=block_hash)
                return await self.rpc_request(
                    method,
                    params,
                    result_handler,
                    block_hash,
                    reuse_block_hash,
                    runtime=runtime,
                )
            elif (
                "Client error: Api called for an unknown Block: State already discarded"
                in err_msg
            ):
                bh = err_msg.split("State already discarded for ")[1].strip()
                raise StateDiscardedError(bh)
            else:
                logger.error(f"Substrate Request Exception: {result[payload_id]}")
                raise SubstrateRequestException(err_msg)
        if "result" in result[payload_id][0]:
            return result[payload_id][0]
        else:
            logger.error(f"Substrate Request Exception: {result[payload_id]}")
            raise SubstrateRequestException(result[payload_id][0])

    async def get_block_hash(self, block_id: Optional[int]) -> str:
        """
        Retrieves the hash of the specified block number, or the chaintip if None
        Args:
            block_id: block number

        Returns:
            Hash of the block
        """
        if block_id is None:
            return await self.get_chain_head()
        else:
            if (block_hash := self.runtime_cache.blocks.get(block_id)) is not None:
                return block_hash

            block_hash = await self._cached_get_block_hash(block_id)
            self.runtime_cache.add_item(block_hash=block_hash, block=block_id)
            return block_hash

    @cached_fetcher(max_size=SUBSTRATE_CACHE_METHOD_SIZE)
    async def _cached_get_block_hash(self, block_id: int) -> str:
        """
        The design of this method is as such, because it allows for an easy drop-in for a different cache, such
        as is the case with DiskCachedAsyncSubstrateInterface._cached_get_block_hash
        """
        return await self._get_block_hash(block_id)

    async def _get_block_hash(self, block_id: Optional[int]) -> str:
        return (await self.rpc_request("chain_getBlockHash", [block_id]))["result"]

    async def get_chain_head(self) -> str:
        response = await self._make_rpc_request(
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

    async def compose_call(
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

        runtime = await self.init_runtime(block_hash=block_hash)

        call: GenericCall = runtime.runtime_config.create_scale_object(
            type_string="Call", metadata=runtime.metadata
        )

        call.encode(
            {
                "call_module": call_module,
                "call_function": call_function,
                "call_args": call_params,
            }
        )

        return call

    async def query_multi(
        self,
        storage_keys: list[StorageKey],
        block_hash: Optional[str] = None,
        runtime: Optional[Runtime] = None,
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
            runtime: Optional `Runtime` to be used for decoding. If not specified, the currently-loaded `self.runtime`
                is used.

        Returns:
            list of `(storage_key, scale_obj)` tuples
        """
        if runtime is None:
            runtime = await self.init_runtime(block_hash=block_hash)

        # Retrieve corresponding value
        response = await self.rpc_request(
            "state_queryStorageAt",
            [[s.to_hex() for s in storage_keys], block_hash],
            runtime=runtime,
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

    async def create_scale_object(
        self,
        type_string: str,
        data: Optional[ScaleBytes] = None,
        block_hash: Optional[str] = None,
        runtime: Optional[Runtime] = None,
        **kwargs,
    ) -> "ScaleType":
        """
        Convenience method to create a SCALE object of type `type_string`, this will initialize the runtime
        automatically at moment of `block_hash`, or chain tip if omitted.

        Args:
            type_string: Name of SCALE type to create
            data: ScaleBytes: ScaleBytes to decode
            block_hash: block hash for moment of decoding, when omitted the chain tip will be used
            runtime: Optional `Runtime` to use for the creation of the scale object. If not specified, the
                currently-loaded `self.runtime` will be used.
            kwargs: keyword args for the Scale Type constructor

        Returns:
             The created Scale Type object
        """
        if runtime is None:
            runtime = await self.init_runtime(block_hash=block_hash)
        if "metadata" not in kwargs:
            kwargs["metadata"] = runtime.metadata

        return runtime.runtime_config.create_scale_object(
            type_string, data=data, **kwargs
        )

    async def generate_signature_payload(
        self,
        call: GenericCall,
        era=None,
        nonce: int = 0,
        tip: int = 0,
        tip_asset_id: Optional[int] = None,
        include_call_length: bool = False,
        runtime: Optional[Runtime] = None,
    ) -> ScaleBytes:
        # Retrieve genesis hash
        genesis_hash = await self.get_block_hash(0)
        if runtime is None:
            runtime = await self.init_runtime(block_hash=None)

        if not era:
            era = "00"

        if era == "00":
            # Immortal extrinsic
            block_hash = genesis_hash
        else:
            # Determine mortality of extrinsic
            era_obj = runtime.runtime_config.create_scale_object("Era")

            if isinstance(era, dict) and "current" not in era and "phase" not in era:
                raise ValueError(
                    'The era dict must contain either "current" or "phase" element to encode a valid era'
                )

            era_obj.encode(era)
            block_hash = await self.get_block_hash(
                block_id=era_obj.birth(era.get("current"))
            )

        # Create signature payload
        signature_payload = runtime.runtime_config.create_scale_object(
            "ExtrinsicPayloadValue"
        )

        # Process signed extensions in metadata
        if "signed_extensions" in runtime.metadata[1][1]["extrinsic"]:
            # Base signature payload
            signature_payload.type_mapping = [["call", "CallBytes"]]

            # Add signed extensions to payload
            signed_extensions = runtime.metadata.get_signed_extensions()

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
            length_obj = runtime.runtime_config.create_scale_object("Bytes")
            call_data = str(length_obj.encode(str(call.data)))

        else:
            call_data = str(call.data)

        payload_dict = {
            "call": call_data,
            "era": era,
            "nonce": nonce,
            "tip": tip,
            "spec_version": runtime.runtime_version,
            "genesis_hash": genesis_hash,
            "block_hash": block_hash,
            "transaction_version": runtime.transaction_version,
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

    async def create_signed_extrinsic(
        self,
        call: GenericCall,
        keypair: Keypair,
        era: Optional[dict | str] = None,
        nonce: Optional[int] = None,
        tip: int = 0,
        tip_asset_id: Optional[int] = None,
        signature: Optional[bytes | str] = None,
    ) -> scalecodec.types.GenericExtrinsic:
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
        runtime = await self.init_runtime()

        # Check requirements
        if not isinstance(call, GenericCall):
            raise TypeError("'call' must be of type Call")

        # Check if extrinsic version is supported
        if runtime.metadata[1][1]["extrinsic"]["version"] != 4:  # type: ignore
            raise NotImplementedError(
                f"Extrinsic version {runtime.metadata[1][1]['extrinsic']['version']} not supported"  # type: ignore
            )

        # Retrieve nonce
        if nonce is None:
            nonce = await self.get_account_nonce(keypair.ss58_address) or 0

        # Process era
        if era is None:
            era = "00"
        else:
            if isinstance(era, dict) and "current" not in era and "phase" not in era:
                # Retrieve current block id
                era["current"] = await self.get_block_number(
                    await self.get_chain_finalised_head()
                )

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
            signature_payload = await self.generate_signature_payload(
                call=call,
                era=era,
                nonce=nonce,
                tip=tip,
                tip_asset_id=tip_asset_id,
                runtime=runtime,
            )

            # Set Signature version to crypto type of keypair
            signature_version = keypair.crypto_type

            # Sign payload
            signature = keypair.sign(signature_payload.data)
            if inspect.isawaitable(signature):
                signature = await signature

        # Create extrinsic
        extrinsic = runtime.runtime_config.create_scale_object(
            type_string="Extrinsic", metadata=runtime.metadata
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
        signature_cls = runtime.runtime_config.get_decoder_class("ExtrinsicSignature")
        if issubclass(signature_cls, runtime.runtime_config.get_decoder_class("Enum")):
            value["signature_version"] = signature_version

        extrinsic.encode(value)

        return extrinsic

    async def create_unsigned_extrinsic(self, call: GenericCall) -> GenericExtrinsic:
        """
        Create unsigned extrinsic for given `Call`

        Args:
            call: GenericCall the call the extrinsic should contain

        Returns:
            GenericExtrinsic
        """

        runtime = await self.init_runtime()

        # Create extrinsic
        extrinsic = runtime.runtime_config.create_scale_object(
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

    async def get_chain_finalised_head(self):
        """
        A pass-though to existing JSONRPC method `chain_getFinalizedHead`

        Returns
        -------

        """
        response = await self.rpc_request("chain_getFinalizedHead", [])
        return response["result"]

    async def _do_runtime_call_old(
        self,
        api: str,
        method: str,
        params: Optional[list | dict] = None,
        block_hash: Optional[str] = None,
        runtime: Optional[Runtime] = None,
    ) -> Any:
        logger.debug(
            f"Decoding old runtime call: {api}.{method} with params: {params} at block hash: {block_hash}"
        )
        runtime_call_def = _TYPE_REGISTRY["runtime_api"][api]["methods"][method]
        params = params or []
        # Encode params
        param_data = b""

        if "encoder" in runtime_call_def:
            if runtime is None:
                runtime = await self.init_runtime(block_hash=block_hash)
            param_data = runtime_call_def["encoder"](params, runtime)
        else:
            for idx, param in enumerate(runtime_call_def["params"]):
                param_type_string = f"{param['type']}"
                if isinstance(params, list):
                    param_data += await self.encode_scale(
                        param_type_string, params[idx], runtime=runtime
                    )
                else:
                    if param["name"] not in params:
                        raise ValueError(
                            f"Runtime Call param '{param['name']}' is missing"
                        )

                    param_data += await self.encode_scale(
                        param_type_string, params[param["name"]], runtime=runtime
                    )

        # RPC request
        result_data = await self.rpc_request(
            "state_call",
            [f"{api}_{method}", param_data.hex(), block_hash],
            runtime=runtime,
        )
        if "error" in result_data:
            raise SubstrateRequestException(result_data["error"]["message"])
        result_vec_u8_bytes = hex_to_bytes(result_data["result"])
        result_bytes = await self.decode_scale(
            "Vec<u8>", result_vec_u8_bytes, runtime=runtime
        )

        # Decode result
        # Get correct type
        if isinstance(result_bytes, str):
            raw_bytes = hex_to_bytes(result_bytes)
        else:
            raw_bytes = bytes(result_bytes)
        result = runtime_call_def["decoder"](raw_bytes, runtime)
        return result

    async def runtime_call(
        self,
        api: str,
        method: str,
        params: Optional[list | dict] = None,
        block_hash: Optional[str] = None,
    ) -> Any:
        """
        Calls a runtime API method

        Args:
            api: Name of the runtime API e.g. 'TransactionPaymentApi'
            method: Name of the method e.g. 'query_fee_details'
            params: List of parameters needed to call the runtime API
            block_hash: Hash of the block at which to make the runtime API call

        Returns:
             Decoded runtime call
        """
        runtime = await self.init_runtime(block_hash=block_hash)

        if params is None:
            params = {}

        try:
            if runtime.metadata_v15 is None:
                _ = runtime.runtime_config.type_registry["runtime_api"][api]["methods"][
                    method
                ]
                runtime_api_types = runtime.runtime_config.type_registry["runtime_api"][
                    api
                ].get("types", {})
                runtime.runtime_config.update_type_registry_types(runtime_api_types)
                return await self._do_runtime_call_old(
                    api, method, params, block_hash, runtime=runtime
                )

            else:
                runtime_call_def = runtime.runtime_api_map[api][method]
                if _determine_if_old_runtime_call(runtime_call_def, runtime):
                    return await self._do_runtime_call_old(
                        api, method, params, block_hash, runtime=runtime
                    )
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
                param_data += await self.encode_scale(
                    param_type_string, params[idx], runtime=runtime
                )
            else:
                if param["name"] not in params:
                    raise ValueError(f"Runtime Call param '{param['name']}' is missing")

                param_data += await self.encode_scale(
                    param_type_string, params[param["name"]], runtime=runtime
                )

        # RPC request
        result_data = await self.rpc_request(
            "state_call",
            [f"{api}_{method}", param_data.hex(), block_hash],
            runtime=runtime,
        )
        if "error" in result_data:
            raise SubstrateRequestException(result_data["error"]["message"])
        output_type_string = f"scale_info::{runtime_call_def['output']}"

        # Decode result
        result_bytes = hex_to_bytes(result_data["result"])
        return await self.decode_scale(
            output_type_string, result_bytes, runtime=runtime
        )

    async def get_account_nonce(self, account_address: str) -> int:
        """
        Returns current nonce for given account address

        Args:
            account_address: SS58 formatted address

        Returns:
            Nonce for given account address
        """
        if await self.supports_rpc_method("state_call"):
            nonce_obj = await self.runtime_call(
                "AccountNonceApi", "account_nonce", [account_address]
            )
            return getattr(nonce_obj, "value", nonce_obj)
        else:
            response = await self.query(
                module="System", storage_function="Account", params=[account_address]
            )
            return response["nonce"]

    async def get_account_next_index(
        self, account_address: str, use_cache: bool = True
    ) -> int:
        """
        This method maintains a cache of nonces for each account ss58address.
        Upon subsequent calls, it will return the cached nonce + 1 instead of fetching from the chain.
        This allows for correct nonce management in-case of async context when gathering co-routines.

        Args:
            account_address: SS58 formatted address
            use_cache: If True, bypass local nonce cache and always request fresh value from RPC.

        Returns:
            Next index for the given account address
        """

        async def _get_account_next_index():
            """Inner RPC call to get `account_nextIndex`."""
            nonce_obj_ = await self.rpc_request("account_nextIndex", [account_address])
            if "error" in nonce_obj_:
                raise SubstrateRequestException(nonce_obj_["error"]["message"])
            return nonce_obj_["result"]

        if not await self.supports_rpc_method("account_nextIndex"):
            # Unlikely to happen, this is a common RPC method
            raise Exception("account_nextIndex not supported")

        if not use_cache:
            return await _get_account_next_index()

        async with self._lock:
            if self._nonces.get(account_address) is None:
                nonce_obj = await _get_account_next_index()
                self._nonces[account_address] = nonce_obj
            else:
                self._nonces[account_address] += 1
        return self._nonces[account_address]

    async def get_metadata_constants(self, block_hash=None) -> list[dict]:
        """
        Retrieves a list of all constants in metadata active at given block_hash (or chaintip if block_hash is omitted)

        Args:
            block_hash: hash of the block

        Returns:
            list of constants
        """

        runtime = await self.init_runtime(block_hash=block_hash)
        return self._get_metadata_constants(runtime)

    async def get_metadata_constant(
        self,
        module_name: str,
        constant_name: str,
        block_hash: Optional[str] = None,
        runtime: Optional[Runtime] = None,
    ) -> Optional[scalecodec.ScaleInfoModuleConstantMetadata]:
        """
        Retrieves the details of a constant for given module name, call function name and block_hash
        (or chaintip if block_hash is omitted)

        Args:
            module_name: name of the module you are querying
            constant_name: name of the constant you are querying
            block_hash: hash of the block at which to make the runtime API call
            runtime: Runtime whose metadata you are querying.

        Returns:
            MetadataModuleConstants
        """
        if runtime is None:
            runtime = await self.init_runtime(block_hash=block_hash)
        return self._get_metadata_constant(module_name, constant_name, runtime)

    async def get_constant(
        self,
        module_name: str,
        constant_name: str,
        block_hash: Optional[str] = None,
        reuse_block_hash: bool = False,
        runtime: Optional[Runtime] = None,
    ) -> Optional[ScaleType[Any]]:
        """
        Returns the decoded `ScaleType` object of the constant for given module name, call function name and block_hash
        (or chaintip if block_hash is omitted)

        Args:
            module_name: Name of the module to query
            constant_name: Name of the constant to query
            block_hash: Hash of the block at which to make the runtime API call
            reuse_block_hash: Reuse last-used block hash if set to true
            runtime: Runtime to use for querying the constant

        Returns:
             ScaleType from the runtime call
        """
        block_hash = await self._get_current_block_hash(block_hash, reuse_block_hash)
        constant = await self.get_metadata_constant(
            module_name, constant_name, block_hash=block_hash, runtime=runtime
        )
        if constant:
            # Decode to ScaleType
            return await self.decode_scale(
                constant.type,
                bytes(constant.constant_value),
                runtime=runtime,
            )
        else:
            return None

    async def get_payment_info(
        self,
        call: GenericCall,
        keypair: Keypair,
        era: Optional[dict | str] = None,
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
        extrinsic = await self.create_signed_extrinsic(
            call=call,
            keypair=keypair,
            era=era,
            nonce=nonce,
            tip=tip,
            tip_asset_id=tip_asset_id,
            signature=signature,
        )
        extrinsic_len = len(extrinsic.data)

        result = await self.runtime_call(
            "TransactionPaymentApi", "query_info", [extrinsic, extrinsic_len]
        )

        return result.value

    async def get_type_registry(
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
        runtime = await self.init_runtime(block_hash=block_hash)

        if not runtime.implements_scaleinfo:
            raise NotImplementedError("MetadataV14 or higher runtimes is required")

        type_registry = {}

        for scale_info_type in runtime.metadata.portable_registry["types"]:
            if (
                "path" in scale_info_type.value["type"]
                and len(scale_info_type.value["type"]["path"]) > 0
            ):
                type_string = "::".join(scale_info_type.value["type"]["path"])
            else:
                type_string = f"scale_info::{scale_info_type.value['id']}"

            scale_cls = runtime.runtime_config.get_decoder_class(type_string)
            type_registry[type_string] = scale_cls.generate_type_decomposition(
                max_recursion=max_recursion
            )

        return type_registry

    async def get_type_definition(
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
        scale_obj = await self.create_scale_object(type_string, block_hash=block_hash)
        return scale_obj.generate_type_decomposition()

    async def get_metadata_modules(self, block_hash=None) -> list[dict[str, Any]]:
        """
        Retrieves a list of modules in metadata for given block_hash (or chaintip if block_hash is omitted)

        Args:
            block_hash: hash of the blockchain block

        Returns:
            List of metadata modules
        """
        runtime = await self.init_runtime(block_hash=block_hash)
        return self._get_metadata_modules(runtime)

    async def get_metadata_module(self, name, block_hash=None) -> ScaleType:
        """
        Retrieves modules in metadata by name for given block_hash (or chaintip if block_hash is omitted)

        Args:
            name: Name of the module
            block_hash: hash of the blockchain block

        Returns:
            MetadataModule
        """
        runtime = await self.init_runtime(block_hash=block_hash)

        return runtime.metadata.get_metadata_pallet(name)

    async def query(
        self,
        module: str,
        storage_function: str,
        params: Optional[list] = None,
        block_hash: Optional[str] = None,
        raw_storage_key: Optional[bytes] = None,
        subscription_handler=None,
        reuse_block_hash: bool = False,
        runtime: Optional[Runtime] = None,
    ) -> Optional[ScaleType[Any]]:
        """
        Queries substrate. This should only be used when making a single request. For multiple requests,
        you should use `self.query_multi`
        """
        block_hash = await self._get_current_block_hash(block_hash, reuse_block_hash)
        if block_hash:
            self.last_block_hash = block_hash
        if runtime is None:
            runtime = await self.init_runtime(block_hash=block_hash)
        preprocessed: Preprocessed = await self._preprocess(
            params,
            block_hash,
            storage_function,
            module,
            raw_storage_key,
            runtime=runtime,
        )
        payload = [
            self.make_payload(
                preprocessed.queryable, preprocessed.method, preprocessed.params
            )
        ]
        value_scale_type = preprocessed.value_scale_type
        storage_item = preprocessed.storage_item

        responses = await self._make_rpc_request(
            payload,
            value_scale_type,
            storage_item,
            result_handler=subscription_handler,
            runtime=runtime,
        )
        result = responses[preprocessed.queryable][0]
        return result

    async def query_map(
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
        fully_exhaust: bool = False,
    ) -> AsyncQueryMapResult:
        """
        Iterates over all key-pairs located at the given module and storage_function. The storage
        item must be a map.

        Example:

        ```
        result = await substrate.query_map('System', 'Account', max_results=100)

        async for account, account_info in result:
            print(f"Free balance of account '{account.value}': {account_info.value['data']['free']}")
        ```

        Note: it is important that you do not use `for x in result.records`, as this will sidestep possible
        pagination. You must do `async for x in result`.

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
            fully_exhaust: Pull the entire result at once, rather than paginating. Only use if you need the entire query
                map result.

        Returns:
             AsyncQueryMapResult object
        """
        params = params or []
        block_hash = await self._get_current_block_hash(block_hash, reuse_block_hash)
        if block_hash:
            self.last_block_hash = block_hash
        runtime = await self.init_runtime(block_hash=block_hash)

        metadata_pallet = runtime.metadata.get_metadata_pallet(module)
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
        storage_key = StorageKey.create_from_storage_function(
            module,
            storage_item.value["name"],
            params,
            runtime_config=runtime.runtime_config,
            metadata=runtime.metadata,
        )
        prefix = storage_key.to_hex()

        if not start_key:
            start_key = prefix

        # Make sure if the max result is smaller than the page size, adjust the page size
        if max_results is not None and max_results < page_size:
            page_size = max_results

        # Retrieve storage keys
        if not fully_exhaust:
            response = await self.rpc_request(
                method="state_getKeysPaged",
                params=[prefix, page_size, start_key, block_hash],
                runtime=runtime,
            )
        else:
            response = await self.rpc_request(
                method="state_getKeys", params=[prefix, block_hash], runtime=runtime
            )

        result_keys = response.get("result")

        result = []
        last_key = None

        if len(result_keys) > 0:
            last_key = result_keys[-1]

            # Retrieve corresponding value(s)
            if not fully_exhaust:
                response = await self.rpc_request(
                    method="state_queryStorageAt",
                    params=[result_keys, block_hash],
                    runtime=runtime,
                )
                changes = []
                for result_group in response["result"]:
                    changes.extend(result_group["changes"])
                result = decode_query_map(
                    changes,
                    prefix,
                    runtime,
                    param_types,
                    params,
                    value_type,
                    key_hashers,
                    ignore_decoding_errors,
                )
            else:
                # storage item and value scale type are not included here because this is batch-decoded in rust
                page_batches = [
                    result_keys[i : i + page_size]
                    for i in range(0, len(result_keys), page_size)
                ]
                changes = []
                payloads = []
                for idx, page_batch in enumerate(page_batches):
                    payloads.append(
                        self.make_payload(
                            str(idx), "state_queryStorageAt", [page_batch, block_hash]
                        )
                    )
                results: RequestResults = await self._make_rpc_request(
                    payloads, runtime=runtime
                )
                for result_ in results.values():
                    res = result_[0]
                    if "error" in res:
                        err_msg = res["error"]["message"]
                        if (
                            "Client error: Api called for an unknown Block: State already discarded"
                            in err_msg
                        ):
                            bh = err_msg.split("State already discarded for ")[
                                1
                            ].strip()
                            raise StateDiscardedError(bh)
                        else:
                            raise SubstrateRequestException(err_msg)
                    elif "result" not in res:
                        raise SubstrateRequestException(res)
                    else:
                        for result_group in res["result"]:
                            changes.extend(result_group["changes"])
                result = decode_query_map(
                    changes,
                    prefix,
                    runtime,
                    param_types,
                    params,
                    value_type,
                    key_hashers,
                    ignore_decoding_errors,
                )
        return AsyncQueryMapResult(
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

    async def create_multisig_extrinsic(
        self,
        call: GenericCall,
        keypair: Keypair,
        multisig_account: MultiAccountId,
        max_weight: Optional[dict | int] = None,
        era: Optional[dict] = None,
        nonce: Optional[int] = None,
        tip: int = 0,
        tip_asset_id: Optional[int] = None,
        signature: Optional[bytes | str] = None,
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
            payment_info = await self.get_payment_info(call, keypair)
            max_weight = payment_info["weight"]

        # Check if call has existing approvals
        multisig_details_ = await self.query(
            "Multisig", "Multisigs", [multisig_account.value, call.call_hash]
        )
        multisig_details = getattr(multisig_details_, "value", multisig_details_)
        if multisig_details:
            maybe_timepoint = multisig_details["when"]
        else:
            maybe_timepoint = None

        # Compose 'as_multi' when final, 'approve_as_multi' otherwise
        if (
            multisig_details.value
            and len(multisig_details.value["approvals"]) + 1
            == multisig_account.threshold
        ):
            multi_sig_call = await self.compose_call(
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
            multi_sig_call = await self.compose_call(
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

        return await self.create_signed_extrinsic(
            multi_sig_call,
            keypair,
            era=era,
            nonce=nonce,
            tip=tip,
            tip_asset_id=tip_asset_id,
            signature=signature,
        )

    async def submit_extrinsic(
        self,
        extrinsic: GenericExtrinsic,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = False,
    ) -> "AsyncExtrinsicReceipt":
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

        async def result_handler(message: dict, subscription_id) -> tuple[dict, bool]:
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
                    async with self.ws as ws:
                        await ws.unsubscribe(subscription_id)
                    logger.error(failure_message)
                    raise SubstrateRequestException(failure_message)

                if "finalized" in message_result and wait_for_finalization:
                    logger.debug("Extrinsic finalized. Unsubscribing.")
                    async with self.ws as ws:
                        await ws.unsubscribe(subscription_id)
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
                    logger.debug("Extrinsic included. Unsubscribing.")
                    async with self.ws as ws:
                        await ws.unsubscribe(subscription_id)
                    return {
                        "block_hash": message_result.get(
                            "inblock", message_result.get("inBlock")
                        ),
                        "extrinsic_hash": "0x{}".format(extrinsic.extrinsic_hash.hex()),
                        "finalized": False,
                    }, True

            elif "params" in message and message["params"].get("result") == "invalid":
                failure_message = f"Subscription {subscription_id} invalid: {message}"
                async with self.ws as ws:
                    await ws.unsubscribe(subscription_id)
                logger.error(failure_message)
                raise SubstrateRequestException(failure_message)

            return message, False

        if wait_for_inclusion or wait_for_finalization:
            responses = (
                await self._make_rpc_request(
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
            result = AsyncExtrinsicReceipt(
                substrate=self,
                extrinsic_hash=response["extrinsic_hash"],
                block_hash=response["block_hash"],
                finalized=response["finalized"],
            )

        else:
            response = await self.rpc_request(
                "author_submitExtrinsic", [str(extrinsic.data)]
            )

            result = AsyncExtrinsicReceipt(
                substrate=self, extrinsic_hash=response["result"]
            )

        return result

    async def get_metadata_call_functions(
        self, block_hash: Optional[str] = None, runtime: Optional[Runtime] = None
    ) -> dict[str, dict[str, dict[str, dict[str, str | int | list]]]]:
        """
        Retrieves calls functions for the metadata at the specified block_hash or runtime. If neither are specified,
        the metadata at chaintip is used.

        Args:
            block_hash: block hash to retrieve metadata for, unused if supplying runtime
            runtime: Runtime object containing the metadata you wish to parse

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
        if runtime is None:
            runtime = await self.init_runtime(block_hash=block_hash)
        return self._get_metadata_call_functions(runtime)

    async def get_metadata_call_function(
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
        runtime = await self.init_runtime(block_hash=block_hash)

        return self._get_metadata_call_function(
            module_name, call_function_name, runtime
        )

    async def get_metadata_events(self, block_hash=None) -> list[dict]:
        """
        Retrieves a list of all events in metadata active for given block_hash (or chaintip if block_hash is omitted)

        Args:
            block_hash

        Returns:
            list of module events
        """

        runtime = await self.init_runtime(block_hash=block_hash)
        return self._get_metadata_events(runtime)

    async def get_metadata_event(
        self, module_name, event_name, block_hash=None
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

        runtime = await self.init_runtime(block_hash=block_hash)
        return self._get_metadata_event(module_name, event_name, runtime)

    async def get_block_number(self, block_hash: Optional[str] = None) -> int:
        """Async version of `substrateinterface.base.get_block_number` method."""
        if block_hash is None:
            return await self._get_block_number(None)
        if (block := self.runtime_cache.blocks_reverse.get(block_hash)) is not None:
            return block
        block = await self._cached_get_block_number(block_hash)
        self.runtime_cache.add_item(block_hash=block_hash, block=block)
        return block

    @cached_fetcher(max_size=SUBSTRATE_CACHE_METHOD_SIZE)
    async def _cached_get_block_number(self, block_hash: str) -> int:
        """
        The design of this method is as such, because it allows for an easy drop-in for a different cache, such
        as is the case with DiskCachedAsyncSubstrateInterface._cached_get_block_number
        """
        return await self._get_block_number(block_hash=block_hash)

    async def _get_block_number(self, block_hash: Optional[str]) -> int:
        response = await self.rpc_request("chain_getHeader", [block_hash])
        return int(response["result"]["number"], 16)

    async def close(self):
        """
        Closes the substrate connection, and the websocket connection.
        """
        try:
            if self.startup_runtime_task is not None:
                self.startup_runtime_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self.startup_runtime_task
            await self.ws.shutdown()
        except AttributeError:
            pass

    async def wait_for_block(
        self,
        block: int,
        result_handler: Callable[[dict], Awaitable[Any]],
        task_return: bool = True,
    ) -> asyncio.Task | bool | Any:
        """
        Executes the result_handler when the chain has reached the block specified.

        Args:
            block: block number
            result_handler: coroutine executed upon reaching the block number. This can be basically anything, but
                must accept one single arg, a dict with the block data; whether you use this data or not is entirely
                up to you.
            task_return: True to immediately return the result of wait_for_block as an asyncio Task, False to wait
                for the block to be reached, and return the result of the result handler.

        Returns:
            Either an asyncio.Task (which contains the running subscription, and whose `result()` will contain the
                return of the result_handler), or the result itself, depending on `task_return` flag.
                Note that if your result_handler returns `None`, this method will return `True`, otherwise
                the return will be the result of your result_handler.
        """

        async def _handler(block_data: dict[str, Any]):
            required_number = block
            number = block_data["header"]["number"]
            if number >= required_number:
                return (
                    r if (r := await result_handler(block_data)) is not None else True
                )

        args = inspect.getfullargspec(result_handler).args
        if len(args) != 1:
            raise ValueError(
                "result_handler must take exactly one arg: the dict block data."
            )

        co = self._get_block_handler(
            self.last_block_hash, subscription_handler=_handler
        )
        if task_return is True:
            return asyncio.create_task(co)
        else:
            return await co


class DiskCachedAsyncSubstrateInterface(AsyncSubstrateInterface):
    """
    Uses disk-caching in addition to memory-caching for the cached methods

    Loads the cache from the disk at startup, where it is kept in-memory, and dumps to the disk
    when the connection is closed.

    For `wss://` endpoints, a persistent `_SessionResumingSSLContext` is created so
    that TLS sessions are reused across reconnections.  The effective session TTL is the minimum
    of `ssl_session_ttl` (default `SSL_SESSION_TTL`) and the server-advertised timeout.
    """

    def __init__(
        self,
        url: str,
        *args,
        ssl_session_ttl: int = SSL_SESSION_TTL,
        **kwargs,
    ):
        ssl_context: Optional[_SessionResumingSSLContext] = None
        if url.startswith("wss://") and not kwargs.get("_mock", False):
            ssl_context = _SessionResumingSSLContext(session_ttl=ssl_session_ttl)
            ssl_context.set_default_verify_paths()
        super().__init__(url, *args, _ssl_context=ssl_context, **kwargs)

    async def initialize(self) -> None:
        db = AsyncSqliteDB(self.url)
        cached = await db.load_dns_cache(self.url)
        if cached is not None:
            addrinfos, saved_at_unix = cached
            age = time.time() - saved_at_unix
            # Reconstruct a monotonic timestamp so _resolve_host's TTL check works correctly
            self.ws._dns_cache = (addrinfos, time.monotonic() - age)
            logger.debug(f"Loaded DNS cache from disk (age={age:.0f}s)")
        await self.runtime_cache.load_from_disk(self.url)
        await self._initialize()

    async def close(self):
        """
        Closes the substrate connection and the websocket connection, dumps the runtime and DNS
        caches to disk.
        """
        db = AsyncSqliteDB(self.url)
        dns_cache = getattr(self.ws, "_dns_cache", None)
        if dns_cache is not None:
            addrinfos, _ = dns_cache
            await db.save_dns_cache(self.url, addrinfos)
        try:
            await self.runtime_cache.dump_to_disk(self.url)
            await self.ws.shutdown()
        except AttributeError:
            pass
        await db.close()

    @async_sql_lru_cache(maxsize=SUBSTRATE_CACHE_METHOD_SIZE)
    async def get_parent_block_hash(self, block_hash):
        return await self._get_parent_block_hash(block_hash)

    @async_sql_lru_cache(maxsize=SUBSTRATE_RUNTIME_CACHE_SIZE)
    async def get_block_runtime_info(self, block_hash: str) -> dict:
        return await self._get_block_runtime_info(block_hash)

    @async_sql_lru_cache(maxsize=SUBSTRATE_CACHE_METHOD_SIZE)
    async def get_block_runtime_version_for(self, block_hash: str):
        return await self._get_block_runtime_version_for(block_hash)

    @async_sql_lru_cache(maxsize=SUBSTRATE_CACHE_METHOD_SIZE)
    async def _cached_get_block_hash(self, block_id: int) -> str:
        return await self._get_block_hash(block_id)

    @async_sql_lru_cache(maxsize=SUBSTRATE_CACHE_METHOD_SIZE)
    async def _cached_get_block_number(self, block_hash: str) -> int:
        return await self._get_block_number(block_hash=block_hash)


async def get_async_substrate_interface(
    url: str,
    use_remote_preset: bool = False,
    auto_discover: bool = True,
    ss58_format: Optional[int] = None,
    type_registry: Optional[dict] = None,
    chain_name: Optional[str] = None,
    max_retries: int = 5,
    retry_timeout: float = 60.0,
    _mock: bool = False,
) -> "AsyncSubstrateInterface":
    """
    Factory function for creating an initialized AsyncSubstrateInterface
    """
    substrate = AsyncSubstrateInterface(
        url,
        use_remote_preset=use_remote_preset,
        auto_discover=auto_discover,
        ss58_format=ss58_format,
        type_registry=type_registry,
        chain_name=chain_name,
        max_retries=max_retries,
        retry_timeout=retry_timeout,
        _mock=_mock,
    )
    await substrate.initialize()
    return substrate
