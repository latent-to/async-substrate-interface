import asyncio
import inspect
import time
import weakref
from collections import OrderedDict
import functools
import logging
import os
import pickle
import sqlite3
from pathlib import Path
from typing import Callable, Any, Awaitable, Hashable, Optional

import aiosqlite


USE_CACHE = True if os.getenv("NO_CACHE") != "1" else False
CACHE_LOCAL = os.getenv("CACHE_LOCAL") == "1"
CACHE_LOCATION = (
    os.path.expanduser(
        os.getenv("CACHE_LOCATION", "~/.cache/async-substrate-interface")
    )
    if USE_CACHE
    else ":memory:"
)
SUBSTRATE_CACHE_METHOD_SIZE = int(os.getenv("SUBSTRATE_CACHE_METHOD_SIZE", "512"))

logger = logging.getLogger("async_substrate_interface")


class AsyncSqliteDB:
    _instances: dict[str, "AsyncSqliteDB"] = {}
    _db: Optional[aiosqlite.Connection] = None
    _lock: asyncio.Lock
    _created_tables: set

    def __new__(cls, chain_endpoint: str):
        try:
            return cls._instances[chain_endpoint]
        except KeyError:
            instance = super().__new__(cls)
            instance._lock = asyncio.Lock()
            instance._created_tables = set()
            cls._instances[chain_endpoint] = instance
            return instance

    async def close(self):
        async with self._lock:
            if self._db:
                await self._db.close()
                self._db = None
                self._created_tables.clear()

    async def _create_if_not_exists(self, chain: str, table_name: str):
        if table_name in self._created_tables:
            return _check_if_local(chain)
        if not (local_chain := _check_if_local(chain)) or not USE_CACHE:
            assert self._db is not None
            await self._db.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table_name} 
                    (
                       rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                       key BLOB,
                       value BLOB,
                       chain TEXT,
                       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                       UNIQUE(key, chain)
                    );
                """
            )
            await self._db.commit()
            await self._db.execute(
                f"""
                CREATE TRIGGER IF NOT EXISTS prune_rows_trigger_{table_name} AFTER INSERT ON {table_name}
                        BEGIN
                          DELETE FROM {table_name}
                          WHERE rowid IN (
                            SELECT rowid FROM {table_name}
                            ORDER BY created_at DESC
                            LIMIT -1 OFFSET {SUBSTRATE_CACHE_METHOD_SIZE}
                          );
                        END;
                """
            )
            await self._db.commit()
            self._created_tables.add(table_name)
        return local_chain

    async def __call__(self, chain, other_self, func, args, kwargs) -> Optional[Any]:
        async with self._lock:
            if not self._db:
                _ensure_dir()
                self._db = await aiosqlite.connect(CACHE_LOCATION)
            table_name = _get_table_name(func)
            local_chain = await self._create_if_not_exists(chain, table_name)
        key = pickle.dumps((args, kwargs or None))
        if not local_chain or not USE_CACHE:
            try:
                cursor: aiosqlite.Cursor = await self._db.execute(
                    f"SELECT value FROM {table_name} WHERE key=? AND chain=?",
                    (key, chain),
                )
                result = await cursor.fetchone()
                await cursor.close()
                if result is not None:
                    return pickle.loads(result[0])
            except (pickle.PickleError, sqlite3.Error) as e:
                logger.exception("Cache error", exc_info=e)
        result = await func(other_self, *args, **kwargs)
        if not local_chain or not USE_CACHE:
            # TODO use a task here
            await self._db.execute(
                f"INSERT OR REPLACE INTO {table_name} (key, value, chain) VALUES (?,?,?)",
                (key, pickle.dumps(result), chain),
            )
            await self._db.commit()
        return result

    async def _ensure_dns_table(self):
        assert self._db is not None
        await self._db.execute(
            """CREATE TABLE IF NOT EXISTS dns_cache (
                url TEXT PRIMARY KEY,
                addrinfos BLOB,
                saved_at REAL
            )"""
        )
        await self._db.commit()

    async def load_dns_cache(self, url: str) -> Optional[tuple[list, float]]:
        """
        Load a previously saved DNS result for ``url``.

        Returns ``(addrinfos, saved_at_unix)`` where ``saved_at_unix`` is the Unix
        timestamp at which the result was saved, or ``None`` if nothing is cached.
        Skips localhost URLs.
        """
        if _check_if_local(url):
            return None
        async with self._lock:
            if not self._db:
                _ensure_dir()
                self._db = await aiosqlite.connect(CACHE_LOCATION)
            await self._ensure_dns_table()
        try:
            cursor = await self._db.execute(
                "SELECT addrinfos, saved_at FROM dns_cache WHERE url=?", (url,)
            )
            row = await cursor.fetchone()
            await cursor.close()
            if row is not None:
                return pickle.loads(row[0]), row[1]
        except (pickle.PickleError, sqlite3.Error) as e:
            logger.debug(f"DNS cache load error: {e}")
        return None

    async def save_dns_cache(self, url: str, addrinfos: list) -> None:
        """Persist DNS results for ``url`` to disk. Skips localhost URLs."""
        if _check_if_local(url):
            return
        async with self._lock:
            if not self._db:
                _ensure_dir()
                self._db = await aiosqlite.connect(CACHE_LOCATION)
            await self._ensure_dns_table()
            try:
                await self._db.execute(
                    "INSERT OR REPLACE INTO dns_cache (url, addrinfos, saved_at) VALUES (?,?,?)",
                    (url, pickle.dumps(addrinfos), time.time()),
                )
                await self._db.commit()
            except (pickle.PickleError, sqlite3.Error) as e:
                logger.debug(f"DNS cache save error: {e}")

    async def load_runtime_cache(
        self, chain: str
    ) -> tuple[OrderedDict[int, str], OrderedDict[str, int], OrderedDict[int, dict]]:
        async with self._lock:
            if not self._db:
                _ensure_dir()
                self._db = await aiosqlite.connect(CACHE_LOCATION)
        block_mapping: OrderedDict[int, str] = OrderedDict()
        block_hash_mapping: OrderedDict[str, int] = OrderedDict()
        version_mapping: OrderedDict[int, dict] = OrderedDict()
        tables: dict[str, OrderedDict] = {
            "RuntimeCache_blocks": block_mapping,
            "RuntimeCache_block_hashes": block_hash_mapping,
            "RuntimeCache_versions": version_mapping,
        }
        for table in tables.keys():
            async with self._lock:
                local_chain = await self._create_if_not_exists(chain, table)
            if local_chain:
                return block_mapping, block_hash_mapping, version_mapping
        for table_name, mapping in tables.items():
            try:
                async with self._lock:
                    assert self._db is not None
                    cursor: aiosqlite.Cursor = await self._db.execute(
                        f"SELECT key, value FROM {table_name} WHERE chain=?",
                        (chain,),
                    )
                    results = await cursor.fetchall()
                    await cursor.close()
                if results is None:
                    continue
                for row in results:
                    key, value = row
                    runtime = pickle.loads(value)
                    mapping[key] = runtime
            except (pickle.PickleError, sqlite3.Error) as e:
                logger.exception("Cache error", exc_info=e)
                return block_mapping, block_hash_mapping, version_mapping
        return block_mapping, block_hash_mapping, version_mapping

    async def dump_runtime_cache(
        self,
        chain: str,
        block_mapping: dict,
        block_hash_mapping: dict,
        version_mapping: dict,
    ) -> None:
        async with self._lock:
            if not self._db:
                _ensure_dir()
                self._db = await aiosqlite.connect(CACHE_LOCATION)

            tables = {
                "RuntimeCache_blocks": block_mapping,
                "RuntimeCache_block_hashes": block_hash_mapping,
                "RuntimeCache_versions": version_mapping,
            }
            for table, mapping in tables.items():
                local_chain = await self._create_if_not_exists(chain, table)
                if local_chain:
                    return None
                serialized_mapping = {}
                for key, value in mapping.items():
                    if not isinstance(value, (str, int)):
                        serialized_value = pickle.dumps(value.serialize())
                    else:
                        serialized_value = pickle.dumps(value)
                    serialized_mapping[key] = serialized_value

                await self._db.executemany(
                    f"INSERT OR REPLACE INTO {table} (key, value, chain) VALUES (?,?,?)",
                    [
                        (key, serialized_value_, chain)
                        for key, serialized_value_ in serialized_mapping.items()
                    ],
                )

            await self._db.commit()

            return None


def _ensure_dir():
    path = Path(CACHE_LOCATION).parent
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def _get_table_name(func):
    """Convert "ClassName.method_name" to "ClassName_method_name"""
    return func.__qualname__.replace(".", "_")


def _check_if_local(chain: str) -> bool:
    if CACHE_LOCAL:
        return False
    return any([x in chain for x in ["127.0.0.1", "localhost", "0.0.0.0"]])


def _create_table(c, conn, table_name):
    c.execute(
        f"""CREATE TABLE IF NOT EXISTS {table_name} 
        (
           rowid INTEGER PRIMARY KEY AUTOINCREMENT,
           key BLOB,
           value BLOB,
           chain TEXT,
           created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
           UNIQUE(key, chain)
        );
        """
    )
    c.execute(
        f"""CREATE TRIGGER IF NOT EXISTS prune_rows_trigger AFTER INSERT ON {table_name}
            BEGIN
              DELETE FROM {table_name}
              WHERE rowid IN (
                SELECT rowid FROM {table_name}
                ORDER BY created_at DESC
                LIMIT -1 OFFSET {SUBSTRATE_CACHE_METHOD_SIZE}
              );
            END;"""
    )
    conn.commit()


def _retrieve_from_cache(c, table_name, key, chain):
    try:
        c.execute(
            f"SELECT value FROM {table_name} WHERE key=? AND chain=?", (key, chain)
        )
        result = c.fetchone()
        if result is not None:
            return pickle.loads(result[0])
    except (pickle.PickleError, sqlite3.Error) as e:
        logger.exception("Cache error", exc_info=e)
        pass


def _insert_into_cache(c, conn, table_name, key, result, chain):
    try:
        c.execute(
            f"INSERT OR REPLACE INTO {table_name} (key, value, chain) VALUES (?,?,?)",
            (key, pickle.dumps(result), chain),
        )
        conn.commit()
    except (pickle.PickleError, sqlite3.Error) as e:
        logger.exception("Cache error", exc_info=e)
        pass


def _shared_inner_fn_logic(func, self, args, kwargs):
    chain = self.url
    if not (local_chain := _check_if_local(chain)) or not USE_CACHE:
        _ensure_dir()
        conn = sqlite3.connect(CACHE_LOCATION)
        c = conn.cursor()
        table_name = _get_table_name(func)
        _create_table(c, conn, table_name)
        key = pickle.dumps((args, kwargs))
        result = _retrieve_from_cache(c, table_name, key, chain)
    else:
        result = None
        c = None
        conn = None
        table_name = None
        key = None
    return c, conn, table_name, key, result, chain, local_chain


def sql_lru_cache(maxsize=None):
    def decorator(func):
        @functools.lru_cache(maxsize=maxsize)
        def inner(self, *args, **kwargs):
            c, conn, table_name, key, result, chain, local_chain = (
                _shared_inner_fn_logic(func, self, args, kwargs)
            )

            # If not in DB, call func and store in DB
            if result is None:
                result = func(self, *args, **kwargs)

            if not local_chain or not USE_CACHE:
                _insert_into_cache(c, conn, table_name, key, result, chain)

            return result

        return inner

    return decorator


def async_sql_lru_cache(maxsize: Optional[int] = None):
    def decorator(func):
        @cached_fetcher(max_size=maxsize, cache_key_index=None)
        async def inner(self, *args, **kwargs):
            async_sql_db = AsyncSqliteDB(self.url)
            result = await async_sql_db(self.url, self, func, args, kwargs)
            return result

        return inner

    return decorator


class LRUCache:
    """
    Basic Least-Recently-Used Cache, with simple methods `set` and `get`
    """

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()

    def set(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def get(self, key):
        if key in self.cache:
            # Mark as recently used
            self.cache.move_to_end(key)
            return self.cache[key]
        return None


class CachedFetcher:
    """
    Async caching class that allows the standard async LRU cache system, but also allows for concurrent
    asyncio calls (with the same args) to use the same result of a single call.

    This should only be used for asyncio calls where the result is immutable.

    Concept and usage:
        ```
        async def fetch(self, block_hash: str) -> str:
            return await some_resource(block_hash)

        a1, a2, b = await asyncio.gather(fetch("a"), fetch("a"), fetch("b"))
        ```

        Here, you are making three requests, but you really only need to make two I/O requests
        (one for "a", one for "b"), and while you wouldn't typically make a request like this directly, it's very
        common in using this library to inadvertently make these requests y gathering multiple resources that depend
        on the calls like this under the hood.

        By using

        ```
        @cached_fetcher(max_size=512)
        async def fetch(self, block_hash: str) -> str:
            return await some_resource(block_hash)

        a1, a2, b = await asyncio.gather(fetch("a"), fetch("a"), fetch("b"))
        ```

        You are only making two I/O calls, and a2 will simply use the result of a1 when it lands.
    """

    def __init__(
        self,
        max_size: int,
        method: Callable[..., Awaitable[Any]],
        cache_key_index: Optional[int] = 0,
    ):
        """
        Args:
            max_size: max size of the cache (in items)
            method: the function to cache
            cache_key_index: if the method takes multiple args, this is the index of that cache key in the args list
                (default is the first arg). By setting this to `None`, it will use all args as the cache key.
        """
        self._inflight: dict[Hashable, asyncio.Future] = {}
        self._method = method
        self._max_size = max_size
        self._cache = LRUCache(max_size=max_size)
        self._cache_key_index = cache_key_index

    def make_cache_key(self, args: tuple, kwargs: dict) -> Hashable:
        bound = inspect.signature(self._method).bind(*args, **kwargs)
        bound.apply_defaults()

        if self._cache_key_index is not None:
            key_name = list(bound.arguments)[self._cache_key_index]
            return bound.arguments[key_name]

        return pickle.dumps(dict(bound.arguments))

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        key = self.make_cache_key(args, kwargs)

        if item := self._cache.get(key):
            return item

        if key in self._inflight:
            return await self._inflight[key]

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._inflight[key] = future

        try:
            result = await self._method(*args, **kwargs)
            self._cache.set(key, result)
            future.set_result(result)
            return result
        except Exception as e:
            self._inflight.pop(key, None)
            future.cancel()
            raise
        finally:
            self._inflight.pop(key, None)


class _WeakMethod:
    """
    Weak reference to a bound method that allows the instance to be garbage collected.
    Preserves the method's signature for introspection.
    """

    def __init__(self, method):
        self._func = method.__func__
        self._instance_ref = weakref.ref(method.__self__)
        # Store the bound method's signature (without 'self') for inspect.signature() to find.
        # We capture this once at creation time to avoid holding references to the bound method.
        self.__signature__ = inspect.signature(method)

    def __call__(self, *args, **kwargs):
        instance = self._instance_ref()
        if instance is None:
            raise ReferenceError("Instance has been garbage collected")
        return self._func(instance, *args, **kwargs)


class _CachedFetcherMethod:
    """
    Helper class for using CachedFetcher with method caches (rather than functions)
    """

    def __init__(self, method, max_size: int, cache_key_index: int):
        self.method = method
        self.max_size = max_size
        self.cache_key_index = cache_key_index
        # Use WeakKeyDictionary to avoid preventing garbage collection of instances
        self._instances: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

    def __get__(self, instance, owner):
        if instance is None:
            return self

        # Cache per-instance (weak references allow GC when instance is no longer used)
        if instance not in self._instances:
            bound_method = self.method.__get__(instance, owner)
            # Use weak reference wrapper to avoid preventing GC of instance
            weak_method = _WeakMethod(bound_method)
            self._instances[instance] = CachedFetcher(
                max_size=self.max_size,
                method=weak_method,
                cache_key_index=self.cache_key_index,
            )
        return self._instances[instance]


def cached_fetcher(max_size: Optional[int] = None, cache_key_index: Optional[int] = 0):
    """Wrapper for CachedFetcher. See example in CachedFetcher docstring."""

    def wrapper(method):
        return _CachedFetcherMethod(method, max_size, cache_key_index)

    return wrapper
