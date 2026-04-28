# Async Substrate Interface

This project provides an asynchronous interface for interacting with [Substrate](https://substrate.io/)-based
blockchains. Its API is based on the [py-substrate-interface](https://github.com/polkascan/py-substrate-interface)
project, but is not fully compatible.

Additionally, this project uses [cyscale](https://github.com/latent-to/cyscale) instead
of [py-scale-codec](https://github.com/polkascan/py-scale-codec) for
faster [SCALE](https://docs.substrate.io/reference/scale-codec/) decoding. Since v2.0, because cyscale and
py-substrate-interface share the same namespace, we require that only
cyscale is installed to be able to use this. If you run into a runtime error stating that both cannot be installed at
the same time, simply remove them both, and reinstall cyscale:

```shell
pip uninstall cyscale scalecodec -y
pip install cyscale
```

## Installation

To install the package, use the following command:

```bash
pip install async-substrate-interface
```

## Usage

Here are examples of how to use the sync and async interfaces:

```python
from async_substrate_interface import SubstrateInterface


def main():
    substrate = SubstrateInterface(
        url="wss://rpc.polkadot.io"
    )
    with substrate:
        result = substrate.query(
            module='System',
            storage_function='Account',
            params=['5CZs3T15Ky4jch1sUpSFwkUbYEnsCfe1WCY51fH3SPV6NFnf']
        )

        print(result)


main()
```

```python
import asyncio
from async_substrate_interface import AsyncSubstrateInterface


async def main():
    substrate = AsyncSubstrateInterface(
        url="wss://rpc.polkadot.io"
    )
    async with substrate:
        result = await substrate.query(
            module='System',
            storage_function='Account',
            params=['5CZs3T15Ky4jch1sUpSFwkUbYEnsCfe1WCY51fH3SPV6NFnf']
        )

        print(result)


asyncio.run(main())
```

### Caching

There are a few different cache types used in this library to improve the performance overall. The one with which
you are probably familiar is the typical `functools.lru_cache` used in `sync_substrate.SubstrateInterface`.

By default, it uses a max cache size of 512 for smaller returns, and 16 for larger ones. These cache sizes are
user-configurable using the respective env vars, `SUBSTRATE_CACHE_METHOD_SIZE` and `SUBSTRATE_RUNTIME_CACHE_SIZE`.

They are applied only on methods whose results cannot change — such as the block hash for a given block number
(small, 512 default), or the runtime for a given runtime version (large, 16 default).

Additionally, in `AsyncSubstrateInterface`, because of its asynchronous nature, we developed our own asyncio-friendly
LRU caches. The primary one is the `CachedFetcher` which wraps the same methods as `functools.lru_cache` does in
`SubstrateInterface`, but the key difference here is that each request is assigned a future that is returned when the
initial request completes. So, if you were to do:

```python
bn = 5000
bh1, bh2 = await asyncio.gather(
    asi.get_block_hash(bn),
    asi.get_block_hash(bn)
)
```

it would actually only make one single network call, and return the result to both requests. Like `SubstrateInterface`,
it also takes the `SUBSTRATE_CACHE_METHOD_SIZE` and `SUBSTRATE_RUNTIME_CACHE_SIZE` vars to set cache size.

The third and final caching mechanism we use is
`async_substrate_interface.async_substrate.DiskCachedAsyncSubstrateInterface`,
which functions the same as the normal `AsyncSubstrateInterface`, but that also saves this cache to the disk, so the
cache
is preserved between runs. This is product for a fairly nice use-case (such as `btcli`). As you may call different
networks
with entirely different results, this cache is keyed by the uri supplied at instantiation of the
`DiskCachedAsyncSubstrateInterface`
object, so `DiskCachedAsyncSubstrateInterface(network_1)` and `DiskCachedAsyncSubstrateInterface(network_2)` will not
share the same on-disk cache.

As with the other two caches, this also takes `SUBSTRATE_CACHE_METHOD_SIZE` and `SUBSTRATE_RUNTIME_CACHE_SIZE` env vars.

### ENV VARS

The following environment variables are used within async-substrate-interface

- NO_CACHE (default 0): if set to 1, when using the DiskCachedAsyncSubstrateInterface class, no persistent on-disk cache
  will be stored, instead using only in-memory cache.
- CACHE_LOCATION (default `~/.cache/async-substrate-interface`): this determines the location for the cache file, if
  using DiskCachedAsyncSubstrateInterface
- SUBSTRATE_CACHE_METHOD_SIZE (default 512): the cache size (either in-memory or on-disk) of the smaller return-size
  methods (see the Caching section for more info)
- SUBSTRATE_RUNTIME_CACHE_SIZE (default 16): the cache size (either in-memory or on-disk) of the larger return-size
  methods (see the Caching section for more info)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request to the `staging` branch.

### Signed Commits

All commits in pull requests must be signed. We require signed commits to verify the authenticity of contributions and
ensure code integrity.

To sign your commits, you must have GPG signing configured in Git:

```bash
git commit -S -m "your commit message"
```

Or configure Git to sign all commits automatically:

```bash
git config --global commit.gpgsign true
```

For instructions on setting up GPG key signing,
see [GitHub's documentation on signing commits](https://docs.github.com/en/authentication/managing-commit-signature-verification/signing-commits).

> **Note:** Pull requests containing unsigned commits will not be merged.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or inquiries, please join the Bittensor Development Discord
server: [Church of Rao](https://discord.gg/XC7ucQmq2Q).
