# Changelog

## 2.0.2 /2025-04-24

## What's Changed

* use cyscale 0.3.1 by @thewhaleking in https://github.com/latent-to/async-substrate-interface/pull/332

**Full Changelog**: https://github.com/latent-to/async-substrate-interface/compare/v2.0.1...v2.0.2

## 2.0.1 /2025-04-23

## What's Changed

* Bump aiohttp from 3.13.3 to 3.13.4 by @dependabot[bot]
  in https://github.com/latent-to/async-substrate-interface/pull/325
* Improves the scalecodec conflict detected message by @thewhaleking
  in https://github.com/latent-to/async-substrate-interface/pull/329
* Restricts the pickling to certain classes by @thewhaleking
  in https://github.com/latent-to/async-substrate-interface/pull/328

## New Contributors

* @dependabot[bot] made their first contribution in https://github.com/latent-to/async-substrate-interface/pull/325

**Full Changelog**: https://github.com/latent-to/async-substrate-interface/compare/v2.0.0...v2.0.1

## 2.0.0 /2025-04-21

### Breaking changes:

- SS58 format now defaults to 42 instead of None. It can still be manually set to `None`, which will trigger the chain
  pull.

- query now always returns a ScaleType object. Previously it could return ScaleObj | None | dict. The underlying data
  has not changed, but always needs to be retrieved by the `.value` (or `.value_object` or `.value_serialized`).

- runtime_call now always returns the decoded value. Previously it could return the decoded object, or a ScaleObj. This
  was done for consistency, as legacy methods (pre-v15 metadata) required their own predefined decoders, and the
  decoders did not give a ScaleType object, so to keep things consistent, it's always the decoded value.

- query_multi previously returned `[(StorageKey, ScaleType), ...]`, now returns `[(StorageKey, decoded value), ...]`. We
  made significant speed improvements in cyscale with batch-decoding, and this works great here.

- query_map.records previously returned `[(key, ScaleObj), ...]`. Now returns `[(key, decoded value), ...]` for the same
  reasons as listed in the query_multi note.

- query_multiple: removed, was not so useful

- reuse_block_hash: removed, was not so useful

- SS58 addresses are always decoded now. No more need to manually decode after retrieving results.

- For instantiation of any of the *Substrate classes, only the uri is a positional argument. All other arguments are
  keyword-only.

### Non-breaking improvements:

- Better DiskCached behavior. Loading from cache is 2-3x faster, and dumping to cache 3-∞ faster (does not dump if the
  object is unchanged).

- Mypy support added.

### Important for upgrade:

Because of the change from the combination py-scale-codec and bt-decode to cyscale, we no longer support having the
py-scale-codec installed (cyscale fits the same namespace, `scalecodec`). If you attempt to run ASI with py-scale-codec
installed, you will receive an error and instructions to uninstall that first. To prevent this, before upgrade, you can
remove it as such:

```sh
pip uninstall scalecodec -y
pip install -U cyscale 
```

## 1.6.4 /2025-03-25

## What's Changed

* Better typing for ScaleObj by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/278
* Faster startup by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/277
* DNS/SSL Caching by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/279
* Added info about signed commits by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/280
* [fix] change legacy (old) runtimeApi params encoding by @camfairchild
  in https://github.com/opentensor/async-substrate-interface/pull/194
* Prefer v15 metadata when available by @thewhaleking
  in https://github.com/opentensor/async-substrate-interface/pull/282

**Full Changelog**: https://github.com/opentensor/async-substrate-interface/compare/v1.6.3...v1.6.4

## 1.6.3 /2025-02-24

## What's Changed

* Fix: Supp for third-party runtimes by @ibraheem-abe
  in https://github.com/opentensor/async-substrate-interface/pull/274
* Extend `get_account_next_index` logic by @basfroman
  in https://github.com/opentensor/async-substrate-interface/pull/273

**Full Changelog**: https://github.com/opentensor/async-substrate-interface/compare/v1.6.2...v1.6.3

## 1.6.2 /2025-02-19

## What's Changed

* Typing by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/265
* Cache Improvements by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/267
* improve (async) query map result by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/266
* Use threaded bt-decode by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/268
* Feat: Handle new error message fmt for InBlock source by @ibraheem-abe
  in https://github.com/opentensor/async-substrate-interface/pull/269
* Update: Remove python 3.9 support by @ibraheem-abe in https://github.com/opentensor/async-substrate-interface/pull/271

**Full Changelog**: https://github.com/opentensor/async-substrate-interface/compare/v1.6.1...v1.6.2

## 1.6.1 /2025-02-03

* RuntimeCache updates by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/260
* fix memory leak by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/261
* Avoid Race Condition on SQLite Table Creation by @thewhaleking
  in https://github.com/opentensor/async-substrate-interface/pull/263

**Full Changelog**: https://github.com/opentensor/async-substrate-interface/compare/v1.6.0...v1.6.1

## 1.6.0 /2025-01-27

* Fix typo by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/258
* Improve Disk Caching by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/227

**Full Changelog**: https://github.com/opentensor/async-substrate-interface/compare/v1.5.15...v1.6.0

## 1.5.15 /2025-12-22

* Modifies the CachedFetcher to not keep pending exceptions by @thewhaleking
  in https://github.com/opentensor/async-substrate-interface/pull/253

**Full Changelog**: https://github.com/opentensor/async-substrate-interface/compare/v1.5.14...v1.5.15

## 1.5.14 /2025-12-04

* Update: Adds support for custom extrinsic `DecryptionFailed` by @ibraheem-abe
  in https://github.com/opentensor/async-substrate-interface/pull/247

**Full Changelog**: https://github.com/opentensor/async-substrate-interface/compare/v1.5.13...v1.5.14

## 1.5.13 /2025-12-01

* Update `Checkout PR branch in async-substrate-interface` step by @basfroman
  in https://github.com/opentensor/async-substrate-interface/pull/240
* No continual reconnection without cause by @thewhaleking
  in https://github.com/opentensor/async-substrate-interface/pull/241
* Feat: Add support for MeV shield extrinsics by @ibraheem-abe
  in https://github.com/opentensor/async-substrate-interface/pull/242
* Handle subscription failures from substrate by @thewhaleking
  in https://github.com/opentensor/async-substrate-interface/pull/243

**Full Changelog**: https://github.com/opentensor/async-substrate-interface/compare/v1.5.12...v1.5.13

## 1.5.12 /2025-11-17

* RecursionError in `_wait_with_activity_timeout` with concurrent tasks by @Arthurdw
  in https://github.com/opentensor/async-substrate-interface/pull/238
* Improved Test Running + Race Condition Catch by @thewhaleking
  in https://github.com/opentensor/async-substrate-interface/pull/236

**Full Changelog**: https://github.com/opentensor/async-substrate-interface/compare/v1.5.11...v1.5.12

## 1.5.11 /2025-11-14

* Race Condition Bug fixes by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/234

**Full Changelog**: https://github.com/opentensor/async-substrate-interface/compare/v1.5.10...v1.5.11

## 1.5.10 /2025-11-12

* bug fixes 1.5.10 by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/231
    * no double-sleep in async-substrate-interface websocket querying
    * KeyError catching on websocket race conditions
    * corrected state check on disconnecting
    * reset ws attempts on close

**Full Changelog**: https://github.com/opentensor/async-substrate-interface/compare/v1.5.9...v1.5.10

## 1.5.9 /2025-10-29

* Adds metadata call functions retrieval by @thewhaleking
  in https://github.com/opentensor/async-substrate-interface/pull/223
* move metadata methods to SubstrateMixin by @thewhaleking
  in https://github.com/opentensor/async-substrate-interface/pull/224
* Update get_payment_info to include addl params by @thewhaleking
  in https://github.com/opentensor/async-substrate-interface/pull/226
* Python 3.14 by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/228
* Support python 3.14 by @Moisan in https://github.com/opentensor/async-substrate-interface/pull/210

## New Contributors

* @Moisan made their first contribution in https://github.com/opentensor/async-substrate-interface/pull/210

**Full Changelog**: https://github.com/opentensor/async-substrate-interface/compare/v1.5.8...v1.5.9

## 1.5.8 /2025-10-21

* Fix parameter name conflict in retry substrate _retry() methods by @Arthurdw
  in https://github.com/opentensor/async-substrate-interface/pull/218
* Use uv for test dependencies by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/219
* Reconnection/Resubmission Logic Improved by @thewhaleking
  in https://github.com/opentensor/async-substrate-interface/pull/217

## New Contributors

* @Arthurdw made their first contribution in https://github.com/opentensor/async-substrate-interface/pull/218

**Full Changelog**: https://github.com/opentensor/async-substrate-interface/compare/v1.5.7...v1.5.8

## 1.5.7 /2025-10-15

* Updates the type hint on ws_shutdown_timer in RetryAsyncSubstrate by @thewhaleking
  in https://github.com/opentensor/async-substrate-interface/pull/203
* correct type hint by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/204
* Clear asyncio.Queue after retrieval by @thewhaleking
  in https://github.com/opentensor/async-substrate-interface/pull/206
* Add the option to manually specify the Bittensor branch when running with `workflow_dispatch` by @basfroman
  in https://github.com/opentensor/async-substrate-interface/pull/208
* Subscription Exception Handling by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/207
* more efficient query map by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/211
* Unique keys in request manager by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/212
* Adds type annotations for Runtime by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/214
* Edge case ss58 decoding in decode_query_map by @thewhaleking
  in https://github.com/opentensor/async-substrate-interface/pull/213

**Full Changelog**: https://github.com/opentensor/async-substrate-interface/compare/v1.5.6...v1.5.7

## 1.5.6 /2025-10-08

* Clean Up Error Handling by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/193
* Avoids ID of 'None' in queries by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/196
* Allows AsyncSubstrateInterface's Websocket connection to not automatically shut down by @thewhaleking
  in https://github.com/opentensor/async-substrate-interface/pull/197
* return type annotation for `get_metadata_call_function` by @thewhaleking
  in https://github.com/opentensor/async-substrate-interface/pull/199
* Change responses["results"] to deque by @thewhaleking
  in https://github.com/opentensor/async-substrate-interface/pull/198
* do not attempt to reconnect if there are open subscriptions by @thewhaleking
  in https://github.com/opentensor/async-substrate-interface/pull/200

**Full Changelog**: https://github.com/opentensor/async-substrate-interface/compare/v1.5.5...v1.5.6

## 1.5.5 /2025-10-06

* Improve timeout task cancellation by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/190

**Full Changelog**: https://github.com/opentensor/async-substrate-interface/compare/v1.5.4...v1.5.5

## 1.5.4 /2025-09-23

* Raw Websocket Logger Inconsistency Fix by @thewhaleking
  in https://github.com/opentensor/async-substrate-interface/pull/188

**Full Changelog**: https://github.com/opentensor/async-substrate-interface/compare/v1.5.3...v1.5.4

## 1.5.3 /2025-09-16

* edge case query map keys by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/186

**Full Changelog**: https://github.com/opentensor/async-substrate-interface/compare/v1.5.2...v1.5.3

## 1.5.2 /2025-09-08

* Improve test workflows by @basfroman in https://github.com/opentensor/async-substrate-interface/pull/173
* Adds env var support for setting cache size by @thewhaleking
  in https://github.com/opentensor/async-substrate-interface/pull/174
* Set env vars as str in unit test by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/177
* DiskCachedAsyncSubstrateInterface: use aiosqlite by @thewhaleking
  in https://github.com/opentensor/async-substrate-interface/pull/176
* Additional Debug Logging by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/178
* None type edge case catch by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/184

**Full Changelog**: https://github.com/opentensor/async-substrate-interface/compare/v1.5.1...v1.5.2

## 1.5.1 /2025-08-05

* query multiple/decoding fix by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/168
* Fix reconnection logic by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/169

**Full Changelog**: https://github.com/opentensor/async-substrate-interface/compare/v1.5.0...v1.5.1

## 1.5.0 /2025-08-04

* ConcurrencyError fix by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/162
* Added better typing by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/163
* Fix arg order in retries by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/165
    * removes "bool object has no attribute Metadata" errors
* Concurrency improvements by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/164
    * True Runtime independence in AsyncSubstrateInterface:
        * ensures no need to reload types from a shared object that may interfere with concurrency
        * increases memory usage slightly, but drops CPU usage dramatically by not needing to reload the type registry
          when retrieving from cache
    * RuntimeCache improved to automatically add additional mappings
    * Uses a single dispatcher queue for concurrent sending/receiving which eliminates the need for coroutines to manage
      their own state in regard to connection management.
    * Futures from the Websocket now get assigned their own exceptions
    * Overall cleaner logic flow with regard to rpc requests
    * The Websocket object now handles reconnections/timeouts
    * Separation of normal ping-pong calls and longer-running subscriptions

**Full Changelog**: https://github.com/opentensor/async-substrate-interface/compare/v1.4.3...v1.5.0

## 1.4.3 /2025-07-28

* Add "Token" to caught error messages for extrinsic receipts by @thewhaleking
  in https://github.com/opentensor/async-substrate-interface/pull/156
* runtime version switching by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/157

**Full Changelog**: https://github.com/opentensor/async-substrate-interface/compare/v1.4.2...v1.4.3

## 1.4.2 /2025-07-23

* Use scalecodec rather than bt-decode for query_multi by @thewhaleking
  in https://github.com/opentensor/async-substrate-interface/pull/152

**Full Changelog**: https://github.com/opentensor/async-substrate-interface/compare/v1.4.1...v1.4.2

## 1.4.1 /2025-07-09

* Missed passing runtime in encoding by @thewhaleking
  in https://github.com/opentensor/async-substrate-interface/pull/149

**Full Changelog**: https://github.com/opentensor/async-substrate-interface/compare/v1.4.0...v1.4.1

## 1.4.0 /2025-07-07

* Removes unused imports by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/139
* Improve CachedFetcher by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/140
* Only use Runtime objects in AsyncSubstrateInterface by @thewhaleking
  in https://github.com/opentensor/async-substrate-interface/pull/141
* python ss58 conversion by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/143
* fully exhaust query map by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/144
* Only use v14 decoding for events by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/145

**Full Changelog**: https://github.com/opentensor/async-substrate-interface/compare/v1.3.1...v1.4.0

## 1.3.1 /2025-06-11

* Fixes default vals for archive_nodes by @thewhaleking
  in https://github.com/opentensor/async-substrate-interface/pull/134
* Adds ability to log raw websockets for debugging. by @thewhaleking
  in https://github.com/opentensor/async-substrate-interface/pull/133

**Full Changelog**: https://github.com/opentensor/async-substrate-interface/compare/v1.3.0...v1.3.1

## 1.3.0 /2025-06-10

* Add GH test runner by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/129
* Edge Case Fixes by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/127
* Add archive node to retry substrate by @thewhaleking
  in https://github.com/opentensor/async-substrate-interface/pull/128

**Full Changelog**: https://github.com/opentensor/async-substrate-interface/compare/v1.2.2...v1.3.0

## 1.2.2 /2025-05-22

## What's Changed

* Add proper mock support by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/123
* Handle Incorrect Timeouts by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/124

**Full Changelog**: https://github.com/opentensor/async-substrate-interface/compare/v1.2.1...v1.2.2

## 1.2.1 /2025-05-12

## What's Changed

* Remove testing print calls. by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/117
* Fix name shadowing by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/118

**Full Changelog**: https://github.com/opentensor/async-substrate-interface/compare/v1.2.0...v1.2.1

## 1.2.0 /2025-05-07

## What's Changed

* Add missing methods by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/104
* Max subscriptions semaphore added by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/107
* Expose `_get_block_handler` publicly by @thewhaleking
  in https://github.com/opentensor/async-substrate-interface/pull/108
* safe `__del__` by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/110
* Tensorshield/main by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/111
* Support async key implementations by @immortalizzy in https://github.com/opentensor/async-substrate-interface/pull/94
* Add MetadataAtVersionNotFound error by @thewhaleking
  in https://github.com/opentensor/async-substrate-interface/pull/113
* Fallback chains by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/100

## New Contributors

* @immortalizzy made their first contribution in https://github.com/opentensor/async-substrate-interface/pull/94

**Full Changelog**: https://github.com/opentensor/async-substrate-interface/compare/v1.1.1...v1.2.0

## 1.1.1 /2025-04-26

## What's Changed

* State-Safe RNG by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/97
* Fix tests requirements by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/98
* Update maintainers emails by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/99
* Adds additional exception for catching by @thewhaleking
  in https://github.com/opentensor/async-substrate-interface/pull/96

**Full Changelog**: https://github.com/opentensor/async-substrate-interface/compare/v1.1.0...v1.1.1

## 1.1.0 /2025-04-07

## What's Changed

* Fix: response is still missing for callback by @zyzniewski-reef
  in https://github.com/opentensor/async-substrate-interface/pull/90
* Expose websockets exceptions by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/91
* Improved Query Map Decodes by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/84

**Full Changelog**: https://github.com/opentensor/async-substrate-interface/compare/v1.0.9...v1.1.0

## 1.0.9 /2025-03-26

## What's Changed

* Add workflows for run SDK and BTCLI tests if labels are applied by @roman-opentensor
  in https://github.com/opentensor/async-substrate-interface/pull/83
* Update docker image name by @roman-opentensor in https://github.com/opentensor/async-substrate-interface/pull/85
* Updates `_load_registry_type_map` by @ibraheem-opentensor
  in https://github.com/opentensor/async-substrate-interface/pull/87

**Full Changelog**: https://github.com/opentensor/async-substrate-interface/compare/v1.0.8...v1.0.9

## 1.0.8 /2025-03-17

## What's Changed

* Allows installing on Python 3.13 by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/79
* Support Option types by @ibraheem-opentensor in https://github.com/opentensor/async-substrate-interface/pull/80

**Full Changelog**: https://github.com/opentensor/async-substrate-interface/compare/v1.0.7...v1.0.8

## 1.0.7 /2025-03-12

## What's Changed

* Improves the logic of the disk cache so that it doesn't spill over by @thewhaleking
  in https://github.com/opentensor/async-substrate-interface/pull/76

**Full Changelog**: https://github.com/opentensor/async-substrate-interface/compare/v1.0.6...v1.0.7

## 1.0.6 /2025-03-12

## What's Changed

* On-disk cache by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/67

**Full Changelog**: https://github.com/opentensor/async-substrate-interface/compare/v1.0.5...v1.0.6

## 1.0.5 /2025-03-06

## What's Changed

* Fixes a memory leak by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/70
* Backmerge main to staging 104 by @ibraheem-opentensor
  in https://github.com/opentensor/async-substrate-interface/pull/71

**Full Changelog**: https://github.com/opentensor/async-substrate-interface/compare/v1.0.4...v1.0.5

## 1.0.4 /2025-03-05

## What's Changed

* Warn users about too old blocks by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/60
* Runtime version fixes by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/65
* Feat/mvds00/runtime version fixes by @mvds00 in https://github.com/opentensor/async-substrate-interface/pull/63
* Backmerge main to staging 103 by @ibraheem-opentensor
  in https://github.com/opentensor/async-substrate-interface/pull/66

## New Contributors

* @mvds00 made their first contribution in https://github.com/opentensor/async-substrate-interface/pull/63

**Full Changelog**: https://github.com/opentensor/async-substrate-interface/compare/v1.0.3...v1.0.4

## 1.0.3 /2025-02-20

## What's Changed

* Refactor generate_unique_id by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/56
* Backmerge main to staging 103 by @ibraheem-opentensor
  in https://github.com/opentensor/async-substrate-interface/pull/57

**Full Changelog**: https://github.com/opentensor/async-substrate-interface/compare/v1.0.1...v1.0.3

## 1.0.2 /2025-02-19

## What's Changed

* Closes the connection on the object being garbage-collected by @thewhaleking
  in https://github.com/opentensor/async-substrate-interface/pull/51
* Generate UIDs for websockets by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/50
* Dynamically pulls the info for Vec<AccountId> from the metadata by @thewhaleking
  in https://github.com/opentensor/async-substrate-interface/pull/47
* Fix readme by @igorsyl in https://github.com/opentensor/async-substrate-interface/pull/46
* Handle options with bt-decode by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/52
* Backmerge main to staging 101 by @ibraheem-opentensor
  in https://github.com/opentensor/async-substrate-interface/pull/53
* Handles None change_data by @ibraheem-opentensor in https://github.com/opentensor/async-substrate-interface/pull/54

## New Contributors

* @igorsyl made their first contribution in https://github.com/opentensor/async-substrate-interface/pull/46

**Full Changelog**: https://github.com/opentensor/async-substrate-interface/compare/v1.0.1...v1.0.2

## 1.0.1 /2025-02-17

## What's Changed

* Updates type for vec acc id by @ibraheem-opentensor in https://github.com/opentensor/async-substrate-interface/pull/45
* Backmerge main staging 101 by @ibraheem-opentensor in https://github.com/opentensor/async-substrate-interface/pull/48

**Full Changelog**: https://github.com/opentensor/async-substrate-interface/compare/v1.0.0...v1.0.1

## 1.0.0 /2025-02-13

## What's new

* New Async Substrate Interface by @thewhaleking and @roman-opentensor
  in https://github.com/opentensor/async-substrate-interface/tree/main
* Github release + bumps version by @ibraheem-opentensor
  in https://github.com/opentensor/async-substrate-interface/pull/1
* Improve `ScaleObj`  by @roman-opentensor in https://github.com/opentensor/async-substrate-interface/pull/2
* Backmerge staging main by @ibraheem-opentensor in https://github.com/opentensor/async-substrate-interface/pull/4
* Release/1.0.0rc2 by @ibraheem-opentensor in https://github.com/opentensor/async-substrate-interface/pull/5
* EventLoopManager, factory function by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/3
* Adds nonce implementation by @ibraheem-opentensor in https://github.com/opentensor/async-substrate-interface/pull/8
* Exception for uninitialised by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/6
* Update build/release to use pyproject.toml by @thewhaleking
  in https://github.com/opentensor/async-substrate-interface/pull/10
* Fixes nonce management & bumps version by @ibraheem-opentensor
  in https://github.com/opentensor/async-substrate-interface/pull/11
* Sync Substrate Rewritten by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/9
* Remove ujson by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/12
* Backmerge main to staging rc4 by @ibraheem-opentensor
  in https://github.com/opentensor/async-substrate-interface/pull/13
* Release/1.0.0rc5 by @ibraheem-opentensor in https://github.com/opentensor/async-substrate-interface/pull/14
* Update project name for PyPI by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/16
* Fixes _metadata_cache, bumps version and changelog by @ibraheem-opentensor
  in https://github.com/opentensor/async-substrate-interface/pull/17
* feat: use bt_decode in runtime_call by @zyzniewski-reef
  in https://github.com/opentensor/async-substrate-interface/pull/15
* Move logic to mixin + fix tests by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/18
* Fix decode scale by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/19
* Backmerge main to staging rc5 by @ibraheem-opentensor
  in https://github.com/opentensor/async-substrate-interface/pull/20
* Release/1.0.0rc7 by @ibraheem-opentensor in https://github.com/opentensor/async-substrate-interface/pull/21
* Release/1.0.0rc8 by @ibraheem-opentensor in https://github.com/opentensor/async-substrate-interface/pull/22
* Fixes decoding acc ids by @ibraheem-opentensor in https://github.com/opentensor/async-substrate-interface/pull/23
* Backmerge/1.0.0rc8 by @ibraheem-opentensor in https://github.com/opentensor/async-substrate-interface/pull/24
* Release/1.0.0rc9 by @ibraheem-opentensor in https://github.com/opentensor/async-substrate-interface/pull/25
* Fixes sync ss58 decoding by @ibraheem-opentensor in https://github.com/opentensor/async-substrate-interface/pull/26
* Backmerge main staging rc9 by @ibraheem-opentensor in https://github.com/opentensor/async-substrate-interface/pull/27
* Release/1.0.0rc10 by @ibraheem-opentensor in https://github.com/opentensor/async-substrate-interface/pull/28
* Reuses the websocket for sync Substrate by @thewhaleking
  in https://github.com/opentensor/async-substrate-interface/pull/29
* Feat/metadata v15 cache by @camfairchild in https://github.com/opentensor/async-substrate-interface/pull/30
* Backmerge main to staging rc10 by @ibraheem-opentensor
  in https://github.com/opentensor/async-substrate-interface/pull/31
* Release/1.0.0rc11 by @ibraheem-opentensor in https://github.com/opentensor/async-substrate-interface/pull/32
* python 3.9 support by @roman-opentensor in https://github.com/opentensor/async-substrate-interface/pull/33
* Backmerge main to staging RC11 by @ibraheem-opentensor
  in https://github.com/opentensor/async-substrate-interface/pull/34
* Release/1.0.0rc12 by @ibraheem-opentensor in https://github.com/opentensor/async-substrate-interface/pull/35
* Improve logging by @roman-opentensor in https://github.com/opentensor/async-substrate-interface/pull/36
* Backmerge main to staging rc12 by @ibraheem-opentensor
  in https://github.com/opentensor/async-substrate-interface/pull/37
* Release/1.0.0rc13 by @ibraheem-opentensor in https://github.com/opentensor/async-substrate-interface/pull/38
* Improves the error-handling of initialisation of the object by @thewhaleking
  in https://github.com/opentensor/async-substrate-interface/pull/39
* Backmerge main to staging rc12 by @ibraheem-opentensor
  in https://github.com/opentensor/async-substrate-interface/pull/40
* Release/1.0.0rc14 by @ibraheem-opentensor in https://github.com/opentensor/async-substrate-interface/pull/41

**Full Changelog**: https://github.com/opentensor/async-substrate-interface/commits/v1.0.0

## 1.0.0rc14 /2025-02-11

* Improves the error-handling of initialisation of the object @thewhaleking
  in https://github.com/opentensor/async-substrate-interface/pull/39
* Backmerge main to staging rc12 by @ibraheem-opentensor
  in https://github.com/opentensor/async-substrate-interface/pull/40

## 1.0.0rc13 /2025-02-10

* Improve logging by @roman-opentensor and @thewhaleking
  in https://github.com/opentensor/async-substrate-interface/pull/36
* Backmerge main to staging rc12 by @ibraheem-opentensor
  in https://github.com/opentensor/async-substrate-interface/pull/37

## 1.0.0rc12 /2025-02-07

* python 3.9 support by @roman-opentensor in https://github.com/opentensor/async-substrate-interface/pull/33

## 1.0.0rc11 /2025-02-06

* Reuses the websocket for sync Substrate by @thewhaleking
  in https://github.com/opentensor/async-substrate-interface/pull/29
* Feat/metadata v15 cache by @camfairchild in https://github.com/opentensor/async-substrate-interface/pull/30
* Backmerge main to staging rc10 by @ibraheem-opentensor
  in https://github.com/opentensor/async-substrate-interface/pull/31

## 1.0.0rc10 /2025-02-04

* Fixes decoding account ids for sync substrate

## 1.0.0rc9 /2025-01-30

* Fixes decoding account ids

## 1.0.0rc8 /2025-01-30

* Minor bug fixes

## 1.0.0rc7 /2025-01-29

## What's Changed

* feat: use bt_decode in runtime_call by @zyzniewski-reef
  in https://github.com/opentensor/async-substrate-interface/pull/15
* Move logic to mixin + fix tests by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/18
* Fix decode scale by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/19
* Backmerge main to staging rc5 by @ibraheem-opentensor
  in https://github.com/opentensor/async-substrate-interface/pull/20

## 1.0.0rc6 /2025-01-28

## What's Changed

* Minor bug fix

## 1.0.0rc5 /2025-01-28

## What's Changed

* Backmerge staging main by @ibraheem-opentensor in https://github.com/opentensor/async-substrate-interface/pull/4
* EventLoopManager, factory function by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/3
* Exception for uninitialised by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/6
* Update build/release to use pyproject.toml by @thewhaleking
  in https://github.com/opentensor/async-substrate-interface/pull/10
* Sync Substrate Rewritten by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/9
* Remove ujson by @thewhaleking in https://github.com/opentensor/async-substrate-interface/pull/12

## 1.0.0rc4 /2025-01-17

## What's Changed

* Minor bug fixes and improvements

## 1.0.0rc3 /2025-01-17

## What's Changed

* Adds nonce implementation @ibraheem-opentensor in https://github.com/opentensor/async-substrate-interface/pull/8

## 1.0.0rc2 /2025-01-15

## What's Changed

* Improve ScaleObj by @roman-opentensor in https://github.com/opentensor/async-substrate-interface/pull/2

## 1.0.0rc1 /2025-01-15

## What's Changed

* New Async Substrate Interface by @thewhaleking and @roman-opentensor
  in https://github.com/opentensor/async-substrate-interface/tree/main
