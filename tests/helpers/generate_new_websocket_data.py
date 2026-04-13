"""
This is a script whose purpose is to generate new stub data for given commands for use in the integration tests.

The integration tests rely on actual websocket sends/responses, which have been more-or-less manually entered.

The Async Substrate Interface package includes a raw websocket logger [logging.getLogger("raw_websocket")] which will
be used to gather this data. It is imperative that this script only uses the SubstrateInterface class, as sorting the
sent/received data from an asynchronous manner will be significantly more difficult to parse (though it's doable by
checking IDs if we ever would absolutely need to.

Notes:
 - received websocket responses begin with `WEBSOCKET_RECEIVE> `
 - sent websocket begin with `WEBSOCKET_SEND> `
 - both are stringified JSON
 - logging level is DEBUG
 - metadata and metadataV15 (metadata at version) must be discarded, or rather just dumped to their respective txt files
 - metadata/metadataV15 txt files are just the "result" portion of the response:
    e.g. `{"jsonrpc": "2.0", "id": _id, "result": METADATA}`
 - This script should NOT overwrite "retry_archive", as that uses a specific cycling mechanism to cycle between possible
    runtime responses
 - the data is structured as follows:
    seed_name: {
      rpc_method: {
        params: {
          response
        }
      }
    }

 - seed_name is basically just the name of the test that is being run. Specifying the seed name tells the FakeWebsocket
    which set of data to use
 - some workflows may specify the same parameters for a given RPC call, and expect different results. This is alleviated
    typically by specifying block hashes/numbers, though that does not always fit well into the tests.
 - the tests should be short and concise, testing just a single "thing" because of the possibility of conflicting rpc
    requests with the same params
 - all requests include {"json_rpc": "2.0"} in them — this can be removed for the sake of saving space, but is not
    imperative to do so
 - the websocket logger will include the id of the request: these should be stripped away, as they are dynamically
    created in SubstrateInterface, and then attached to the next response by FakeWebsocket
"""

import os
from typing import Any, Callable

# Not really necessary, but doesn't hurt to have
os.environ["SUBSTRATE_CACHE_METHOD_SIZE"] = "0"
os.environ["SUBSTRATE_RUNTIME_CACHE_SIZE"] = "0"

import logging
import json
import subprocess
import pathlib

from async_substrate_interface.sync_substrate import (
    raw_websocket_logger,
    SubstrateInterface,
)

from tests.helpers.settings import ARCHIVE_ENTRYPOINT


RAW_WS_LOG = "/tmp/bittensor-raw-ws.log"
OUTPUT_DIR = "/tmp/bittensor-ws-output.txt"
OUTPUT_METADATA = "/tmp/integration_websocket_metadata.txt"
OUTPUT_METADATA_V15 = "/tmp/integration_websocket_at_version.txt"
INTEGRATION_WS_DATA = pathlib.Path(__file__).parent / "integration_websocket_data.py"


def main(seed: str, method: Callable[[SubstrateInterface], Any]):
    """
    Runs the given method on Subtensor, processes the websocket data that occurred during that method's execution,
    attaches it with the "seed" arg as a key to a new tmp file ("/tmp/bittensor-ws-output.txt")

    Existing keys within `integration_websocket_data.py` are updated automatically. New items should be manually
    copy-pasted in, as we may want to arrange them in some certain way.

    The metadata and metadataV15 are dumped to the same `/tmp` dir, but with their respective txt names, as exist in
    `bittensor/tests/helpers/`: `integration_websocket_metadata.txt` and `integration_websocket_at_version.txt`

    """
    if os.path.isfile(RAW_WS_LOG):
        os.remove(RAW_WS_LOG)
    if os.path.isfile(OUTPUT_DIR):
        os.remove(OUTPUT_DIR)

    raw_websocket_logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(RAW_WS_LOG)
    handler.setLevel(logging.DEBUG)
    raw_websocket_logger.addHandler(handler)

    substrate = SubstrateInterface(
        ARCHIVE_ENTRYPOINT,
        chain_name="Bittensor",
        ss58_format=42,
        _log_raw_websockets=True,
    )
    result = method(substrate)
    print(result)

    substrate.close()
    raw_websocket_logger.removeHandler(handler)

    with open(RAW_WS_LOG, "r") as f:
        all_ws_data = f.readlines()

    metadata = None
    metadataV15 = None

    output_dict = {seed: {}}
    output_dict_at_seed = output_dict[seed]
    upcoming_metadata = False
    upcoming_metadataV15 = False

    for l in all_ws_data:
        if l.startswith("WEBSOCKET_SEND> "):
            data = json.loads(l[len("WEBSOCKET_SEND> ") :])
            del data["jsonrpc"]
            del data["id"]
            send_method = data["method"]
            if send_method == "state_getMetadata":
                upcoming_metadata = True
                continue
            send_params = json.dumps(data["params"])
            if (
                send_method == "state_call"
                and "Metadata_metadata_at_version" in send_params
            ):
                upcoming_metadataV15 = True
                continue
            if send_method in output_dict_at_seed.keys():
                output_dict_at_seed[send_method][send_params] = {}
            else:
                output_dict_at_seed[send_method] = {send_params: {}}
        elif l.startswith("WEBSOCKET_RECEIVE> "):
            data = json.loads(l[len("WEBSOCKET_RECEIVE> ") :])
            if upcoming_metadata:
                upcoming_metadata = False
                metadata = data["result"]
                continue
            elif upcoming_metadataV15:
                upcoming_metadataV15 = False
                metadataV15 = data["result"]
                continue
            del data["id"]
            del data["jsonrpc"]
            try:
                output_dict_at_seed[send_method][send_params] = data
            except (NameError, KeyError):
                raise KeyError(
                    f"Attempting to add a received value before its keys have been added: {l}"
                )

    with open(OUTPUT_DIR, "w+") as f:
        f.write(str(output_dict))
    subprocess.run(["ruff", "format", OUTPUT_DIR])
    if metadata is not None:
        with open(OUTPUT_METADATA, "w+") as f:
            f.write(metadata)
    if metadataV15 is not None:
        with open(OUTPUT_METADATA_V15, "w+") as f:
            f.write(metadataV15)

    with open(INTEGRATION_WS_DATA, "r") as f:
        # Read the current integration_websocket_data.py file
        all_integration_ws_data = f.readlines()
    watching = False
    start_idx = 0
    end_idx = 0
    # look for the line of the dict matching the seed whose data we want to update
    sought_line = f'    "{seed}": ' + "{\n"
    for line_idx, line in enumerate(all_integration_ws_data):
        if watching:
            if line == "    },\n":
                # the part of the dict matching the end of the bit of data we need to update for this seed
                end_idx = line_idx
                break
        if line == sought_line:
            watching = True
            start_idx = line_idx
    if start_idx == 0 or end_idx == 0:
        if start_idx == 0:
            last_entry = None
            # new seed key, should be appended to the bottom
            for line_idx, line in enumerate(all_integration_ws_data):
                # because this is is a loop, it will replace til the end of the file
                if line == "    },\n":
                    last_entry = line_idx
            if last_entry is not None:
                insertion_point = last_entry + 1
                all_integration_ws_data.insert(insertion_point, "")
                start_idx = insertion_point
                end_idx = insertion_point
        else:
            print(
                f"Unable to find seed {seed} in current `websocket_integration_data.py` file. You should manually add this"
                f" new seed key."
            )
            return
    # only retain the portions of the file before and after the seed we want to update
    first_part = all_integration_ws_data[:start_idx]
    last_part = all_integration_ws_data[end_idx + 1 :]
    with open(OUTPUT_DIR, "r") as f:
        # read the **ruff formatted** output of this seed's new data
        formatted_output = f.readlines()
    foutput_len = len(formatted_output)
    insertion_data = []
    for line_idx, line in enumerate(formatted_output):
        if line_idx == 0 and line.startswith("{"):
            # remove the first { as we want to insert this dict into another dict (WEBSOCKET_DATA)
            line = line[1:]
        elif line_idx == foutput_len - 1:
            # same with the end of this dict
            line = line.replace("}", ",")
        insertion_data.append(line)
    with open(INTEGRATION_WS_DATA, "w") as f:
        # rewrite the corrected file with our data inserted into its proper place
        f.writelines(first_part + insertion_data + last_part)
    subprocess.run(
        ["ruff", "format", INTEGRATION_WS_DATA]
    )  # ruff format again for good measure


if __name__ == "__main__":
    # Example usage
    def fn_(substrate: SubstrateInterface) -> Any:
        block = 7959635
        block_hash = substrate.get_block_hash(block)
        print(block_hash)
        all_events = substrate.get_events(block_hash=block_hash)
        event = all_events[1]
        print(event)
        # assert event["attributes"] == (
        #     "5G1NjW9YhXLadMWajvTkfcJy6up3yH2q1YzMXDTi6ijanChe",
        #     30,
        #     "0xa6b4e5c8241d60ece0c25056b19f7d21ae845269fc771ad46bf3e011865129a5",
        # )

    main("test_get_events_proper_decoding", fn_)
