from collections import namedtuple
from typing import TYPE_CHECKING, Union

from scalecodec import ss58_decode

if TYPE_CHECKING:
    from async_substrate_interface.types import Runtime


def _cyscale_decode(type_name: str, raw_bytes: bytes, runtime: "Runtime"):
    """Decode raw_bytes as type_name using cyscale's portable registry."""
    type_id = runtime.registry_type_map.get(type_name)
    if type_id is None:
        raise ValueError(f"Type '{type_name}' not found in registry_type_map")
    return runtime.runtime_config.batch_decode([f"scale_info::{type_id}"], [raw_bytes])[
        0
    ]


def stake_info_decode_vec_legacy_compatibility(
    raw_bytes: bytes, runtime: "Runtime"
) -> list[dict[str, Union[str, int, bytes, bool]]]:
    stake_infos = _cyscale_decode("Vec<StakeInfo>", raw_bytes, runtime)
    NewStakeInfo = namedtuple(
        "NewStakeInfo",
        [
            "netuid",
            "hotkey",
            "coldkey",
            "stake",
            "locked",
            "emission",
            "drain",
            "is_registered",
        ],
    )
    return [
        NewStakeInfo(
            0,
            si["hotkey"],
            si["coldkey"],
            si["stake"],
            0,
            0,
            0,
            False,
        )
        for si in stake_infos
    ]


def preprocess_get_stake_info_for_coldkeys(addrs):
    output = []
    if isinstance(addrs[0], list):  # I think
        for addr in addrs[0]:
            output.append(list(bytes.fromhex(ss58_decode(addr))))
    else:
        if isinstance(addrs[0], dict):
            for addr in addrs[0]["coldkey_accounts"]:
                output.append(list(bytes.fromhex(ss58_decode(addr))))
    return output


def _encode_vec_u8(params, runtime: "Runtime") -> bytes:
    """Encode a single SS58 address as Vec<u8>."""
    addr = (
        params
        if isinstance(params, str)
        else params[0]
        if isinstance(params, list)
        else params["coldkey_account"]
    )
    raw = bytes.fromhex(ss58_decode(addr))
    return bytes(
        runtime.runtime_config.create_scale_object("Vec<u8>").encode(list(raw)).data
    )


def _encode_vec_u8_coldkey(params, runtime: "Runtime") -> bytes:
    """Encode a coldkey account as Vec<u8>, handling list or dict input."""
    if isinstance(params, list):
        addr = params[0]
    elif isinstance(params, dict):
        addr = params["coldkey_account"]
    else:
        addr = params
    raw = bytes.fromhex(ss58_decode(addr))
    return bytes(
        runtime.runtime_config.create_scale_object("Vec<u8>").encode(list(raw)).data
    )


def _encode_vec_vec_u8(params, runtime: "Runtime") -> bytes:
    """Encode multiple coldkey accounts as Vec<Vec<u8>>."""
    preprocessed = preprocess_get_stake_info_for_coldkeys(params)
    return bytes(
        runtime.runtime_config.create_scale_object("Vec<Vec<u8>>")
        .encode(preprocessed)
        .data
    )


_TYPE_REGISTRY: dict[str, dict] = {
    "types": {
        "Balance": "u64",  # Need to override default u128
    },
    "runtime_api": {
        "DelegateInfoRuntimeApi": {
            "methods": {
                "get_delegated": {
                    "params": [
                        {
                            "name": "coldkey",
                            "type": "Vec<u8>",
                        },
                    ],
                    "encoder": _encode_vec_u8,
                    "type": "Vec<u8>",
                    "decoder": lambda raw_bytes, runtime: _cyscale_decode(
                        "Vec<DelegateInfo>", raw_bytes, runtime
                    ),
                },
                "get_delegates": {
                    "params": [],
                    "type": "Vec<u8>",
                    "decoder": lambda raw_bytes, runtime: _cyscale_decode(
                        "Vec<DelegateInfo>", raw_bytes, runtime
                    ),
                },
            }
        },
        "NeuronInfoRuntimeApi": {
            "methods": {
                "get_neuron_lite": {
                    "params": [
                        {
                            "name": "netuid",
                            "type": "u16",
                        },
                        {
                            "name": "uid",
                            "type": "u16",
                        },
                    ],
                    "type": "Vec<u8>",
                    "decoder": lambda raw_bytes, runtime: _cyscale_decode(
                        "NeuronInfoLite", raw_bytes, runtime
                    ),
                },
                "get_neurons_lite": {
                    "params": [
                        {
                            "name": "netuid",
                            "type": "u16",
                        },
                    ],
                    "type": "Vec<u8>",
                    "decoder": lambda raw_bytes, runtime: _cyscale_decode(
                        "Vec<NeuronInfoLite>", raw_bytes, runtime
                    ),
                },
                "get_neuron": {
                    "params": [
                        {
                            "name": "netuid",
                            "type": "u16",
                        },
                        {
                            "name": "uid",
                            "type": "u16",
                        },
                    ],
                    "type": "Vec<u8>",
                    "decoder": lambda raw_bytes, runtime: _cyscale_decode(
                        "NeuronInfo", raw_bytes, runtime
                    ),
                },
                "get_neurons": {
                    "params": [
                        {
                            "name": "netuid",
                            "type": "u16",
                        },
                    ],
                    "type": "Vec<u8>",
                    "decoder": lambda raw_bytes, runtime: _cyscale_decode(
                        "Vec<NeuronInfo>", raw_bytes, runtime
                    ),
                },
            }
        },
        "StakeInfoRuntimeApi": {
            "methods": {
                "get_stake_info_for_coldkey": {
                    "params": [
                        {
                            "name": "coldkey_account_vec",
                            "type": "Vec<u8>",
                        },
                    ],
                    "type": "Vec<u8>",
                    "encoder": _encode_vec_u8_coldkey,
                    "decoder": stake_info_decode_vec_legacy_compatibility,
                },
                "get_stake_info_for_coldkeys": {
                    "params": [
                        {
                            "name": "coldkey_account_vecs",
                            "type": "Vec<Vec<u8>>",
                        },
                    ],
                    "type": "Vec<u8>",
                    "encoder": _encode_vec_vec_u8,
                    "decoder": lambda raw_bytes, runtime: _cyscale_decode(
                        "Vec<(Vec<u8>, Vec<StakeInfo>)>", raw_bytes, runtime
                    ),
                },
            },
        },
        "SubnetInfoRuntimeApi": {
            "methods": {
                "get_subnet_hyperparams": {
                    "params": [
                        {
                            "name": "netuid",
                            "type": "u16",
                        },
                    ],
                    "type": "Vec<u8>",
                    "decoder": lambda raw_bytes, runtime: _cyscale_decode(
                        "Option<SubnetHyperparameters>", raw_bytes, runtime
                    ),
                },
                "get_subnet_info": {
                    "params": [
                        {
                            "name": "netuid",
                            "type": "u16",
                        },
                    ],
                    "type": "Vec<u8>",
                    "decoder": lambda raw_bytes, runtime: _cyscale_decode(
                        "Option<SubnetInfo>", raw_bytes, runtime
                    ),
                },
                "get_subnet_info_v2": {
                    "params": [
                        {
                            "name": "netuid",
                            "type": "u16",
                        },
                    ],
                    "type": "Vec<u8>",
                    "decoder": lambda raw_bytes, runtime: _cyscale_decode(
                        "Option<SubnetInfoV2>", raw_bytes, runtime
                    ),
                },
                "get_subnets_info": {
                    "params": [],
                    "type": "Vec<u8>",
                    "decoder": lambda raw_bytes, runtime: _cyscale_decode(
                        "Vec<Option<SubnetInfo>>", raw_bytes, runtime
                    ),
                },
                "get_subnets_info_v2": {
                    "params": [],
                    "type": "Vec<u8>",
                    "decoder": lambda raw_bytes, runtime: _cyscale_decode(
                        "Vec<Option<SubnetInfo>>", raw_bytes, runtime
                    ),
                },
            }
        },
    },
}
