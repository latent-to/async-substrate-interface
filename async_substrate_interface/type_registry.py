from collections import namedtuple
from typing import TYPE_CHECKING, Any

from scalecodec import ScaleBytes
from scalecodec.base import RuntimeConfigurationObject
from scalecodec.utils.ss58 import ss58_decode
from scalecodec.type_registry import load_type_registry_preset

if TYPE_CHECKING:
    from async_substrate_interface.types import Runtime

# Bittensor-specific types needed for legacy (pre-V15) blocks
_BITTENSOR_LEGACY_TYPES = {
    "types": {
        "Balance": "u64",
        "StakeInfo": {
            "type": "struct",
            "type_mapping": [
                ["hotkey", "AccountId"],
                ["coldkey", "AccountId"],
                ["stake", "Compact<u64>"],
            ],
        },
    }
}

_legacy_rc: RuntimeConfigurationObject | None = None


def _get_legacy_rc(ss58_format: int | None = None) -> RuntimeConfigurationObject:
    """Return a lazily-initialised RC with legacy + Bittensor types."""
    global _legacy_rc
    if _legacy_rc is None:
        rc = RuntimeConfigurationObject(ss58_format=ss58_format)
        rc.update_type_registry(load_type_registry_preset(name="legacy") or {})
        rc.update_type_registry(_BITTENSOR_LEGACY_TYPES)
        _legacy_rc = rc
    return _legacy_rc


def _legacy_decode(type_string: str, raw_bytes: bytes, ss58_format: int) -> Any:
    """Decode raw_bytes using the legacy scalecodec type registry."""
    rc = _get_legacy_rc(ss58_format=ss58_format)
    obj = rc.create_scale_object(type_string, data=ScaleBytes(raw_bytes))
    return obj.decode()


def _cyscale_decode(type_name: str, raw_bytes: bytes, runtime: "Runtime") -> Any:
    """Decode raw_bytes as type_name using cyscale's portable registry."""
    type_id = runtime.registry_type_map.get(type_name)
    if type_id is None:
        raise ValueError(f"Type '{type_name}' not found in registry_type_map")
    return runtime.runtime_config.batch_decode([f"scale_info::{type_id}"], [raw_bytes])[
        0
    ]


def stake_info_decode_vec_legacy_compatibility(
    raw_bytes: bytes, runtime: "Runtime"
) -> list[dict[str, str | int | bytes | bool]]:
    ss58_format = runtime.ss58_format
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

    stake_infos = _legacy_decode("Vec<StakeInfo>", raw_bytes, ss58_format=ss58_format)
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
        )._asdict()
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
