import asyncio
from typing import Union, TYPE_CHECKING

from bt_decode import AxonInfo, PrometheusInfo
from scalecodec import ScaleBytes

from async_substrate_interface.utils import hex_to_bytes
from async_substrate_interface.types import ScaleObj

if TYPE_CHECKING:
    from async_substrate_interface.types import Runtime


def _determine_if_old_runtime_call(runtime_call_def, metadata_v15_value) -> bool:
    # Check if the output type is a Vec<u8>
    # If so, call the API using the old method
    output_type_def = [
        x
        for x in metadata_v15_value["types"]["types"]
        if x["id"] == runtime_call_def["output"]
    ]
    if output_type_def:
        output_type_def = output_type_def[0]

        if "sequence" in output_type_def["type"]["def"]:
            output_type_seq_def_id = output_type_def["type"]["def"]["sequence"]["type"]
            output_type_seq_def = [
                x
                for x in metadata_v15_value["types"]["types"]
                if x["id"] == output_type_seq_def_id
            ]
            if output_type_seq_def:
                output_type_seq_def = output_type_seq_def[0]
                if (
                    "primitive" in output_type_seq_def["type"]["def"]
                    and output_type_seq_def["type"]["def"]["primitive"] == "u8"
                ):
                    return True
    return False


def _bt_decode_to_dict_or_list(obj) -> Union[dict, list[dict]]:
    if isinstance(obj, list):
        return [_bt_decode_to_dict_or_list(item) for item in obj]

    as_dict = {}
    for key in dir(obj):
        if not key.startswith("_"):
            val = getattr(obj, key)
            if isinstance(val, (AxonInfo, PrometheusInfo)):
                as_dict[key] = _bt_decode_to_dict_or_list(val)
            else:
                as_dict[key] = val
    return as_dict


def _decode_scale_list_with_runtime(
    type_strings: list[str],
    scale_bytes_list: list[bytes],
    runtime: "Runtime",
    return_scale_obj: bool = False,
):
    if runtime.metadata_v15 is not None:
        obj = runtime.runtime_config.batch_decode(type_strings, scale_bytes_list)
    else:
        obj = [
            legacy_scale_decode(x, y, runtime)
            for (x, y) in zip(type_strings, scale_bytes_list)
        ]
    if return_scale_obj:
        return [ScaleObj(x) for x in obj]
    else:
        return obj


async def _async_decode_scale_list_with_runtime(
    type_strings: list[str],
    scale_bytes_list: list[bytes],
    runtime: "Runtime",
    return_scale_obj: bool = False,
):
    if runtime.metadata_v15 is not None:
        obj = await asyncio.to_thread(
            runtime.runtime_config.batch_decode, type_strings, scale_bytes_list
        )
    else:
        obj = [
            legacy_scale_decode(x, y, runtime)
            for (x, y) in zip(type_strings, scale_bytes_list)
        ]
    if return_scale_obj:
        return [ScaleObj(x) for x in obj]
    else:
        return obj


def _decode_query_map_pre(
    result_group_changes: list,
    prefix,
    param_types,
    params,
    value_type,
    key_hashers,
):
    def concat_hash_len(key_hasher: str) -> int:
        """
        Helper function to avoid if statements
        """
        if key_hasher == "Blake2_128Concat":
            return 16
        elif key_hasher == "Twox64Concat":
            return 8
        elif key_hasher == "Identity":
            return 0
        else:
            raise ValueError("Unsupported hash type")

    hex_to_bytes_ = hex_to_bytes

    # Determine type string
    n_free_keys = len(param_types) - len(params)
    if n_free_keys == 1:
        # Single-key map: skip the hash prefix bytes entirely — no need to decode them
        n = len(params)
        hash_len = concat_hash_len(key_hashers[n])
        key_type_string = param_types[n]
    else:
        key_type_string_ = []
        for n in range(len(params), len(param_types)):
            key_type_string_.append(f"[u8; {concat_hash_len(key_hashers[n])}]")
            key_type_string_.append(param_types[n])
        key_type_string = f"({', '.join(key_type_string_)})"
        hash_len = None

    pre_decoded_keys = []
    pre_decoded_key_types = [key_type_string] * len(result_group_changes)
    pre_decoded_values = []
    pre_decoded_value_types = [value_type] * len(result_group_changes)

    for item in result_group_changes:
        raw_key = bytes.fromhex(item[0][len(prefix) :])
        pre_decoded_keys.append(raw_key[hash_len:] if hash_len else raw_key)
        pre_decoded_values.append(
            hex_to_bytes_(item[1]) if item[1] is not None else b""
        )
    return (
        pre_decoded_key_types,
        pre_decoded_value_types,
        pre_decoded_keys,
        pre_decoded_values,
    )


def _decode_query_map_post(
    pre_decoded_key_types,
    pre_decoded_value_types,
    all_decoded,
    runtime: "Runtime",
    param_types,
    params,
    ignore_decoding_errors,
):
    result = []
    middl_index = len(all_decoded) // 2
    decoded_keys = all_decoded[:middl_index]
    decoded_values = all_decoded[middl_index:]
    for kts, vts, dk, dv in zip(
        pre_decoded_key_types,
        pre_decoded_value_types,
        decoded_keys,
        decoded_values,
    ):
        try:
            # strip key_hashers to use as item key
            if len(param_types) - len(params) == 1:
                item_key = dk
            else:
                try:
                    item_key = tuple(
                        dk[i * 2 + 1] for i in range(len(param_types) - len(params))
                    )
                except IndexError:
                    item_key = dk
        except Exception as _:
            if not ignore_decoding_errors:
                raise
            item_key = None
        result.append([item_key, ScaleObj(dv)])
    return result


async def decode_query_map_async(
    result_group_changes: list,
    prefix,
    runtime: "Runtime",
    param_types,
    params,
    value_type,
    key_hashers,
    ignore_decoding_errors,
):
    (
        pre_decoded_key_types,
        pre_decoded_value_types,
        pre_decoded_keys,
        pre_decoded_values,
    ) = _decode_query_map_pre(
        result_group_changes,
        prefix,
        param_types,
        params,
        value_type,
        key_hashers,
    )
    all_decoded = await _async_decode_scale_list_with_runtime(
        pre_decoded_key_types + pre_decoded_value_types,
        pre_decoded_keys + pre_decoded_values,
        runtime,
    )
    return _decode_query_map_post(
        pre_decoded_key_types,
        pre_decoded_value_types,
        all_decoded,
        runtime,
        param_types,
        params,
        ignore_decoding_errors,
    )


def decode_query_map(
    result_group_changes: list,
    prefix,
    runtime: "Runtime",
    param_types,
    params,
    value_type,
    key_hashers,
    ignore_decoding_errors,
):
    (
        pre_decoded_key_types,
        pre_decoded_value_types,
        pre_decoded_keys,
        pre_decoded_values,
    ) = _decode_query_map_pre(
        result_group_changes,
        prefix,
        param_types,
        params,
        value_type,
        key_hashers,
    )
    all_decoded = _decode_scale_list_with_runtime(
        pre_decoded_key_types + pre_decoded_value_types,
        pre_decoded_keys + pre_decoded_values,
        runtime,
    )
    return _decode_query_map_post(
        pre_decoded_key_types,
        pre_decoded_value_types,
        all_decoded,
        runtime,
        param_types,
        params,
        ignore_decoding_errors,
    )


def legacy_scale_decode(
    type_string: str, scale_bytes: Union[str, bytes, ScaleBytes], runtime: "Runtime"
):
    if isinstance(scale_bytes, (str, bytes)):
        scale_bytes = ScaleBytes(scale_bytes)

    obj = runtime.runtime_config.create_scale_object(
        type_string=type_string, data=scale_bytes, metadata=runtime.metadata
    )

    obj.decode(check_remaining=runtime.config.get("strict_scale_decode"))

    return obj.value
