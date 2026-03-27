import asyncio
from typing import Union, TYPE_CHECKING, Any

from bt_decode import AxonInfo, PrometheusInfo, decode_list
from scalecodec import ScaleBytes, ss58_encode

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
        obj = decode_list(type_strings, runtime.registry, scale_bytes_list)
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
            decode_list, type_strings, runtime.registry, scale_bytes_list
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
    decode_ss58: bool = False,
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
                if decode_ss58 and kts == "scale_info::0":
                    if isinstance(item_key, str) and item_key.startswith("0x"):
                        # cyscale fast path returns hex string
                        item_key = ss58_encode(
                            bytes.fromhex(item_key[2:]), runtime.ss58_format
                        )
                    elif isinstance(item_key[0], (tuple, list)):
                        # bt_decode returns a tuple-wrapped byte array
                        item_key = ss58_encode(bytes(item_key[0]), runtime.ss58_format)
            else:
                try:
                    item_key = tuple(
                        dk[key + 1]
                        for key in range(len(params), len(param_types) + 1, 2)
                    )
                except IndexError:
                    item_key = dk

        except Exception as _:
            if not ignore_decoding_errors:
                raise
            item_key = None
        item_value = dv
        if decode_ss58:
            try:
                value_type_str_int = int(vts.split("::")[1])
                decoded_type_str = runtime.type_id_to_name[value_type_str_int]
                item_value = convert_account_ids(
                    dv, decoded_type_str, runtime.ss58_format
                )
            except (ValueError, KeyError):
                pass
        result.append([item_key, ScaleObj(item_value)])
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
    decode_ss58: bool = False,
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
        decode_ss58=decode_ss58,
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
    decode_ss58: bool = False,
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
        decode_ss58=decode_ss58,
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


def is_accountid32(value: Any) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 32
        and all(isinstance(b, int) and 0 <= b <= 255 for b in value)
    )


def convert_account_ids(value: Any, type_str: str, ss58_format=42) -> Any:
    if "AccountId32" not in type_str:
        return value

    # Option<T>
    if type_str.startswith("Option<") and value is not None:
        inner_type = type_str[7:-1]
        return convert_account_ids(value, inner_type)
    # Vec<T>
    if type_str.startswith("Vec<") and isinstance(value, (list, tuple)):
        inner_type = type_str[4:-1]
        return tuple(convert_account_ids(v, inner_type) for v in value)

    # Vec<Vec<T>>
    if type_str.startswith("Vec<Vec<") and isinstance(value, (list, tuple)):
        inner_type = type_str[8:-2]
        return tuple(
            tuple(convert_account_ids(v2, inner_type) for v2 in v1) for v1 in value
        )

    # Tuple
    if type_str.startswith("(") and isinstance(value, (list, tuple)):
        inner_parts = split_tuple_type(type_str)
        return tuple(convert_account_ids(v, t) for v, t in zip(value, inner_parts))

    # AccountId32
    if type_str == "AccountId32" and is_accountid32(value[0]):
        return ss58_encode(bytes(value[0]), ss58_format=ss58_format)

    # Fallback
    return value


def split_tuple_type(type_str: str) -> list[str]:
    """
    Splits a type string like '(AccountId32, Vec<StakeInfo>)' into ['AccountId32', 'Vec<StakeInfo>']
    Handles nested generics.
    """
    s = type_str[1:-1]
    parts = []
    depth = 0
    current = ""
    for char in s:
        if char == "," and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            if char == "<":
                depth += 1
            elif char == ">":
                depth -= 1
            current += char
    if current:
        parts.append(current.strip())
    return parts
