from typing import TYPE_CHECKING, Optional, Any

from scalecodec import ScaleBytes
from scalecodec.base import ScaleType

from async_substrate_interface.utils import hex_to_bytes

if TYPE_CHECKING:
    from async_substrate_interface.types import Runtime
    from async_substrate_interface.utils.storage import StorageKey


def _determine_if_old_runtime_call(runtime_call_def, runtime) -> bool:
    # Runtime calls whose output is Vec<u8> must use the old decode path
    return runtime.type_id_to_name.get(runtime_call_def["output"]) == "Vec<u8>"


def try_batch_decode(
    items: list[tuple["StorageKey", Optional[ScaleBytes]]],
    runtime: "Runtime",
) -> list:
    """
    Decode a list of (StorageKey, data) pairs, using cyscale's batch_decode when
    available and falling back to StorageKey.decode_scale_value otherwise.

    Mirrors the None-data logic in StorageKey.decode_scale_value:
      - data present      → decode as value_scale_type
      - None + Default    → decode default bytes as value_scale_type
      - None + Optional   → decode default bytes (0x00) as Option<value_scale_type>
    """
    if not runtime.implements_scaleinfo:
        return [sk.decode_scale_value(data).value for sk, data in items]

    type_strings = []
    raw_bytes_list = []
    for storage_key, data in items:
        msf = storage_key.metadata_storage_function
        if data is not None:
            type_strings.append(storage_key.value_scale_type)
            raw_bytes_list.append(bytes(data.data))
        elif msf.value["modifier"] == "Default":
            type_strings.append(storage_key.value_scale_type)
            raw_bytes_list.append(bytes(msf.value_object["default"].value_object))
        else:
            type_strings.append(f"Option<{storage_key.value_scale_type}>")
            raw_bytes_list.append(bytes(msf.value_object["default"].value_object))

    return runtime.runtime_config.batch_decode(type_strings, raw_bytes_list)


def _decode_scale_list_with_runtime(
    type_strings: list[str],
    scale_bytes_list: list[bytes],
    runtime: "Runtime",
):
    if runtime.implements_scaleinfo:
        obj = runtime.runtime_config.batch_decode(type_strings, scale_bytes_list)
    else:
        obj = [
            scale_decode(x, y, runtime).value
            for (x, y) in zip(type_strings, scale_bytes_list)
        ]
    return obj


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
    all_decoded = _decode_scale_list_with_runtime(
        pre_decoded_key_types + pre_decoded_value_types,
        pre_decoded_keys + pre_decoded_values,
        runtime,
    )
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
        result.append([item_key, dv])
    return result


def scale_decode(
    type_string: str, scale_bytes: str | bytes | ScaleBytes, runtime: "Runtime"
) -> Any:
    if isinstance(scale_bytes, (str, bytes)):
        scale_bytes = ScaleBytes(scale_bytes)

    obj = runtime.runtime_config.create_scale_object(
        type_string=type_string, data=scale_bytes, metadata=runtime.metadata
    )

    obj.decode(check_remaining=runtime.config.get("strict_scale_decode"))

    return obj.value
