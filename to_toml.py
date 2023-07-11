

def _dumps_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, str):
        return f'"{value}"'
    elif isinstance(value, list):
        return f"[{', '.join(_dumps_value(v) for v in value)}]"
    else:
        raise TypeError(f"{type(value).__name__} {value!r} is not supported")


def dumps(toml_dict, table=""):
    toml = []
    for key, value in toml_dict.items():
        if isinstance(value, dict):
            table_key = f"{key}" if table else key
            toml.append(f"\n[{table_key}]\n{dumps(value, table_key)}")
        else:
            toml.append(f"{key} = {_dumps_value(value)}")
    return "\n".join(toml)


# def dumps(toml_dict, table=""):
#     def tables_at_end(item):
#         _, value = item
#         return isinstance(value, dict)

#     toml = []
#     for key, value in sorted(toml_dict.items(), key=tables_at_end):
#         if isinstance(value, dict):
#             table_key = f"{table}.{key}" if table else key
#             toml.append(f"\n[{table_key}]\n{dumps(value, table_key)}")
#         else:
#             toml.append(f"{key} = {_dumps_value(value)}")
#     return "\n".join(toml)