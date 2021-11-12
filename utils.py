import ast

def parse_argdict(argdict):
    result = {}
    for kv in argdict:
        key, val = kv.split("=")
        try:
            result[key] = ast.literal_eval(val)
        except ValueError:
            result[key] = str(val)
    return result
