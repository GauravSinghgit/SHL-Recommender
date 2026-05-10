"""Fix raw newlines/control chars inside JSON string values, then save cleaned catalog."""
import json
import re

SRC = "data/shl_product_catalog.json"
DST = "data/shl_product_catalog_clean.json"


def fix_json_strings(s: str) -> str:
    """
    Walk the raw JSON text character by character.
    Inside a JSON string, replace bare newlines/carriage-returns/tabs
    with a space so the parser doesn't choke.
    """
    result = []
    in_string = False
    i = 0
    while i < len(s):
        c = s[i]
        # Handle escape sequences inside strings
        if in_string and c == "\\":
            result.append(c)
            if i + 1 < len(s):
                result.append(s[i + 1])
                i += 2
            else:
                i += 1
            continue
        # Toggle string mode on unescaped quote
        if c == '"':
            in_string = not in_string
            result.append(c)
        elif in_string and c in ("\n", "\r"):
            # Bare newline inside a string — collapse to space
            result.append(" ")
        elif in_string and c == "\t":
            result.append(" ")
        else:
            result.append(c)
        i += 1
    return "".join(result)


with open(SRC, encoding="utf-8") as f:
    raw = f.read()

print(f"Raw size: {len(raw):,} chars")
fixed = fix_json_strings(raw)
data = json.loads(fixed)

print(f"Parsed OK — {len(data) if isinstance(data, list) else 'dict'} items")

with open(DST, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Saved clean catalog → {DST}")

# Show structure
if isinstance(data, list) and data:
    print("\nKeys:", list(data[0].keys()))
    print("\nSample item:")
    print(json.dumps(data[0], indent=2)[:800])
elif isinstance(data, dict):
    print("Top-level keys:", list(data.keys()))
