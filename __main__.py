# SPDX-License-Identifier: Apache-2.0
import os
import sys

if os.name == "nt":
    possible_entrypoints: list[str] = ["`bin\\celune.exe`", "`bin\\celune`"]
elif sys.platform.startswith("linux"):
    possible_entrypoints: list[str] = ["`./bin/celune`"]
else:
    print("I think you may have mistyped that.")
    print("I don't run on your platform anyways.")
    sys.exit(1)

print("I think you may have mistyped that.")
commands: list[str] = ["`uv run python main.py`", *possible_entrypoints]
print(f"Try running {', '.join(commands[:-1])} or {commands[-1]}.")
sys.exit(1)
