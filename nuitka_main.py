# SPDX-License-Identifier: Apache-2.0

import sys

# Nuitka sets __compiled__ on __main__, so this branch will be ignored if the user attempts to run it
# if you meant to run Celune, invoke her via `main.py` instead, or launch:
# `celune` - Linux builds
# `celune.exe` - Windows builds
if __name__ == "__main__" and getattr(
    sys.modules.get("__main__"), "__compiled__", False
):
    from celune.updater import verify_local_bundle
    from celune.exceptions import UpdateError

    try:
        verify_local_bundle()
    except UpdateError as exc:
        print(exc)
        raise SystemExit(1) from exc

    from celune.entrypoint import main

    main()
