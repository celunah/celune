# SPDX-License-Identifier: Apache-2.0

if __name__ == "__main__":
    from celune.exceptions import UpdateError
    from celune.updater import verify_local_bundle

    try:
        verify_local_bundle()
    except UpdateError as exc:
        print(exc)
        raise SystemExit(1) from exc

    from celune.entrypoint import main

    main()
