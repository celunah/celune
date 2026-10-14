/* SPDX-License-Identifier: Apache-2.0 */
#ifndef CELUNE_RELEASE_KEYS_H
#define CELUNE_RELEASE_KEYS_H

typedef struct {
    const char *key_id;
    unsigned char public_key[32];
} LauncherTrustedReleaseKey;

static const LauncherTrustedReleaseKey launcher_trusted_release_keys[] = {
    {
        "ed25519-0f15a9d5751e0a3a",
        {
            0x46, 0x95, 0xff, 0x08, 0x26, 0xe4, 0xcb, 0x8f,
            0xf4, 0xbd, 0x2d, 0x95, 0x1a, 0x34, 0x27, 0xc1,
            0xf2, 0xd5, 0x34, 0x91, 0xf5, 0x2a, 0x1a, 0xfc,
            0x80, 0x73, 0x3c, 0x40, 0x10, 0x67, 0xe6, 0x51,
        },
    },
};

#define LAUNCHER_TRUSTED_RELEASE_KEY_COUNT \
    (sizeof(launcher_trusted_release_keys) / sizeof(launcher_trusted_release_keys[0]))

#endif
