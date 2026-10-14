#ifndef CELUNE_LAUNCHER_SIGNATURE_H
#define CELUNE_LAUNCHER_SIGNATURE_H

#include <stddef.h>

#define LAUNCHER_KEYRING_MAX_ENTRIES 128U

typedef struct {
    char key_id[25];
    unsigned char public_key[32];
} LauncherTrustedReleaseKey;

typedef enum {
    LAUNCHER_BUNDLE_INVALID = -1,
    LAUNCHER_BUNDLE_UNSIGNED = 0,
    LAUNCHER_BUNDLE_VALID = 1
} LauncherBundleStatus;

LauncherBundleStatus launcher_verify_manifest_data(
    const unsigned char *manifest_data,
    size_t manifest_size,
    const unsigned char *signature_data,
    size_t signature_size,
    const char *target_name,
    const unsigned char target_digest[32],
    const unsigned char public_key[32],
    const char *key_id
);

LauncherBundleStatus launcher_verify_bundle_target(
    const char *target_path,
    const char *launcher_path
);
int launcher_read_appended_keyring(
    const char *launcher_path,
    LauncherTrustedReleaseKey keys[LAUNCHER_KEYRING_MAX_ENTRIES],
    size_t *key_count
);
int launcher_decode_signed_keyring_data(
    const unsigned char *body,
    size_t body_size,
    const unsigned char root_public_key[32],
    LauncherTrustedReleaseKey keys[LAUNCHER_KEYRING_MAX_ENTRIES],
    size_t *key_count
);
int launcher_decode_appended_keyring_data(
    const unsigned char *trailer_data,
    size_t trailer_size,
    const unsigned char root_public_key[32],
    LauncherTrustedReleaseKey keys[LAUNCHER_KEYRING_MAX_ENTRIES],
    size_t *key_count
);
int launcher_hash_file(const char *path, unsigned char digest[32]);

#endif
