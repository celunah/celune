#ifndef CELUNE_LAUNCHER_SIGNATURE_H
#define CELUNE_LAUNCHER_SIGNATURE_H

#include <stddef.h>

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

LauncherBundleStatus launcher_verify_bundle_target(const char *target_path);
int launcher_hash_file(const char *path, unsigned char digest[32]);

#endif
