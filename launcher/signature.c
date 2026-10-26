#include "signature.h"

#include <sodium.h>

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "release_keys.h"

#define MANIFEST_LIMIT (1024U * 1024U)
#define SIGNATURE_LIMIT 4096U
#define SIGNATURE_BASE64_LENGTH 88U
#define PATH_BUFFER_SIZE 4096U
#define KEYRING_PAYLOAD_MAGIC "CELUNE-KEYRING-1"
#define KEYRING_TRAILER_MAGIC "CELUNE-KEYS-END1"
#define KEYRING_KEY_ID_SIZE 24U
#define KEYRING_ENTRY_SIZE (KEYRING_KEY_ID_SIZE + crypto_sign_PUBLICKEYBYTES)
#define KEYRING_SIGNATURE_SIZE crypto_sign_BYTES
#define KEYRING_FOOTER_SIZE (8U + 16U)
#define KEYRING_MAX_BODY_SIZE \
    (16U + 2U + LAUNCHER_KEYRING_MAX_ENTRIES * KEYRING_ENTRY_SIZE + \
     KEYRING_SIGNATURE_SIZE)

static int read_limited_file(
    const char *path,
    size_t limit,
    unsigned char **data,
    size_t *data_size
) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return 0;
    }

    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return 0;
    }
    long file_size = ftell(file);
    if (file_size < 0 || (unsigned long)file_size > limit) {
        fclose(file);
        return 0;
    }
    if (fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return 0;
    }

    size_t size = (size_t)file_size;
    unsigned char *buffer = malloc(size + 1U);
    if (buffer == NULL) {
        fclose(file);
        return 0;
    }
    size_t read_size = fread(buffer, 1U, size, file);
    int read_failed = ferror(file);
    fclose(file);
    if (read_failed || read_size != size) {
        free(buffer);
        return 0;
    }

    buffer[size] = '\0';
    *data = buffer;
    *data_size = size;
    return 1;
}

static int read_signature_file(
    const char *path,
    unsigned char **data,
    size_t *data_size
) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return errno == ENOENT ? LAUNCHER_BUNDLE_UNSIGNED : LAUNCHER_BUNDLE_INVALID;
    }
    fclose(file);
    return read_limited_file(path, SIGNATURE_LIMIT, data, data_size)
        ? LAUNCHER_BUNDLE_VALID
        : LAUNCHER_BUNDLE_INVALID;
}

int launcher_hash_file(const char *path, unsigned char digest[crypto_hash_sha256_BYTES]) {
    if (path == NULL || digest == NULL || sodium_init() < 0) {
        return 0;
    }

    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return 0;
    }

    crypto_hash_sha256_state state;
    if (crypto_hash_sha256_init(&state) != 0) {
        fclose(file);
        return 0;
    }

    unsigned char chunk[65536];
    size_t read_size;
    int failed = 0;
    while ((read_size = fread(chunk, 1U, sizeof(chunk), file)) > 0U) {
        if (crypto_hash_sha256_update(&state, chunk, (unsigned long long)read_size) != 0) {
            failed = 1;
            break;
        }
    }
    if (ferror(file)) {
        failed = 1;
    }
    fclose(file);
    if (failed || crypto_hash_sha256_final(&state, digest) != 0) {
        sodium_memzero(&state, sizeof(state));
        return 0;
    }
    sodium_memzero(&state, sizeof(state));
    return 1;
}

static int decode_signature_envelope(
    const unsigned char *signature_data,
    size_t signature_size,
    const char *key_id,
    unsigned char signature[crypto_sign_BYTES]
) {
    char prefix[256];
    int prefix_size = snprintf(
        prefix,
        sizeof(prefix),
        "{\"algorithm\":\"Ed25519\",\"format\":1,\"key_id\":\"%s\",\"signature\":\"",
        key_id
    );
    if (prefix_size < 0 || (size_t)prefix_size >= sizeof(prefix)) {
        return 0;
    }

    size_t prefix_length = (size_t)prefix_size;
    if (signature_size != prefix_length + SIGNATURE_BASE64_LENGTH + 3U ||
        memcmp(signature_data, prefix, prefix_length) != 0 ||
        memcmp(
            signature_data + prefix_length + SIGNATURE_BASE64_LENGTH,
            "\"}\n",
            3U
        ) != 0) {
        return 0;
    }

    size_t decoded_size = 0U;
    const char *encoded = (const char *)signature_data + prefix_length;
    const char *encoded_end = NULL;
    if (sodium_base642bin(
            signature,
            crypto_sign_BYTES,
            encoded,
            SIGNATURE_BASE64_LENGTH,
            NULL,
            &decoded_size,
            &encoded_end,
            sodium_base64_VARIANT_ORIGINAL
        ) != 0 ||
        decoded_size != crypto_sign_BYTES ||
        encoded_end != encoded + SIGNATURE_BASE64_LENGTH) {
        return 0;
    }

    return 1;
}

static int hex_value(unsigned char character) {
    if (character >= '0' && character <= '9') {
        return (int)(character - '0');
    }
    if (character >= 'a' && character <= 'f') {
        return (int)(character - 'a') + 10;
    }
    return -1;
}

static uint64_t read_little_endian_u64(const unsigned char bytes[8]) {
    uint64_t value = 0U;
    for (size_t index = 0U; index < 8U; index++) {
        value |= (uint64_t)bytes[index] << (index * 8U);
    }
    return value;
}

static int valid_release_key(
    const unsigned char public_key[crypto_sign_PUBLICKEYBYTES],
    const unsigned char key_id[KEYRING_KEY_ID_SIZE]
) {
    unsigned char digest[crypto_hash_sha256_BYTES];
    char expected_key_id[KEYRING_KEY_ID_SIZE + 1U];
    if (crypto_hash_sha256(digest, public_key, crypto_sign_PUBLICKEYBYTES) != 0) {
        return 0;
    }
    int written = snprintf(
        expected_key_id,
        sizeof(expected_key_id),
        "ed25519-%02x%02x%02x%02x%02x%02x%02x%02x",
        digest[0], digest[1], digest[2], digest[3],
        digest[4], digest[5], digest[6], digest[7]
    );
    sodium_memzero(digest, sizeof(digest));
    return written == (int)KEYRING_KEY_ID_SIZE &&
        sodium_memcmp(expected_key_id, key_id, KEYRING_KEY_ID_SIZE) == 0;
}

int launcher_decode_signed_keyring_data(
    const unsigned char *body,
    size_t body_size,
    const unsigned char root_public_key[crypto_sign_PUBLICKEYBYTES],
    LauncherTrustedReleaseKey keys[LAUNCHER_KEYRING_MAX_ENTRIES],
    size_t *key_count
) {
    static const unsigned char payload_magic[] = KEYRING_PAYLOAD_MAGIC;
    if (body == NULL || root_public_key == NULL || keys == NULL ||
        key_count == NULL || sodium_init() < 0) {
        return 0;
    }
    sodium_memzero(keys, LAUNCHER_KEYRING_MAX_ENTRIES * sizeof(*keys));
    *key_count = 0U;
    size_t minimum_size = sizeof(payload_magic) - 1U + 2U +
        KEYRING_ENTRY_SIZE + KEYRING_SIGNATURE_SIZE;
    if (body_size < minimum_size || body_size > KEYRING_MAX_BODY_SIZE ||
        memcmp(body, payload_magic, sizeof(payload_magic) - 1U) != 0) {
        return 0;
    }
    size_t payload_size = body_size - KEYRING_SIGNATURE_SIZE;
    size_t count_offset = sizeof(payload_magic) - 1U;
    size_t count = (size_t)body[count_offset] |
        ((size_t)body[count_offset + 1U] << 8U);
    if (count == 0U || count > LAUNCHER_KEYRING_MAX_ENTRIES ||
        payload_size != count_offset + 2U + count * KEYRING_ENTRY_SIZE ||
        crypto_sign_verify_detached(
            body + payload_size,
            body,
            (unsigned long long)payload_size,
            root_public_key
        ) != 0) {
        return 0;
    }

    size_t offset = count_offset + 2U;
    for (size_t index = 0U; index < count; index++) {
        const unsigned char *key_id = body + offset;
        const unsigned char *public_key = key_id + KEYRING_KEY_ID_SIZE;
        if (!valid_release_key(public_key, key_id)) {
            sodium_memzero(keys, LAUNCHER_KEYRING_MAX_ENTRIES * sizeof(*keys));
            return 0;
        }
        for (size_t prior = 0U; prior < index; prior++) {
            if (sodium_memcmp(
                    keys[prior].key_id,
                    key_id,
                    KEYRING_KEY_ID_SIZE
                ) == 0) {
                sodium_memzero(keys, LAUNCHER_KEYRING_MAX_ENTRIES * sizeof(*keys));
                return 0;
            }
        }
        memcpy(keys[index].key_id, key_id, KEYRING_KEY_ID_SIZE);
        keys[index].key_id[KEYRING_KEY_ID_SIZE] = '\0';
        memcpy(keys[index].public_key, public_key, crypto_sign_PUBLICKEYBYTES);
        offset += KEYRING_ENTRY_SIZE;
    }
    *key_count = count;
    return 1;
}

int launcher_decode_appended_keyring_data(
    const unsigned char *trailer_data,
    size_t trailer_size,
    const unsigned char root_public_key[crypto_sign_PUBLICKEYBYTES],
    LauncherTrustedReleaseKey keys[LAUNCHER_KEYRING_MAX_ENTRIES],
    size_t *key_count
) {
    static const unsigned char trailer_magic[] = KEYRING_TRAILER_MAGIC;
    if (trailer_data == NULL || root_public_key == NULL || keys == NULL ||
        key_count == NULL) {
        return 0;
    }
    sodium_memzero(keys, LAUNCHER_KEYRING_MAX_ENTRIES * sizeof(*keys));
    *key_count = 0U;
    if (trailer_size < KEYRING_FOOTER_SIZE) {
        return 0;
    }
    const unsigned char *footer = trailer_data + trailer_size - KEYRING_FOOTER_SIZE;
    uint64_t body_size = read_little_endian_u64(footer);
    if (memcmp(
            footer + 8U,
            trailer_magic,
            sizeof(trailer_magic) - 1U
        ) != 0 ||
        body_size > KEYRING_MAX_BODY_SIZE ||
        body_size != trailer_size - KEYRING_FOOTER_SIZE) {
        return 0;
    }
    return launcher_decode_signed_keyring_data(
        trailer_data,
        (size_t)body_size,
        root_public_key,
        keys,
        key_count
    );
}

int launcher_read_appended_keyring(
    const char *launcher_path,
    LauncherTrustedReleaseKey keys[LAUNCHER_KEYRING_MAX_ENTRIES],
    size_t *key_count
) {
    if (launcher_path == NULL || keys == NULL || key_count == NULL ||
        sodium_init() < 0) {
        return 0;
    }
    *key_count = 0U;

    FILE *file = fopen(launcher_path, "rb");
    if (file == NULL || fseek(file, 0, SEEK_END) != 0) {
        if (file != NULL) {
            fclose(file);
        }
        return 0;
    }
    long file_size = ftell(file);
    if (file_size < (long)KEYRING_FOOTER_SIZE ||
        fseek(file, -(long)KEYRING_FOOTER_SIZE, SEEK_END) != 0) {
        fclose(file);
        return 0;
    }

    unsigned char footer[KEYRING_FOOTER_SIZE];
    if (fread(footer, 1U, sizeof(footer), file) != sizeof(footer)) {
        fclose(file);
        return 0;
    }
    uint64_t body_size_u64 = read_little_endian_u64(footer);
    if (body_size_u64 > KEYRING_MAX_BODY_SIZE ||
        body_size_u64 > (uint64_t)file_size - KEYRING_FOOTER_SIZE) {
        fclose(file);
        return 0;
    }
    size_t body_size = (size_t)body_size_u64;
    size_t trailer_size = body_size + KEYRING_FOOTER_SIZE;
    unsigned char trailer[KEYRING_MAX_BODY_SIZE + KEYRING_FOOTER_SIZE];
    if (fseek(file, -(long)trailer_size, SEEK_END) != 0 ||
        fread(trailer, 1U, trailer_size, file) != trailer_size) {
        fclose(file);
        return 0;
    }
    fclose(file);

    int decoded = launcher_decode_appended_keyring_data(
        trailer,
        trailer_size,
        launcher_release_keyring_root,
        keys,
        key_count
    );
    sodium_memzero(trailer, sizeof(trailer));
    return decoded;
}

static const unsigned char *find_bytes(
    const unsigned char *data,
    size_t data_size,
    const char *search,
    size_t search_size
) {
    if (search_size > data_size) {
        return NULL;
    }
    for (size_t index = 0U; index <= data_size - search_size; index++) {
        if (memcmp(data + index, search, search_size) == 0) {
            return data + index;
        }
    }
    return NULL;
}

static int extract_target_digest(
    const unsigned char *manifest_data,
    size_t manifest_size,
    const char *target_name,
    unsigned char expected_digest[crypto_hash_sha256_BYTES]
) {
    if (manifest_size > MANIFEST_LIMIT ||
        memchr(manifest_data, '\0', manifest_size) != NULL) {
        return 0;
    }

    char marker[256];
    int marker_size = snprintf(marker, sizeof(marker), "\"%s\": \"", target_name);
    if (marker_size < 0 || (size_t)marker_size >= sizeof(marker)) {
        return 0;
    }

    static const char signature_required[] = "\"signature_required\": true";
    if (find_bytes(
            manifest_data,
            manifest_size,
            signature_required,
            sizeof(signature_required) - 1U
        ) == NULL) {
        return 0;
    }
    static const char files_marker[] = "\"files\": {";
    const unsigned char *files_start = find_bytes(
        manifest_data,
        manifest_size,
        files_marker,
        sizeof(files_marker) - 1U
    );
    if (files_start == NULL) {
        return 0;
    }
    const unsigned char *files_end = memchr(
        files_start,
        '}',
        manifest_size - (size_t)(files_start - manifest_data)
    );
    if (files_end == NULL) {
        return 0;
    }
    size_t files_size = (size_t)(files_end - files_start);
    const unsigned char *entry = find_bytes(
        files_start,
        files_size,
        marker,
        (size_t)marker_size
    );
    if (entry == NULL) {
        return 0;
    }
    size_t remaining_files_size = files_size - (size_t)(entry - files_start) - 1U;
    const unsigned char *duplicate_entry = find_bytes(
        entry + 1,
        remaining_files_size,
        marker,
        (size_t)marker_size
    );
    if (duplicate_entry != NULL && duplicate_entry < files_end) {
        return 0;
    }

    size_t entry_offset = (size_t)(entry - manifest_data);
    size_t digest_offset = entry_offset + (size_t)marker_size;
    size_t digest_end = digest_offset + crypto_hash_sha256_BYTES * 2U;
    if (digest_end >= (size_t)(files_end - manifest_data)) {
        return 0;
    }

    const unsigned char *hex = manifest_data + digest_offset;
    for (size_t index = 0U; index < crypto_hash_sha256_BYTES; index++) {
        int high = hex_value(hex[index * 2U]);
        int low = hex_value(hex[index * 2U + 1U]);
        if (high < 0 || low < 0) {
            return 0;
        }
        expected_digest[index] = (unsigned char)((high << 4) | low);
    }
    return hex[crypto_hash_sha256_BYTES * 2U] == '"';
}

LauncherBundleStatus launcher_verify_manifest_data(
    const unsigned char *manifest_data,
    size_t manifest_size,
    const unsigned char *signature_data,
    size_t signature_size,
    const char *target_name,
    const unsigned char target_digest[crypto_hash_sha256_BYTES],
    const unsigned char public_key[crypto_sign_PUBLICKEYBYTES],
    const char *key_id
) {
    if (signature_data == NULL) {
        return LAUNCHER_BUNDLE_UNSIGNED;
    }
    if (signature_size == 0U || manifest_data == NULL || manifest_size == 0U ||
        target_name == NULL || target_digest == NULL ||
        public_key == NULL || key_id == NULL || sodium_init() < 0) {
        return LAUNCHER_BUNDLE_INVALID;
    }

    unsigned char signature[crypto_sign_BYTES];
    if (!decode_signature_envelope(
            signature_data,
            signature_size,
            key_id,
            signature
        ) ||
        crypto_sign_verify_detached(
            signature,
            manifest_data,
            (unsigned long long)manifest_size,
            public_key
        ) != 0) {
        return LAUNCHER_BUNDLE_INVALID;
    }

    unsigned char expected_digest[crypto_hash_sha256_BYTES];
    if (!extract_target_digest(
            manifest_data,
            manifest_size,
            target_name,
            expected_digest
        ) ||
        sodium_memcmp(expected_digest, target_digest, sizeof(expected_digest)) != 0) {
        return LAUNCHER_BUNDLE_INVALID;
    }

    return LAUNCHER_BUNDLE_VALID;
}

static int valid_target_name(const char *name) {
    if (name == NULL || name[0] == '\0') {
        return 0;
    }
    for (const unsigned char *character = (const unsigned char *)name;
         *character != '\0';
         character++) {
        if (!( (*character >= 'a' && *character <= 'z') ||
               (*character >= 'A' && *character <= 'Z') ||
               (*character >= '0' && *character <= '9') ||
               *character == '.' || *character == '_' || *character == '-') ) {
            return 0;
        }
    }
    return 1;
}

static int sidecar_path(
    const char *target_path,
    const char *name,
    char result[PATH_BUFFER_SIZE]
) {
    const char *separator = strrchr(target_path, '/');
    const char *windows_separator = strrchr(target_path, '\\');
    if (windows_separator != NULL &&
        (separator == NULL || windows_separator > separator)) {
        separator = windows_separator;
    }
    if (separator == NULL) {
        return 0;
    }

    size_t directory_size = (size_t)(separator - target_path) + 1U;
    int name_size = snprintf(
        result,
        PATH_BUFFER_SIZE,
        "%.*s%s",
        (int)directory_size,
        target_path,
        name
    );
    return name_size >= 0 && (size_t)name_size < PATH_BUFFER_SIZE;
}

LauncherBundleStatus launcher_verify_bundle_target(
    const char *target_path,
    const char *launcher_path
) {
    if (target_path == NULL || launcher_path == NULL) {
        return LAUNCHER_BUNDLE_INVALID;
    }

    char manifest_path[PATH_BUFFER_SIZE];
    char signature_path[PATH_BUFFER_SIZE];
    if (!sidecar_path(target_path, "celune-update.json", manifest_path) ||
        !sidecar_path(target_path, "celune-update.sig", signature_path)) {
        return LAUNCHER_BUNDLE_INVALID;
    }

    unsigned char *signature_data = NULL;
    size_t signature_size = 0U;
    LauncherBundleStatus signature_status = read_signature_file(
        signature_path,
        &signature_data,
        &signature_size
    );
    if (signature_status != LAUNCHER_BUNDLE_VALID) {
        return signature_status;
    }

    unsigned char *manifest_data = NULL;
    size_t manifest_size = 0U;
    unsigned char target_digest[crypto_hash_sha256_BYTES];
    const char *target_name = strrchr(target_path, '/');
    const char *windows_separator = strrchr(target_path, '\\');
    if (windows_separator != NULL &&
        (target_name == NULL || windows_separator > target_name)) {
        target_name = windows_separator;
    }
    target_name = target_name == NULL ? target_path : target_name + 1;

    int files_ready = valid_target_name(target_name) &&
        read_limited_file(
            manifest_path,
            MANIFEST_LIMIT,
            &manifest_data,
            &manifest_size
        ) &&
        launcher_hash_file(target_path, target_digest);
    LauncherBundleStatus result = LAUNCHER_BUNDLE_INVALID;
    if (files_ready) {
        LauncherTrustedReleaseKey trusted_keys[LAUNCHER_KEYRING_MAX_ENTRIES];
        size_t trusted_key_count = 0U;
        if (launcher_read_appended_keyring(
                launcher_path,
                trusted_keys,
                &trusted_key_count
            )) {
            for (size_t key_index = 0U;
                 key_index < trusted_key_count;
                 key_index++) {
                result = launcher_verify_manifest_data(
                    manifest_data,
                    manifest_size,
                    signature_data,
                    signature_size,
                    target_name,
                    target_digest,
                    trusted_keys[key_index].public_key,
                    trusted_keys[key_index].key_id
                );
                if (result == LAUNCHER_BUNDLE_VALID) {
                    break;
                }
            }
        }
        sodium_memzero(trusted_keys, sizeof(trusted_keys));
    }

    free(signature_data);
    free(manifest_data);
    return result;
}
