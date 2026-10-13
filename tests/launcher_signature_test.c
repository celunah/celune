/* SPDX-License-Identifier: Apache-2.0 */
#include "signature.h"

#include <sodium.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition) \
    do { \
        if (!(condition)) { \
            fprintf(stderr, "failed: %s\n", #condition); \
            return 1; \
        } \
    } while (0)

static int hash_file_reference(
    const char *path,
    unsigned char digest[crypto_hash_sha256_BYTES]
) {
    FILE *file = fopen(path, "rb");
    if (file == NULL || fseek(file, 0, SEEK_END) != 0) {
        if (file != NULL) {
            fclose(file);
        }
        return 0;
    }

    long file_size = ftell(file);
    if (file_size < 0 || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return 0;
    }
    size_t size = (size_t)file_size;
    unsigned char *data = malloc(size == 0U ? 1U : size);
    if (data == NULL) {
        fclose(file);
        return 0;
    }
    size_t read_size = fread(data, 1U, size, file);
    int read_failed = ferror(file);
    fclose(file);
    int hash_succeeded = !read_failed && read_size == size &&
        crypto_hash_sha256(digest, data, (unsigned long long)size) == 0;
    free(data);
    return hash_succeeded;
}

int main(int argc, char **argv) {
    static const unsigned char seed[crypto_sign_SEEDBYTES] = {
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
        0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
        0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f
    };
    static const unsigned char payload[] = "Celune launcher signature test";
    static const char target_name[] = "celune-signature-test.bin";
    static const char key_id[] = "ed25519-test-key";

    CHECK(argc > 0 && argv[0] != NULL);
    unsigned char public_key[crypto_sign_PUBLICKEYBYTES];
    unsigned char secret_key[crypto_sign_SECRETKEYBYTES];
    unsigned char target_digest[crypto_hash_sha256_BYTES];
    CHECK(sodium_init() >= 0);
    CHECK(crypto_sign_seed_keypair(public_key, secret_key, seed) == 0);
    CHECK(crypto_hash_sha256(target_digest, payload, sizeof(payload) - 1U) == 0);

    char digest_hex[crypto_hash_sha256_BYTES * 2U + 1U];
    for (size_t index = 0U; index < sizeof(target_digest); index++) {
        snprintf(
            digest_hex + index * 2U,
            sizeof(digest_hex) - index * 2U,
            "%02x",
            (unsigned int)target_digest[index]
        );
    }

    char manifest[512];
    int manifest_length = snprintf(
        manifest,
        sizeof(manifest),
        "{\n"
        "  \"artifact\": \"test\",\n"
        "  \"files\": {\n"
        "    \"%s\": \"%s\"\n"
        "  },\n"
        "  \"revision\": \"test\",\n"
        "  \"signature_required\": true,\n"
        "  \"version\": \"0\"\n"
        "}\n",
        target_name,
        digest_hex
    );
    CHECK(manifest_length > 0 && (size_t)manifest_length < sizeof(manifest));

    unsigned char detached_signature[crypto_sign_BYTES];
    unsigned long long detached_signature_size = 0U;
    CHECK(crypto_sign_detached(
              detached_signature,
              &detached_signature_size,
              (const unsigned char *)manifest,
              (unsigned long long)manifest_length,
              secret_key
          ) == 0);
    CHECK(detached_signature_size == crypto_sign_BYTES);

    char encoded_signature[128];
    CHECK(sodium_bin2base64(
              encoded_signature,
              sizeof(encoded_signature),
              detached_signature,
              sizeof(detached_signature),
              sodium_base64_VARIANT_ORIGINAL
          ) != NULL);
    char envelope_prefix[128];
    int prefix_length = snprintf(
        envelope_prefix,
        sizeof(envelope_prefix),
        "{\"algorithm\":\"Ed25519\",\"format\":1,\"key_id\":\"%s\",\"signature\":\"",
        key_id
    );
    CHECK(prefix_length > 0 && (size_t)prefix_length < sizeof(envelope_prefix));

    unsigned char signature_envelope[256];
    int envelope_length = snprintf(
        (char *)signature_envelope,
        sizeof(signature_envelope),
        "%s%s\"}\n",
        envelope_prefix,
        encoded_signature
    );
    CHECK(envelope_length > 0 && (size_t)envelope_length < sizeof(signature_envelope));

    size_t manifest_size = (size_t)manifest_length;
    size_t envelope_size = (size_t)envelope_length;
    CHECK(launcher_verify_manifest_data(
              (const unsigned char *)manifest,
              manifest_size,
              signature_envelope,
              envelope_size,
              target_name,
              target_digest,
              public_key,
              key_id
          ) == LAUNCHER_BUNDLE_VALID);

    unsigned char changed_signature[sizeof(signature_envelope)];
    memcpy(changed_signature, signature_envelope, envelope_size);
    char *changed_signature_text = (char *)changed_signature;
    char *signature_character = changed_signature_text + prefix_length;
    signature_character[0] = signature_character[0] == 'A' ? 'B' : 'A';
    CHECK(launcher_verify_manifest_data(
              (const unsigned char *)manifest,
              manifest_size,
              changed_signature,
              envelope_size,
              target_name,
              target_digest,
              public_key,
              key_id
          ) == LAUNCHER_BUNDLE_INVALID);

    unsigned char changed_digest[crypto_hash_sha256_BYTES];
    memcpy(changed_digest, target_digest, sizeof(changed_digest));
    changed_digest[0] ^= 0x01U;
    CHECK(launcher_verify_manifest_data(
              (const unsigned char *)manifest,
              manifest_size,
              signature_envelope,
              envelope_size,
              target_name,
              changed_digest,
              public_key,
              key_id
          ) == LAUNCHER_BUNDLE_INVALID);

    manifest[manifest_size - 2U] ^= 0x01U;
    CHECK(launcher_verify_manifest_data(
              (const unsigned char *)manifest,
              manifest_size,
              signature_envelope,
              envelope_size,
              target_name,
              target_digest,
              public_key,
              key_id
          ) == LAUNCHER_BUNDLE_INVALID);
    manifest[manifest_size - 2U] ^= 0x01U;

    CHECK(launcher_verify_manifest_data(
              (const unsigned char *)manifest,
              manifest_size,
              NULL,
              0U,
              target_name,
              target_digest,
              public_key,
              key_id
          ) == LAUNCHER_BUNDLE_UNSIGNED);
    CHECK(launcher_verify_manifest_data(
              (const unsigned char *)manifest,
              manifest_size,
              signature_envelope,
              0U,
              target_name,
              target_digest,
              public_key,
              key_id
          ) == LAUNCHER_BUNDLE_INVALID);

    unsigned char reference_file_digest[crypto_hash_sha256_BYTES];
    unsigned char file_digest[crypto_hash_sha256_BYTES];
    CHECK(hash_file_reference(argv[0], reference_file_digest));
    CHECK(launcher_hash_file(argv[0], file_digest));
    CHECK(sodium_memcmp(
              file_digest,
              reference_file_digest,
              sizeof(reference_file_digest)
          ) == 0);

    sodium_memzero(secret_key, sizeof(secret_key));
    return 0;
}
