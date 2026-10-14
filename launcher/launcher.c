#include "launcher_platform.h"
#include "launcher_id.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

const volatile char launcher_id[] =
    CELUNE_LAUNCHER_ID CELUNE_LAUNCHER_VERSION;

void retain_launcher_id(void) { (void)launcher_id[0]; }

static int launcher_is_main_operation(int argc, char **argv) {
    for (int index = 1; index < argc; index++) {
        if (strcmp(argv[index], "--root") == 0) {
            index++;
            continue;
        }
        if (strncmp(argv[index], "--root=", 7) == 0) {
            continue;
        }

        return strcmp(argv[index], "start") == 0 ||
               strcmp(argv[index], "run") == 0;
    }

    return 1;
}

static int copy_launcher_text(char *dest, size_t size, const char *src) {
    size_t length = strlen(src);
    if (length >= size) {
        return 0;
    }

    memcpy(dest, src, length + 1);
    return 1;
}

int launcher_read_root_override(
    int *argc,
    char **argv,
    char *root,
    size_t root_size
) {
    int has_override = 0;
    const char *environment_root = getenv("CELUNE_ROOT");
    if (environment_root != NULL) {
        if (!copy_launcher_text(root, root_size, environment_root) ||
            root[0] == '\0') {
            return -1;
        }
        has_override = 1;
    }

    int write_index = 1;
    for (int read_index = 1; read_index < *argc; read_index++) {
        const char *value = NULL;
        if (strcmp(argv[read_index], "--root") == 0) {
            if (read_index + 1 >= *argc) {
                return -1;
            }
            value = argv[++read_index];
        } else if (strncmp(argv[read_index], "--root=", 7) == 0) {
            value = argv[read_index] + 7;
        }

        if (value != NULL) {
            if (!copy_launcher_text(root, root_size, value) ||
                root[0] == '\0') {
                return -1;
            }
            has_override = 1;
            continue;
        }

        argv[write_index++] = argv[read_index];
    }

    argv[write_index] = NULL;
    *argc = write_index;
    return has_override;
}

const char *launcher_exit_reason(int return_code) {
    switch (return_code) {
        case CELUNE_EXIT_FAILURE:
            return "Celune experienced a general failure.";
        case CELUNE_EXIT_NO_ANSI:
            return "Celune did not find an ANSI capable terminal.";
        case CELUNE_EXIT_ALREADY_RUNNING:
            return "Celune is already running.";
        case CELUNE_EXIT_MISSING_DEPENDENCIES:
            return "Celune is missing required dependencies.";
        case CELUNE_EXIT_UNKNOWN_ARGS:
            return "Celune received an unknown command.";
        case CELUNE_EXIT_BAD_PYTHON:
            return "Celune is running on an unsupported Python interpreter.";
        case CELUNE_EXIT_PENDING_RESTART:
            return NULL;
        case CELUNE_EXIT_LAUNCHER_LOST:
            return NULL;
        case CELUNE_EXIT_UNSUPPORTED_CPU:
            return "Celune requires AVX instructions, but this CPU does not provide them.";
        case CELUNE_EXIT_CELINE_DAY_SIX_SEVEN:
        case CELUNE_EXIT_CELINE_DAY:
            return NULL;
        default:
            return "Celune has crashed.";
    }
}

int main(int argc, char **argv) {
    retain_launcher_id();
    launcher_setup_terminal();
    int main_operation = launcher_is_main_operation(argc, argv);

    if (main_operation && !launcher_cpu_supports_avx()) {
        launcher_reset_terminal_state();
        launcher_prepare_failure_output();
        launcher_report_failure(CELUNE_EXIT_UNSUPPORTED_CPU);
        launcher_wait_after_failure();
        return CELUNE_EXIT_UNSUPPORTED_CPU;
    }

    int return_code = launcher_run(argc, argv);

    if (main_operation &&
        (return_code != 0 || launcher_startup_was_interrupted())) {
        launcher_reset_terminal_state();
        launcher_prepare_failure_output();
        launcher_report_failure(return_code);
        launcher_wait_after_failure();
        return return_code;
    }

    launcher_reset_terminal_state();

    return return_code;
}
