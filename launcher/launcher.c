#include "launcher_platform.h"

#include <stddef.h>

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
        case CELUNE_EXIT_PENDING_UPDATE:
            return "Celune has a pending update.";
        case CELUNE_EXIT_CELINE_DAY_SIX_SEVEN:
        case CELUNE_EXIT_CELINE_DAY:
            return NULL;
        default:
            return "Celune has crashed.";
    }
}

int main(int argc, char **argv) {
    launcher_setup_terminal();
    int return_code = launcher_run(argc, argv);

    if (return_code != 0 || launcher_startup_was_interrupted()) {
        launcher_report_failure(return_code);
        launcher_wait_after_failure();
    }

    launcher_reset_terminal_state();

    return return_code;
}
