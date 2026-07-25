#include "launcher_platform.h"

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
