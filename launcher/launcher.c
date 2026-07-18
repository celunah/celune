#include "launcher_platform.h"

int main(int argc, char **argv) {
    launcher_setup_terminal();
    int return_code = launcher_run(argc, argv);

    launcher_reset_terminal_state();

    if (return_code != 0) {
        launcher_wait_after_failure();
    }

    return return_code;
}
