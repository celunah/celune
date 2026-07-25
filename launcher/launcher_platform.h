#ifndef CELUNE_LAUNCHER_PLATFORM_H
#define CELUNE_LAUNCHER_PLATFORM_H

#define CELUNE_EXIT_SUCCESS 0
#define CELUNE_EXIT_FAILURE 1
#define CELUNE_EXIT_NO_ANSI 2
#define CELUNE_EXIT_ALREADY_RUNNING 3
#define CELUNE_EXIT_MISSING_DEPENDENCIES 4
#define CELUNE_EXIT_UNKNOWN_ARGS 5
#define CELUNE_EXIT_BAD_PYTHON 6
#define CELUNE_EXIT_PENDING_UPDATE 7
#define CELUNE_EXIT_CELINE_DAY_SIX_SEVEN 67
#define CELUNE_EXIT_CELINE_DAY 103

int launcher_run(int argc, char **argv);
const char *launcher_exit_reason(int return_code);
int launcher_startup_was_interrupted(void);
void launcher_report_failure(int return_code);
void launcher_setup_terminal(void);
void launcher_reset_terminal_state(void);
void launcher_wait_after_failure(void);
void launcher_restore_child_terminal(void);

#endif
