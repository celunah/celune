#ifndef CELUNE_LAUNCHER_PLATFORM_H
#define CELUNE_LAUNCHER_PLATFORM_H

int launcher_run(int argc, char **argv);
void launcher_reset_terminal_state(void);
void launcher_wait_after_failure(void);

#endif
