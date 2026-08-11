#define _POSIX_C_SOURCE 200809L

#include "launcher_platform.h"

#include <unistd.h>
#include <termios.h>
#include <signal.h>

#include <stdio.h>

static struct termios saved_terminal_state;
static int saved_terminal_state_valid = 0;
static volatile sig_atomic_t startup_interrupted = 0;

static void record_interrupt(int signal_number) {
    (void)signal_number;
    startup_interrupted = 1;
}

int launcher_startup_was_interrupted(void) {
    return startup_interrupted != 0;
}

void launcher_setup_terminal(void) {
    if (isatty(STDIN_FILENO) && tcgetattr(STDIN_FILENO, &saved_terminal_state) == 0) {
        saved_terminal_state_valid = 1;
    }

    signal(SIGINT, record_interrupt);
#ifdef SIGQUIT
    signal(SIGQUIT, SIG_IGN);
#endif
}

void launcher_reset_terminal_state(void) {
    static const char reset_sequences[] =
        "\033[0m"
        "\033[?25h"
        "\033[?1000l"
        "\033[?1002l"
        "\033[?1003l"
        "\033[?1006l"
        "\033[?1015l"
        "\033[?1049l"
        "\033[?2004l";

    if (!saved_terminal_state_valid) {
        return;
    }

    if (isatty(STDOUT_FILENO)) {
        if (write(STDOUT_FILENO, reset_sequences, sizeof(reset_sequences) - 1) < 0) {
            /* Terminal cleanup is best effort. */
        }
    }

    tcsetattr(STDIN_FILENO, TCSANOW, &saved_terminal_state);
    tcflush(STDIN_FILENO, TCIFLUSH);
}

void launcher_prepare_failure_output(void) {
    if (isatty(STDERR_FILENO)) {
        fputs("\033[999B\r", stderr);
        fflush(stderr);
    }
}

void launcher_wait_after_failure(void) {
    struct termios oldt;
    struct termios newt;

    if (tcgetattr(STDIN_FILENO, &oldt) != 0) {
        return;
    }

    fputs("\nPress any key to exit.\n", stderr);
    fflush(stderr);

    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    getchar();
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
}

void launcher_restore_child_terminal(void) {
    signal(SIGINT, SIG_DFL);
#ifdef SIGQUIT
    signal(SIGQUIT, SIG_DFL);
#endif
}
