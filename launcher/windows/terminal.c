#include "launcher_platform.h"

#include <windows.h>
#include <conio.h>

#include <stdio.h>

static volatile LONG startup_interrupted = 0;

static BOOL WINAPI ignore_console_interrupt(DWORD event_type) {
    if (event_type == CTRL_C_EVENT || event_type == CTRL_BREAK_EVENT) {
        InterlockedExchange(&startup_interrupted, 1);
        return TRUE;
    }

    return FALSE;
}

static DWORD saved_console_input_mode = 0;
static DWORD saved_console_output_mode = 0;
static BOOL saved_console_modes = FALSE;

int launcher_startup_was_interrupted(void) {
    return InterlockedCompareExchange(&startup_interrupted, 0, 0) != 0;
}

void launcher_setup_terminal(void) {
    HANDLE input = GetStdHandle(STD_INPUT_HANDLE);
    HANDLE output = GetStdHandle(STD_OUTPUT_HANDLE);

    saved_console_modes =
        input != INVALID_HANDLE_VALUE && output != INVALID_HANDLE_VALUE &&
        GetConsoleMode(input, &saved_console_input_mode) &&
        GetConsoleMode(output, &saved_console_output_mode);

    SetConsoleCtrlHandler(ignore_console_interrupt, TRUE);
}

void launcher_reset_terminal_state(void) {
    HANDLE input = GetStdHandle(STD_INPUT_HANDLE);
    HANDLE output = GetStdHandle(STD_OUTPUT_HANDLE);
    const char reset_sequences[] =
        "\x1b[0m"
        "\x1b[?25h"
        "\x1b[?1000l"
        "\x1b[?1002l"
        "\x1b[?1003l"
        "\x1b[?1006l"
        "\x1b[?1015l"
        "\x1b[?1049l"
        "\x1b[?2004l";

    if (saved_console_modes) {
        DWORD current_output_mode;
        if (GetConsoleMode(output, &current_output_mode)) {
            DWORD reset_output_mode =
                current_output_mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING;
            if (SetConsoleMode(output, reset_output_mode)) {
                DWORD written;
                WriteFile(
                    output,
                    reset_sequences,
                    (DWORD)(sizeof(reset_sequences) - 1),
                    &written,
                    NULL
                );
                FlushFileBuffers(output);
            }
        }

        SetConsoleMode(output, saved_console_output_mode);
        SetConsoleMode(input, saved_console_input_mode);
        FlushConsoleInputBuffer(input);
    }

    SetConsoleCtrlHandler(ignore_console_interrupt, FALSE);
}

void launcher_wait_after_failure(void) {
    fputs("\nPress any key to exit.\n", stderr);
    fflush(stderr);
    _getch();
}

void launcher_restore_child_terminal(void) {
    SetConsoleCtrlHandler(ignore_console_interrupt, FALSE);
}
