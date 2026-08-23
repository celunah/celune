#include "launcher_platform.h"

#include <windows.h>
#include <conio.h>

#include <stdio.h>
#include <stdlib.h>

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
static CONSOLE_SCREEN_BUFFER_INFO startup_console_info;
static BOOL startup_console_info_valid = FALSE;

int launcher_startup_was_interrupted(void) {
    return InterlockedCompareExchange(&startup_interrupted, 0, 0) != 0;
}

void launcher_setup_terminal(void) {
    HANDLE input = GetStdHandle(STD_INPUT_HANDLE);
    HANDLE output = GetStdHandle(STD_ERROR_HANDLE);

    startup_console_info_valid =
        output != INVALID_HANDLE_VALUE &&
        GetConsoleScreenBufferInfo(output, &startup_console_info);
    saved_console_modes =
        input != INVALID_HANDLE_VALUE && output != INVALID_HANDLE_VALUE &&
        GetConsoleMode(input, &saved_console_input_mode) &&
        GetConsoleMode(output, &saved_console_output_mode);

    SetConsoleCtrlHandler(ignore_console_interrupt, TRUE);
}

void launcher_reset_terminal_state(void) {
    HANDLE input = GetStdHandle(STD_INPUT_HANDLE);
    HANDLE output = GetStdHandle(STD_ERROR_HANDLE);
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

static BOOL console_row_has_text(
    HANDLE output,
    SHORT left,
    SHORT row,
    DWORD width,
    WCHAR *characters
) {
    COORD position = {left, row};
    DWORD characters_read = 0;

    if (!ReadConsoleOutputCharacterW(
            output,
            characters,
            width,
            position,
            &characters_read
        )) {
        return FALSE;
    }

    for (DWORD index = 0; index < characters_read; index++) {
        WCHAR character = characters[index];
        if (character != L' ' && character != L'\0' && character != L'\r' &&
            character != L'\n' && character != L'\t') {
            return TRUE;
        }
    }

    return FALSE;
}

static BOOL find_last_output_row(
    HANDLE output,
    const CONSOLE_SCREEN_BUFFER_INFO *info,
    SHORT *last_row
) {
    SHORT first_row = info->srWindow.Top;
    SHORT last_visible_row = info->srWindow.Bottom;
    DWORD width = (DWORD)(info->srWindow.Right - info->srWindow.Left + 1);
    WCHAR *characters = (WCHAR *)malloc(width * sizeof(WCHAR));

    if (characters == NULL) {
        return FALSE;
    }

    if (startup_console_info_valid &&
        startup_console_info.dwCursorPosition.Y >= first_row &&
        startup_console_info.dwCursorPosition.Y <= last_visible_row) {
        first_row = startup_console_info.dwCursorPosition.Y;
    }

    BOOL found = FALSE;
    for (int row = first_row; row <= last_visible_row; row++) {
        if (console_row_has_text(
                output,
                info->srWindow.Left,
                (SHORT)row,
                width,
                characters
            )) {
            *last_row = (SHORT)row;
            found = TRUE;
        }
    }

    free(characters);
    return found;
}

void launcher_prepare_failure_output(void) {
    HANDLE output = GetStdHandle(STD_ERROR_HANDLE);
    CONSOLE_SCREEN_BUFFER_INFO info;
    COORD failure_start;

    if (output == INVALID_HANDLE_VALUE ||
        !GetConsoleScreenBufferInfo(output, &info)) {
        return;
    }

    SHORT last_output_row;
    if (find_last_output_row(output, &info, &last_output_row)) {
        failure_start.X = info.srWindow.Left;
        failure_start.Y = last_output_row;
        if (last_output_row < info.dwSize.Y - 1) {
            failure_start.Y = (SHORT)(last_output_row + 1);
            SetConsoleCursorPosition(output, failure_start);
            return;
        }

        SetConsoleCursorPosition(output, failure_start);
        fputs("\r\n", stderr);
        fflush(stderr);
        return;
    }

    failure_start = info.dwCursorPosition;
    if (failure_start.X != info.srWindow.Left &&
        failure_start.Y < info.srWindow.Bottom) {
        failure_start.Y++;
    }
    failure_start.X = info.srWindow.Left;
    SetConsoleCursorPosition(output, failure_start);
}

void launcher_wait_after_failure(void) {
    fputs("\nPress any key to exit.\n", stderr);
    fflush(stderr);
    _getch();
}

void launcher_restore_child_terminal(void) {
    SetConsoleCtrlHandler(ignore_console_interrupt, FALSE);
}
