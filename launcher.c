#ifdef __linux__
#include <unistd.h>
#include <sys/wait.h>
#elif defined(_WIN32)
#include <windows.h>
#include <stdlib.h>
#endif

#include <stdio.h>
#define printfe(...) fprintf(stderr, __VA_ARGS__)

#if defined(__linux__) || (defined(__APPLE__) && defined(__MACH__))
int run_unix(void) {
    const char *python = "./.venv/bin/python";

    if (access(python, X_OK) != 0) {
        printfe("Python interpreter not found or is not working.\n");
        printfe("Celune needs a Python interpreter to operate.\n");
        return 1;
    }

    char *args[] = {
        "./.venv/bin/python",
        "main.py",
        NULL,
    };

    pid_t pid = fork();
    if (pid == -1) {
        perror("fork failed");
        return 1;
    }

    if (pid == 0) {
        char *args[] = {"./.venv/bin/python", "main.py", NULL};
        execv(python, args);

        perror("execv failed");
        _exit(1);
    } else {
        int status;
        waitpid(pid, &status, 0);

        if (WIFEXITED(status)) {
            return WEXITSTATUS(status);
        }
    }

    return 1;
}
#elif defined(_WIN32)
int run_windows(void) {
    const char *python = ".\\.venv\\Scripts\\python.exe";

    DWORD attr = GetFileAttributesA(python);
    if (attr == INVALID_FILE_ATTRIBUTES) {
        printfe("Python virtual environment and/or interpreter was not found or isn't working.\n");
        printfe("Celune needs a working Python interpreter and virtual environment to operate.\n");
        return 1;
    }

    STARTUPINFOA si = {0};
    PROCESS_INFORMATION pi = {0};
    si.cb = sizeof(si);

    si.dwFlags = STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_SHOW;

    char cmd[512];
    snprintf(cmd, sizeof(cmd), "\"%s\" main.py", python);

    BOOL ok = CreateProcessA(
        NULL,
        cmd,
        NULL,
        NULL,
        FALSE,
        0,
        NULL,
        NULL,
        &si,
        &pi
    );

    if (!ok) {
        printfe("Celune could not launch Python.\n%lu\n", GetLastError());
        return 1;
    }

    WaitForSingleObject(pi.hProcess, INFINITE);

    DWORD exit_code = 1;
    GetExitCodeProcess(pi.hProcess, &exit_code);

    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);

    return (int)exit_code;
}
#endif

int main(void) {
#if defined(__linux__) || (defined(__APPLE__) && defined(__MACH__))
    return run_unix();
#elif defined(_WIN32)
    return run_windows();
#else
    fprintf("Unsupported operating system.\n");
    fprintf("How do you even run Celune on this thing you have?");
    return 1;
#endif
return 1;
}