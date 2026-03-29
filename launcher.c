#ifdef __linux__
#include <unistd.h>
#include <sys/wait.h>
#elif defined(_WIN32)
#include <windows.h>
#include <stdlib.h>
#endif

#include <stdio.h>

#ifdef __linux__
int run_unix(void) {
    const char *python = "./.venv/bin/python";

    if (access(python, X_OK) != 0) {
        fprintf(stderr, "Python interpreter not found or is not working.\n");
        fprintf(stderr, "Celune needs a Python interpreter to operate.\n");
        return 1;
    }

    char *args[] = {
        (char*)"./.venv/bin/python",
        (char*)"main.py",
        NULL,
    };

    execvp(args[0], args);

    perror("execvp failed");
    return 1;
}
#elif defined(_WIN32)
int run_windows(void) {
    const char *python = ".\\.venv\\Scripts\\python.exe";

    DWORD attr = GetFileAttributesA(python);
    if (attr == INVALID_FILE_ATTRIBUTES) {
        fprintf(stderr, "Python interpreter not found or is not working.\n");
        fprintf(stderr, "Celune needs a Python interpreter to operate.\n");
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
        fprintf(stderr, "Celune could not launch Python.\n%lu\n", GetLastError());
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
#ifdef __linux__
    return run_unix();
#elif defined(_WIN32)
    return run_windows();
#else
    fprintf(stderr, "Unsupported operating system.\n");
    fprintf(stderr, "Celune expects a Windows or Linux system.");
    return 1;
#endif
}