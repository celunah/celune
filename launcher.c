#if defined(__linux__)
#include <unistd.h>
#include <sys/wait.h>
#elif defined(__APPLE__) && defined(__MACH__)
#include <unistd.h>
#include <sys/wait.h>
#include <mach-o/dyld.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define printfe(...) do { fprintf(stderr, __VA_ARGS__); } while (0)

#if defined(__linux__)
int get_exe_dir(char *out, size_t size) {
    ssize_t len = readlink("/proc/self/exe", out, size - 1);

    if (len == -1 || len >= (ssize_t)(size - 1)) {
        return 0;
    }

    out[len] = '\0';

    char *last = strrchr(out, '/');
    if (last) {
        *last = '\0';
        return 1;
    }

    return 0;
}
#elif defined(__APPLE__) && defined(__MACH__)
int get_exe_dir(char *out, size_t size) {
    uint32_t sz = (uint32_t)size;

    if (_NSGetExecutablePath(out, &sz) != 0) {
        return 0;
    }

    char resolved[1024];
    if (realpath(out, resolved) == NULL) {
        return 0;
    }

    strncpy(out, resolved, size - 1);
    out[size - 1] = '\0';

    char *last = strrchr(out, '/');
    if (last) {
        *last = '\0';
        return 1;
    }

    return 0;
}
#elif defined(_WIN32)
int get_exe_dir(char *out, size_t size) {
    DWORD len = GetModuleFileNameA(NULL, out, (DWORD)size);

    if (len == 0 || len == size) {
        return 0;
    }

    char *last = strrchr(out, '\\');
    if (last) {
        *last = '\0';
        return 1;
    }

    return 0;
}
#endif

#if defined(__linux__) || (defined(__APPLE__) && defined(__MACH__))
int run_unix(void) {
	char base[1024];
	char python[1024];
	char main_py[1024];

	setenv("CELUNE_LAUNCHER", "1", 1);

	if (!get_exe_dir(base, sizeof(base))) {
	    printfe("Celune could not determine the launcher location.\n");
	    return 1;
	}

	int python_len = snprintf(python, sizeof(python), "%s/.venv/bin/python", base);
	int main_py_len = snprintf(main_py, sizeof(main_py), "%s/main.py", base);

	if (python_len < 0 || (size_t)python_len >= sizeof(python) ||
	    main_py_len < 0 || (size_t)main_py_len >= sizeof(main_py)) {
	    printfe("Celune cannot start in this location, the path is too long.\n");
	    return 1;
	   }
	
    if (access(python, X_OK) != 0) {
        printfe("Python virtual environment and/or interpreter was not found or isn't working.\n");
        printfe("Celune needs a working Python interpreter and virtual environment to operate.\n");
        return 1;
    }

    pid_t pid = fork();
    if (pid == -1) {
        perror("fork failed");
        return 1;
    }

    if (pid == 0) {
        char *args[] = {python, main_py, NULL};
        if (chdir(base) != 0) {
            perror("chdir failed");
            _exit(1);
        }
        execv(args[0], args);

        perror("execv failed");
        _exit(1);
    } else {
        int status;
        waitpid(pid, &status, 0);

        if (WIFEXITED(status)) {
            return WEXITSTATUS(status);
        }
		else if (WIFSIGNALED(status)) {
			int sig = WTERMSIG(status);

			printfe("Celune was killed by signal %d.\n", sig);
			return 128 + sig;
		}
    }

    return 1;
}
#elif defined(_WIN32)
int run_windows(void) {
	char base[1024];
	char python[1024];
	char main_py[1024];

	SetEnvironmentVariableA("CELUNE_LAUNCHER", "1");

	if (!get_exe_dir(base, sizeof(base))) {
	    printfe("Celune could not determine the launcher location.\n");
	    return 1;
	}

	int python_len = snprintf(python, sizeof(python), "%s\\.venv\\Scripts\\python.exe", base);
	int main_py_len = snprintf(main_py, sizeof(main_py), "%s\\main.py", base);

	if (python_len < 0 || (size_t)python_len >= sizeof(python) ||
	    main_py_len < 0 || (size_t)main_py_len >= sizeof(main_py)) {
	    printfe("Celune cannot start in this location, the path is too long.\n");
	    return 1;
	}

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

    char cmd[2200];
    int written = snprintf(cmd, sizeof(cmd), "\"%s\" \"%s\"", python, main_py);
    if (written < 0 || (size_t)written >= sizeof(cmd)) {
        printfe("Celune cannot start in this location, the command line is too long.\n");
        return 1;
    }

    BOOL ok = CreateProcessA(
        NULL,
        cmd,
        NULL,
        NULL,
        FALSE,
        0,
        NULL,
        base,
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
    printfe("Unsupported operating system.\n");
    printfe("How do you even run Celune on this thing you have?\n");
    return 1;
#endif
}
