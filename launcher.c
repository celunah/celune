#define _POSIX_C_SOURCE 200809L

#ifdef __linux__
#include <unistd.h>
#include <sys/wait.h>
#include <termios.h>
#elif defined(_WIN32)
#include <windows.h>
#include <conio.h>
#endif

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define printfe(...) do { fprintf(stderr, __VA_ARGS__); } while (0)

#ifdef __linux__
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

#ifdef __linux__
int run_unix(void) {
	char base[1024];
	char python[1024];
	char main_py[1024];
	char setup_py[1024];

	setenv("CELUNE_LAUNCHER", "1", 1);

	if (!get_exe_dir(base, sizeof(base))) {
	    printfe("Celune could not determine the launcher location.\n");
	    return 1;
	}

	int python_len = snprintf(python, sizeof(python), "%s/.venv/bin/python", base);
	int main_py_len = snprintf(main_py, sizeof(main_py), "%s/main.py", base);
	int setup_py_len = snprintf(setup_py, sizeof(setup_py), "%s/setup.py", base);

	if (python_len < 0 || (size_t)python_len >= sizeof(python) ||
	    main_py_len < 0 || (size_t)main_py_len >= sizeof(main_py) ||
	    setup_py_len < 0 || (size_t)setup_py_len >= sizeof(setup_py)) {
	    printfe("Celune cannot start in this location, the path is too long.\n");
	    return 1;
	   }
	
    if (access(python, X_OK) != 0) {
        const char *system_python[] = {"python3", "python"};
        int found_system_python = 0;
        int setup_status = 1;

        if (access(setup_py, R_OK) != 0) {
            printfe("Python virtual environment and/or interpreter was not found or isn't working.\n");
            printfe("Celune needs setup.py to create its virtual environment.\n");
            return 1;
        }

        printfe("Python virtual environment was not found. Running setup.py...\n");

        for (size_t i = 0; i < sizeof(system_python) / sizeof(system_python[0]); i++) {
            pid_t setup_pid = fork();
            if (setup_pid == -1) {
                perror("fork failed");
                return 1;
            }

            if (setup_pid == 0) {
                char *args[] = {(char *)system_python[i], setup_py, NULL};
                if (chdir(base) != 0) {
                    perror("chdir failed");
                    _exit(1);
                }
                execvp(args[0], args);
                _exit(127);
            }

            if (waitpid(setup_pid, &setup_status, 0) == -1) {
                perror("waitpid failed");
                return 1;
            }

            if (WIFEXITED(setup_status) && WEXITSTATUS(setup_status) == 127) {
                continue;
            }

            found_system_python = 1;
            break;
        }

        if (!found_system_python) {
            printfe("Celune could not find a system Python interpreter to run setup.py.\n");
            printfe("Install Python 3.12 or 3.13 and run Celune again.\n");
            return 1;
        }

        if (!WIFEXITED(setup_status) || WEXITSTATUS(setup_status) != 0) {
            printfe("Celune setup failed.\n");
            return WIFEXITED(setup_status) ? WEXITSTATUS(setup_status) : 1;
        }

        if (access(python, X_OK) != 0) {
            printfe("Python virtual environment and/or interpreter was not found or isn't working.\n");
            printfe("Celune needs a working Python interpreter and virtual environment to operate.\n");
            return 1;
        }
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
	char setup_py[1024];

	SetEnvironmentVariableA("CELUNE_LAUNCHER", "1");

	if (!get_exe_dir(base, sizeof(base))) {
	    printfe("Celune could not determine the launcher location.\n");
	    return 1;
	}

	int python_len = snprintf(python, sizeof(python), "%s\\.venv\\Scripts\\python.exe", base);
	int main_py_len = snprintf(main_py, sizeof(main_py), "%s\\main.py", base);
	int setup_py_len = snprintf(setup_py, sizeof(setup_py), "%s\\setup.py", base);

	if (python_len < 0 || (size_t)python_len >= sizeof(python) ||
	    main_py_len < 0 || (size_t)main_py_len >= sizeof(main_py) ||
	    setup_py_len < 0 || (size_t)setup_py_len >= sizeof(setup_py)) {
	    printfe("Celune cannot start in this location, the path is too long.\n");
	    return 1;
	}

    DWORD attr = GetFileAttributesA(python);
    if (attr == INVALID_FILE_ATTRIBUTES) {
        DWORD setup_attr = GetFileAttributesA(setup_py);
        if (setup_attr == INVALID_FILE_ATTRIBUTES) {
            printfe("Python virtual environment and/or interpreter was not found or isn't working.\n");
            printfe("Celune needs setup.py to create its virtual environment.\n");
            return 1;
        }

        printfe("Python virtual environment was not found. Running setup.py...\n");

        STARTUPINFOA setup_si = {0};
        PROCESS_INFORMATION setup_pi = {0};
        setup_si.cb = sizeof(setup_si);
        setup_si.dwFlags = STARTF_USESHOWWINDOW;
        setup_si.wShowWindow = SW_SHOW;

        char setup_cmd[2200];
        int setup_written = snprintf(setup_cmd, sizeof(setup_cmd), "python.exe \"%s\"", setup_py);
        if (setup_written < 0 || (size_t)setup_written >= sizeof(setup_cmd)) {
            printfe("Celune cannot start setup.py, the command line is too long.\n");
            return 1;
        }

        BOOL setup_ok = CreateProcessA(
            NULL,
            setup_cmd,
            NULL,
            NULL,
            FALSE,
            0,
            NULL,
            base,
            &setup_si,
            &setup_pi
        );

        if (!setup_ok) {
            DWORD error = GetLastError();
            if (error == ERROR_FILE_NOT_FOUND || error == ERROR_PATH_NOT_FOUND) {
                printfe("Celune could not find a system Python interpreter to run setup.py.\n");
                printfe("Install Python 3.12 or 3.13 and run Celune again.\n");
            } else {
                printfe("Celune could not launch setup.py.\n%lu\n", error);
            }
            return 1;
        }

        WaitForSingleObject(setup_pi.hProcess, INFINITE);

        DWORD setup_exit_code = 1;
        GetExitCodeProcess(setup_pi.hProcess, &setup_exit_code);

        CloseHandle(setup_pi.hThread);
        CloseHandle(setup_pi.hProcess);

        if (setup_exit_code != 0) {
            printfe("Celune setup failed.\n");
            return (int)setup_exit_code;
        }

        attr = GetFileAttributesA(python);
        if (attr == INVALID_FILE_ATTRIBUTES) {
            printfe("Python virtual environment and/or interpreter was not found or isn't working.\n");
            printfe("Celune needs a working Python interpreter and virtual environment to operate.\n");
            return 1;
        }
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
#ifdef __linux__
    int return_code = run_unix();

	if ( return_code != 0 ) {
		struct termios oldt, newt;
        if (tcgetattr(STDIN_FILENO, &oldt) == 0) {
            newt = oldt;
            newt.c_lflag &= ~(ICANON | ECHO);
            tcsetattr(STDIN_FILENO, TCSANOW, &newt);
            getchar();
            tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
        }
	}

	return return_code;
#elif defined(_WIN32)
    int return_code = run_windows();

	if ( return_code != 0 ) {
		_getch();
	}

    return return_code;
#else
    printfe("Unsupported operating system.\n");
    printfe("How do you even run Celune on this thing you have?\n");
    return 1;
#endif
}
