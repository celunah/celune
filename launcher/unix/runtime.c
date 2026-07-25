#define _POSIX_C_SOURCE 200809L

#include "launcher_platform.h"

#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define printfe(...) do { fprintf(stderr, __VA_ARGS__); } while (0)
static int launcher_child_failed = 0;

static int file_exists(const char *path) {
    return access(path, F_OK) == 0;
}

static int copy_text(char *dest, size_t size, const char *src) {
    size_t len = strlen(src);

    if (len >= size) {
        return 0;
    }

    memcpy(dest, src, len + 1);
    return 1;
}

static int parent_dir_of(const char *path, char *out, size_t size) {
    if (!copy_text(out, size, path)) {
        return 0;
    }

    char *last = strrchr(out, '/');
    if (last == NULL) {
        return 0;
    }

    *last = '\0';
    return 1;
}

static int find_repo_root(const char *start_dir, char *out, size_t size) {
    char current[1024];
    if (!copy_text(current, sizeof(current), start_dir)) {
        return 0;
    }

    while (1) {
        char pyvenv_cfg[1200];
        int written = snprintf(pyvenv_cfg, sizeof(pyvenv_cfg), "%s/.venv/pyvenv.cfg", current);
        if (written > 0 && (size_t)written < sizeof(pyvenv_cfg) && file_exists(pyvenv_cfg)) {
            return copy_text(out, size, current);
        }

        char parent[1024];
        if (!parent_dir_of(current, parent, sizeof(parent))) {
            break;
        }

        if (strcmp(parent, current) == 0 || parent[0] == '\0') {
            break;
        }

        if (!copy_text(current, sizeof(current), parent)) {
            return 0;
        }
    }

    return 0;
}

static int get_exe_dir(char *out, size_t size) {
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

static int spawn_update_helper_unix(
    const char *python,
    const char *main_py,
    const char *launcher_path,
    const char *repo_root,
    int argc,
    char **argv
) {
    pid_t pid = fork();
    if (pid == -1) {
        perror("fork failed");
        return 0;
    }

    if (pid == 0) {
        launcher_restore_child_terminal();
        char pid_text[32];
        snprintf(pid_text, sizeof(pid_text), "%ld", (long)getppid());

        char **args = malloc(((size_t)argc + 5U) * sizeof(char *));
        if (args == NULL) {
            perror("malloc failed");
            _exit(1);
        }

        args[0] = (char *)python;
        args[1] = (char *)main_py;
        args[2] = "__apply_update";
        args[3] = pid_text;
        args[4] = (char *)launcher_path;
        for (int i = 1; i < argc; i++) {
            args[i + 4] = argv[i];
        }
        args[argc + 4] = NULL;

        if (chdir(repo_root) != 0) {
            perror("chdir failed");
            _exit(1);
        }

        execv(args[0], args);
        perror("execv failed");
        _exit(1);
    }

    return 1;
}

int launcher_run(int argc, char **argv) {
    char base[1024];
    char repo_root[1024];
    char launcher[1024];
    char target[1024];
    char python[1024];
    char main_py[1024];
    char setup_py[1024];

    char launcher_pid[32];
    snprintf(launcher_pid, sizeof(launcher_pid), "%ld", (long)getpid());

    if (setenv("CELUNE_LAUNCHER", "1", 1) != 0 ||
        setenv("CELUNE_LAUNCHER_PID", launcher_pid, 1) != 0) {
        printfe("Celune could not configure launcher environment variables.\n");
        return 1;
    }

    if (!get_exe_dir(base, sizeof(base))) {
        printfe("Celune could not determine the launcher location.\n");
        return 1;
    }

    if (!find_repo_root(base, repo_root, sizeof(repo_root))) {
        printfe("Celune could not find the repository root with a Python virtual environment.\n");
        return 1;
    }

    int launcher_len = snprintf(launcher, sizeof(launcher), "%s/celune", base);
    int target_len = snprintf(target, sizeof(target), "%s/celune-bin", base);
    int python_len = snprintf(python, sizeof(python), "%s/.venv/bin/python", repo_root);
    int main_py_len = snprintf(main_py, sizeof(main_py), "%s/main.py", repo_root);
    int setup_py_len = snprintf(setup_py, sizeof(setup_py), "%s/setup.py", repo_root);

    if (launcher_len < 0 || (size_t)launcher_len >= sizeof(launcher) ||
        target_len < 0 || (size_t)target_len >= sizeof(target) ||
        python_len < 0 || (size_t)python_len >= sizeof(python) ||
        main_py_len < 0 || (size_t)main_py_len >= sizeof(main_py) ||
        setup_py_len < 0 || (size_t)setup_py_len >= sizeof(setup_py)) {
        printfe("Celune cannot start in this location, the path is too long.\n");
        return 1;
    }

    if (access(target, X_OK) == 0) {
        pid_t pid = fork();
        if (pid == -1) {
            perror("fork failed");
            return 1;
        }

        if (pid == 0) {
            launcher_restore_child_terminal();
            char **args = malloc(((size_t)argc + 1U) * sizeof(char *));
            if (args == NULL) {
                perror("malloc failed");
                _exit(1);
            }

            args[0] = target;
            for (int i = 1; i < argc; i++) {
                args[i] = argv[i];
            }
            args[argc] = NULL;

            if (chdir(repo_root) != 0) {
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
                int exit_code = WEXITSTATUS(status);
                if (exit_code == CELUNE_EXIT_PENDING_UPDATE) {
                    printfe("%s\n", launcher_exit_reason(CELUNE_EXIT_PENDING_UPDATE));
                    if (!spawn_update_helper_unix(python, main_py, launcher, repo_root, argc, argv)) {
                        printfe("Celune could not start her update helper.\n");
                        return 1;
                    }
                    return 0;
                }
                launcher_child_failed = exit_code != 0;
                return exit_code;
            }
            else if (WIFSIGNALED(status)) {
                int sig = WTERMSIG(status);
                launcher_child_failed = 1;
                return 128 + sig;
            }
        }

        return 1;
    }

    if (access(python, X_OK) != 0) {
        const char *system_python[] = {"python3", "python"};
        int found_system_python = 0;
        int setup_status = 1;

        if (access(setup_py, R_OK) != 0) {
            printfe("Celune: Python environment is missing and setup.py is unavailable.\n");
            return 1;
        }

        printfe("Celune: Python environment missing; running setup.py.\n");

        for (size_t i = 0; i < sizeof(system_python) / sizeof(system_python[0]); i++) {
            pid_t setup_pid = fork();
            if (setup_pid == -1) {
                perror("fork failed");
                return 1;
            }

            if (setup_pid == 0) {
                launcher_restore_child_terminal();
                char *args[] = {(char *)system_python[i], setup_py, NULL};
                if (chdir(repo_root) != 0) {
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
            printfe("Celune: no system Python interpreter was found for setup.py.\n");
            return 1;
        }

        if (!WIFEXITED(setup_status) || WEXITSTATUS(setup_status) != 0) {
            printfe("Celune: setup.py failed.\n");
            return WIFEXITED(setup_status) ? WEXITSTATUS(setup_status) : 1;
        }

        if (access(python, X_OK) != 0) {
            printfe("Celune: setup.py completed but the Python environment is still unavailable.\n");
            return 1;
        }
    }

    pid_t pid = fork();
    if (pid == -1) {
        perror("fork failed");
        return 1;
    }

    if (pid == 0) {
        launcher_restore_child_terminal();
        char **args = malloc(((size_t)argc + 2U) * sizeof(char *));
        if (args == NULL) {
            perror("malloc failed");
            _exit(1);
        }

        args[0] = python;
        args[1] = main_py;
        for (int i = 1; i < argc; i++) {
            args[i + 1] = argv[i];
        }
        args[argc + 1] = NULL;

        if (chdir(repo_root) != 0) {
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
            launcher_child_failed = WEXITSTATUS(status) != 0;
            return WEXITSTATUS(status);
        }
        else if (WIFSIGNALED(status)) {
            int sig = WTERMSIG(status);
            launcher_child_failed = 1;
            return 128 + sig;
        }
    }

    return 1;
}

void launcher_report_failure(int return_code) {
    if (launcher_startup_was_interrupted() || return_code == 128 + SIGINT) {
        printfe("Startup was interrupted.\n");
        return;
    }

    if (!launcher_child_failed) {
        return;
    }

    const char *reason = launcher_exit_reason(return_code);
    if (reason != NULL) {
        printfe("%s\n", reason);
    }
}
