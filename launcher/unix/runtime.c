#define _POSIX_C_SOURCE 200809L

#include "launcher_platform.h"

#include <unistd.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <dirent.h>
#include <signal.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#define printfe(...) do { fprintf(stderr, __VA_ARGS__); } while (0)
#define LAUNCHER_SEARCH_MAX_DEPTH 8
#define LAUNCHER_SEARCH_MAX_DIRECTORIES 10000
#define LAUNCHER_SEARCH_MAX_MILLISECONDS 5000

static int launcher_child_failed = 0;
static size_t searched_directories = 0;
static unsigned long long search_deadline = 0;

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

static unsigned long long monotonic_milliseconds(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        return 0;
    }

    return (unsigned long long)now.tv_sec * 1000ULL +
           (unsigned long long)now.tv_nsec / 1000000ULL;
}

static int skip_search_directory(const char *name) {
    return strcmp(name, ".git") == 0 || strcmp(name, ".venv") == 0 ||
           strcmp(name, "node_modules") == 0 || strcmp(name, "proc") == 0 ||
           strcmp(name, "sys") == 0 || strcmp(name, "dev") == 0 ||
           strcmp(name, "run") == 0;
}

static int search_runtime_directory(
    const char *directory,
    int depth,
    char *runtime_dir,
    size_t runtime_dir_size,
    char *repo_root,
    size_t repo_root_size
) {
    if (searched_directories >= LAUNCHER_SEARCH_MAX_DIRECTORIES ||
        monotonic_milliseconds() >= search_deadline) {
        return 0;
    }
    searched_directories++;

    DIR *search = opendir(directory);
    if (search == NULL) {
        return 0;
    }

    int found = 0;
    struct dirent *entry;
    while ((entry = readdir(search)) != NULL) {
        if (strcmp(entry->d_name, ".") == 0 ||
            strcmp(entry->d_name, "..") == 0 ||
            skip_search_directory(entry->d_name)) {
            continue;
        }

        char candidate[1200];
        int candidate_len = snprintf(
            candidate,
            sizeof(candidate),
            "%s/%s",
            directory,
            entry->d_name
        );
        if (candidate_len < 0 || (size_t)candidate_len >= sizeof(candidate)) {
            continue;
        }

        struct stat details;
        if (lstat(candidate, &details) != 0) {
            continue;
        }

        if (S_ISDIR(details.st_mode)) {
            if (depth < LAUNCHER_SEARCH_MAX_DEPTH && !S_ISLNK(details.st_mode) &&
                search_runtime_directory(
                    candidate,
                    depth + 1,
                    runtime_dir,
                    runtime_dir_size,
                    repo_root,
                    repo_root_size
                )) {
                found = 1;
                break;
            }
            continue;
        }

        if (strcmp(entry->d_name, "celune-bin") != 0 ||
            access(candidate, X_OK) != 0 ||
            !find_repo_root(directory, repo_root, repo_root_size) ||
            !copy_text(runtime_dir, runtime_dir_size, directory)) {
            continue;
        }

        found = 1;
        break;
    }

    closedir(search);
    return found;
}

static int set_runtime_target(
    const char *runtime_dir,
    char *target,
    size_t target_size
) {
    int target_len = snprintf(
        target,
        target_size,
        "%s/celune-bin",
        runtime_dir
    );
    return target_len >= 0 && (size_t)target_len < target_size;
}

static int resolve_runtime_location(
    const char *base,
    char *target,
    size_t target_size,
    char *repo_root,
    size_t repo_root_size
) {
    char runtime_dir[1024];
    char candidate[1200];
    int candidate_len = snprintf(
        candidate,
        sizeof(candidate),
        "%s/celune-bin",
        base
    );
    if (candidate_len >= 0 && (size_t)candidate_len < sizeof(candidate) &&
        access(candidate, X_OK) == 0 &&
        find_repo_root(base, repo_root, repo_root_size) &&
        copy_text(runtime_dir, sizeof(runtime_dir), base)) {
        return set_runtime_target(runtime_dir, target, target_size);
    }

    if (find_repo_root(base, repo_root, repo_root_size)) {
        return 0;
    }

    searched_directories = 0;
    search_deadline = monotonic_milliseconds() + LAUNCHER_SEARCH_MAX_MILLISECONDS;

    char current_directory[1024];
    const char *current = getcwd(current_directory, sizeof(current_directory));
    const char *environment_roots[] = {
        current,
        getenv("HOME"),
        getenv("XDG_DATA_HOME"),
        "/opt",
        "/usr/local",
        "/usr",
        "/"
    };
    for (size_t index = 0; index < sizeof(environment_roots) / sizeof(environment_roots[0]); index++) {
        if (environment_roots[index] == NULL ||
            !search_runtime_directory(
                environment_roots[index],
                0,
                runtime_dir,
                sizeof(runtime_dir),
                repo_root,
                repo_root_size
            )) {
            continue;
        }

        return set_runtime_target(runtime_dir, target, target_size);
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

    int launcher_len = snprintf(launcher, sizeof(launcher), "%s/celune", base);
    if (launcher_len < 0 || (size_t)launcher_len >= sizeof(launcher)) {
        printfe("Celune cannot start in this location, the path is too long.\n");
        return 1;
    }

    target[0] = '\0';
    if (!resolve_runtime_location(
            base,
            target,
            sizeof(target),
            repo_root,
            sizeof(repo_root)
        ) && !find_repo_root(base, repo_root, sizeof(repo_root))) {
        printfe("Celune could not find her compiled runtime or repository.\n");
        printfe("Searched beside the launcher and the available filesystem roots.\n");
        return 1;
    }

    int python_len = snprintf(python, sizeof(python), "%s/.venv/bin/python", repo_root);
    int main_py_len = snprintf(main_py, sizeof(main_py), "%s/main.py", repo_root);
    int setup_py_len = snprintf(setup_py, sizeof(setup_py), "%s/setup.py", repo_root);

    if (python_len < 0 || (size_t)python_len >= sizeof(python) ||
        main_py_len < 0 || (size_t)main_py_len >= sizeof(main_py) ||
        setup_py_len < 0 || (size_t)setup_py_len >= sizeof(setup_py)) {
        printfe("Celune cannot start in this location, the path is too long.\n");
        return 1;
    }

    if (access(target, X_OK) == 0) {
        int launcher_pipe[2];
        if (pipe(launcher_pipe) != 0) {
            perror("pipe failed");
            return 1;
        }

        char launcher_pipe_fd[32];
        snprintf(launcher_pipe_fd, sizeof(launcher_pipe_fd), "%d", launcher_pipe[0]);
        if (setenv("CELUNE_LAUNCHER_PIPE_FD", launcher_pipe_fd, 1) != 0) {
            close(launcher_pipe[0]);
            close(launcher_pipe[1]);
            printfe("Celune could not configure the launcher connection pipe.\n");
            return 1;
        }

        pid_t pid = fork();
        if (pid == -1) {
            perror("fork failed");
            close(launcher_pipe[0]);
            close(launcher_pipe[1]);
            return 1;
        }

        if (pid == 0) {
            launcher_restore_child_terminal();
            close(launcher_pipe[1]);
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
            close(launcher_pipe[0]);
            int status;
            waitpid(pid, &status, 0);
            close(launcher_pipe[1]);

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

    printfe("Exit code: %d\n", return_code);
    if (return_code >= 128) {
        printfe("Terminated by signal: %d\n", return_code - 128);
    }
}
