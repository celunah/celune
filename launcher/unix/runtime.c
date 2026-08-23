#define _POSIX_C_SOURCE 200809L

#include "launcher_platform.h"

#include <unistd.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <dirent.h>
#include <signal.h>

#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#define printfe(...) do { fprintf(stderr, __VA_ARGS__); } while (0)
#define LAUNCHER_SEARCH_MAX_DEPTH 16
#define LAUNCHER_SEARCH_MAX_DIRECTORIES 500000
#define LAUNCHER_SEARCH_MAX_MILLISECONDS 60000
#define LAUNCHER_SEARCH_MAX_READ_MILLISECONDS 5000

enum search_limit_reason {
    SEARCH_LIMIT_NONE,
    SEARCH_LIMIT_DEPTH,
    SEARCH_LIMIT_FOLDERS,
    SEARCH_LIMIT_TIME,
    SEARCH_ROOT_NOT_CELUNE
};

static int launcher_child_failed = 0;
static size_t searched_directories = 0;
static unsigned long long search_deadline = 0;
static unsigned long long search_started = 0;
static unsigned long long next_status_update = 0;
static int search_status_started = 0;
static int search_status_ansi = 0;
static int search_status_level = 0;
static enum search_limit_reason search_limit = SEARCH_LIMIT_NONE;

static int file_exists(const char *path) {
    return access(path, F_OK) == 0;
}

static int lookup_file_exists(const char *path) {
    struct stat details;
    return lstat(path, &details) == 0 && S_ISREG(details.st_mode) &&
           access(path, X_OK) == 0;
}

static int copy_text(char *dest, size_t size, const char *src) {
    size_t len = strlen(src);

    if (len >= size) {
        return 0;
    }

    memcpy(dest, src, len + 1);
    return 1;
}

static int trim_line(char *line) {
    size_t length = strlen(line);
    while (length > 0 &&
           (line[length - 1] == '\n' || line[length - 1] == '\r' ||
            line[length - 1] == ' ' || line[length - 1] == '\t')) {
        line[--length] = '\0';
    }
    return 1;
}

static unsigned long long monotonic_milliseconds(void);

static int valid_celune_root_text(const char *text) {
    const char *cursor = text;
    if (*cursor++ != 'v' || !isdigit((unsigned char)*cursor)) {
        return 0;
    }

    while (isdigit((unsigned char)*cursor)) {
        cursor++;
    }
    while (*cursor == '.') {
        cursor++;
        if (!isdigit((unsigned char)*cursor)) {
            return 0;
        }
        while (isdigit((unsigned char)*cursor)) {
            cursor++;
        }
    }

    if (cursor[0] != ' ' || cursor[1] != '(') {
        return 0;
    }
    cursor += 2;
    const char *commit = cursor;
    while (isxdigit((unsigned char)*cursor)) {
        cursor++;
    }
    if ((size_t)(cursor - commit) < 7 || cursor[0] != ')' || cursor[1] != ',') {
        return 0;
    }
    cursor += 2;
    if (*cursor++ != ' ') {
        return 0;
    }

    for (int index = 0; index < 10; index++) {
        if (index == 2 || index == 5) {
            if (*cursor++ != '/') {
                return 0;
            }
        } else if (!isdigit((unsigned char)*cursor++)) {
            return 0;
        }
    }

    return *cursor == '\0';
}

static int read_celune_root(const char *directory) {
    char marker_path[1200];
    int marker_len = snprintf(
        marker_path,
        sizeof(marker_path),
        "%s/.celune-root",
        directory
    );
    if (marker_len < 0 || (size_t)marker_len >= sizeof(marker_path)) {
        return 0;
    }

    struct stat marker_details;
    if (lstat(marker_path, &marker_details) != 0 ||
        !S_ISREG(marker_details.st_mode)) {
        return 0;
    }

    FILE *marker = fopen(marker_path, "r");
    if (marker == NULL) {
        return 0;
    }

    char line[256];
    int readable = fgets(line, sizeof(line), marker) != NULL;
    fclose(marker);
    if (!readable) {
        return 0;
    }

    trim_line(line);
    return valid_celune_root_text(line);
}

static const char *search_limit_text(void) {
    switch (search_limit) {
        case SEARCH_LIMIT_DEPTH:
            return "Depth limit exceeded.";
        case SEARCH_LIMIT_FOLDERS:
            return "Folder limit exceeded.";
        case SEARCH_LIMIT_TIME:
            return "Time limit exceeded.";
        case SEARCH_ROOT_NOT_CELUNE:
            return "Current Celune root is incomplete.";
        default:
            return NULL;
    }
}

static void show_search_status(const char *path, int level) {
    unsigned long long now = monotonic_milliseconds();
    if (!search_status_started) {
        search_status_level = 0;
    }
    if (level < 1) {
        level = 1;
    } else if (level > LAUNCHER_SEARCH_MAX_DEPTH) {
        level = LAUNCHER_SEARCH_MAX_DEPTH;
    }
    if (search_status_started && now < next_status_update &&
        level == search_status_level) {
        return;
    }
    next_status_update = now + 100;
    search_status_level = level;

    if (!search_status_started) {
        search_status_started = 1;
        search_status_ansi = isatty(STDERR_FILENO);
        printfe("Looking for Celune...\n");
    } else if (search_status_ansi) {
        printfe("\033[2A\033[2K\r");
    } else {
        printfe("\n");
    }

    printfe("File: %s\n", path);
    printfe(
        "Level: %d/%d | Folders: %zu/%d | Time: %.2fs",
        level,
        LAUNCHER_SEARCH_MAX_DEPTH,
        searched_directories,
        LAUNCHER_SEARCH_MAX_DIRECTORIES,
        (double)(now - search_started) / 1000.0
    );
    fflush(stderr);
}

static void clear_search_status(void) {
    if (!search_status_started) {
        return;
    }

    if (search_status_ansi) {
        printfe("\033[2A\033[2K\r\033[1B\r\033[2K\r\033[1B\r\033[2K\r\033[2A\r");
    } else {
        printfe("\n");
    }
    fflush(stderr);
    search_status_started = 0;
}

static void report_lookup_failure(void) {
    printfe("Could not find Celune.\n");
    const char *limit = search_limit_text();
    if (limit != NULL) {
        printfe("%s\n", limit);
    }
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
        if (read_celune_root(current)) {
            return copy_text(out, size, current);
        }

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

static int set_runtime_target(
    const char *runtime_dir,
    char *target,
    size_t target_size
);

struct search_queue_entry {
    char *path;
    int depth;
};

struct search_queue {
    struct search_queue_entry *entries;
    size_t count;
    size_t next;
    size_t capacity;
};

static int search_queue_add(
    struct search_queue *queue,
    const char *path,
    int depth
) {
    if (queue->count >= LAUNCHER_SEARCH_MAX_DIRECTORIES) {
        search_limit = SEARCH_LIMIT_FOLDERS;
        return 0;
    }

    if (queue->count == queue->capacity) {
        size_t capacity = queue->capacity == 0 ? 64 : queue->capacity * 2;
        struct search_queue_entry *entries = (struct search_queue_entry *)realloc(
            queue->entries,
            capacity * sizeof(*entries)
        );
        if (entries == NULL) {
            return 0;
        }
        queue->entries = entries;
        queue->capacity = capacity;
    }

    size_t length = strlen(path);
    char *copy = (char *)malloc(length + 1);
    if (copy == NULL) {
        return 0;
    }
    memcpy(copy, path, length + 1);
    queue->entries[queue->count].path = copy;
    queue->entries[queue->count].depth = depth;
    queue->count++;
    return 1;
}

static void search_queue_clear(struct search_queue *queue) {
    for (size_t index = queue->next; index < queue->count; index++) {
        free(queue->entries[index].path);
    }
    free(queue->entries);
    queue->entries = NULL;
    queue->count = 0;
    queue->next = 0;
    queue->capacity = 0;
}

static int search_directory_contents(
    const char *directory,
    int depth,
    char *runtime_dir,
    size_t runtime_dir_size,
    char *repo_root,
    size_t repo_root_size,
    struct search_queue *queue
) {
    unsigned long long now = monotonic_milliseconds();
    if (now >= search_deadline) {
        search_limit = SEARCH_LIMIT_TIME;
        return 0;
    }
    if (searched_directories >= LAUNCHER_SEARCH_MAX_DIRECTORIES) {
        search_limit = SEARCH_LIMIT_FOLDERS;
        return 0;
    }
    searched_directories++;
    show_search_status(directory, depth + 1);

    if (read_celune_root(directory)) {
        char marker_target[1200];
        if (set_runtime_target(directory, marker_target, sizeof(marker_target)) &&
            lookup_file_exists(marker_target)) {
            return copy_text(runtime_dir, runtime_dir_size, directory) &&
                   copy_text(repo_root, repo_root_size, directory);
        }
        search_limit = SEARCH_ROOT_NOT_CELUNE;
        return 0;
    }

    DIR *search = opendir(directory);
    if (search == NULL) {
        return 0;
    }

    unsigned long long read_deadline =
        monotonic_milliseconds() + LAUNCHER_SEARCH_MAX_READ_MILLISECONDS;
    int found = 0;
    struct dirent *entry;
    while (1) {
        unsigned long long read_now = monotonic_milliseconds();
        if (read_now >= search_deadline || read_now >= read_deadline) {
            search_limit = SEARCH_LIMIT_TIME;
            break;
        }
        entry = readdir(search);
        if (entry == NULL) {
            if (monotonic_milliseconds() >= read_deadline &&
                search_limit == SEARCH_LIMIT_NONE) {
                search_limit = SEARCH_LIMIT_TIME;
            }
            break;
        }
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
        if (S_ISLNK(details.st_mode)) {
            continue;
        }

        if (S_ISDIR(details.st_mode)) {
            if (depth >= LAUNCHER_SEARCH_MAX_DEPTH) {
                search_limit = SEARCH_LIMIT_DEPTH;
                continue;
            }
            search_queue_add(queue, candidate, depth + 1);
            continue;
        }

        if (strcmp(entry->d_name, "celune-bin") != 0 ||
            !lookup_file_exists(candidate) ||
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

static int search_runtime_directory(
    const char *directory,
    int depth,
    char *runtime_dir,
    size_t runtime_dir_size,
    char *repo_root,
    size_t repo_root_size
) {
    struct search_queue queue = {0};
    if (!search_queue_add(&queue, directory, depth)) {
        search_queue_clear(&queue);
        return 0;
    }

    int found = 0;
    while (queue.next < queue.count) {
        struct search_queue_entry entry = queue.entries[queue.next++];
        found = search_directory_contents(
            entry.path,
            entry.depth,
            runtime_dir,
            runtime_dir_size,
            repo_root,
            repo_root_size,
            &queue
        );
        free(entry.path);
        if (found || search_limit == SEARCH_LIMIT_TIME ||
            search_limit == SEARCH_LIMIT_FOLDERS) {
            break;
        }
    }

    search_queue_clear(&queue);
    return found;
}

static int set_runtime_target(
    const char *runtime_dir,
    char *target,
    size_t target_size
) {
    char candidate[1200];
    int candidate_len = snprintf(
        candidate,
        sizeof(candidate),
        "%s/celune-bin",
        runtime_dir
    );
    if (candidate_len >= 0 && (size_t)candidate_len < sizeof(candidate) &&
        lookup_file_exists(candidate)) {
        return copy_text(target, target_size, candidate);
    }

    candidate_len = snprintf(
        candidate,
        sizeof(candidate),
        "%s/bin/celune-bin",
        runtime_dir
    );
    if (candidate_len >= 0 && (size_t)candidate_len < sizeof(candidate) &&
        lookup_file_exists(candidate)) {
        return copy_text(target, target_size, candidate);
    }

    int target_len = snprintf(
        target,
        target_size,
        "%s/bin/celune-bin",
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
        lookup_file_exists(candidate) &&
        find_repo_root(base, repo_root, repo_root_size) &&
        copy_text(runtime_dir, sizeof(runtime_dir), base)) {
        return set_runtime_target(runtime_dir, target, target_size);
    }

    if (find_repo_root(base, repo_root, repo_root_size)) {
        return 0;
    }

    char current_directory[1024];
    const char *current = getcwd(current_directory, sizeof(current_directory));
    if (current != NULL &&
        find_repo_root(current, repo_root, repo_root_size) &&
        set_runtime_target(current, target, target_size) &&
        lookup_file_exists(target)) {
        return 1;
    }

    searched_directories = 0;
    search_deadline = monotonic_milliseconds() + LAUNCHER_SEARCH_MAX_MILLISECONDS;
    search_started = monotonic_milliseconds();
    next_status_update = search_started;
    search_status_started = 0;
    search_status_level = 0;
    search_limit = SEARCH_LIMIT_NONE;

    const char *environment_roots[] = {
        getenv("HOME"),
        current,
        getenv("XDG_DATA_HOME"),
        "/opt",
        "/usr/local",
        "/usr",
        "/"
    };
    for (size_t index = 0; index < sizeof(environment_roots) / sizeof(environment_roots[0]); index++) {
        if (environment_roots[index] == NULL) {
            continue;
        }
        int found = search_runtime_directory(
                environment_roots[index],
                0,
                runtime_dir,
                sizeof(runtime_dir),
                repo_root,
                repo_root_size
        );
        if (search_limit == SEARCH_ROOT_NOT_CELUNE) {
            clear_search_status();
            return 0;
        }
        if (!found) {
            continue;
        }

        clear_search_status();
        return set_runtime_target(runtime_dir, target, target_size);
    }

    clear_search_status();
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
    char configure_py[1024];

    char launcher_pid[32];
    snprintf(launcher_pid, sizeof(launcher_pid), "%ld", (long)getpid());

    if (setenv("CELUNE_LAUNCHER", "1", 1) != 0 ||
        setenv("CELUNE_LAUNCHER_PID", launcher_pid, 1) != 0) {
        printfe("Celune could not configure launcher environment variables.\n");
        return 1;
    }

    if (!get_exe_dir(base, sizeof(base))) {
        report_lookup_failure();
        return 1;
    }

    int launcher_len = snprintf(launcher, sizeof(launcher), "%s/celune", base);
    if (launcher_len < 0 || (size_t)launcher_len >= sizeof(launcher)) {
        report_lookup_failure();
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
        report_lookup_failure();
        return 1;
    }

    int python_len = snprintf(python, sizeof(python), "%s/.venv/bin/python", repo_root);
    int main_py_len = snprintf(main_py, sizeof(main_py), "%s/main.py", repo_root);
    int configure_py_len = snprintf(configure_py, sizeof(configure_py), "%s/configure.py", repo_root);

    if (python_len < 0 || (size_t)python_len >= sizeof(python) ||
        main_py_len < 0 || (size_t)main_py_len >= sizeof(main_py) ||
        configure_py_len < 0 || (size_t)configure_py_len >= sizeof(configure_py)) {
        report_lookup_failure();
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

        if (access(configure_py, R_OK) != 0) {
            printfe("Celune: Python environment is missing and configure.py is unavailable.\n");
            return 1;
        }

        printfe("Celune: Python environment missing; running configure.py.\n");

        for (size_t i = 0; i < sizeof(system_python) / sizeof(system_python[0]); i++) {
            pid_t setup_pid = fork();
            if (setup_pid == -1) {
                perror("fork failed");
                return 1;
            }

            if (setup_pid == 0) {
                launcher_restore_child_terminal();
                char *args[] = {(char *)system_python[i], configure_py, NULL};
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
            printfe("Celune: no system Python interpreter was found for configure.py.\n");
            return 1;
        }

        if (!WIFEXITED(setup_status) || WEXITSTATUS(setup_status) != 0) {
            printfe("Celune: configure.py failed.\n");
            return WIFEXITED(setup_status) ? WEXITSTATUS(setup_status) : 1;
        }

        if (access(python, X_OK) != 0) {
            printfe("Celune: configure.py completed but the Python environment is still unavailable.\n");
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
