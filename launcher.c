#define _POSIX_C_SOURCE 200809L

#ifdef __linux__
#include <unistd.h>
#include <sys/wait.h>
#include <termios.h>
#elif defined(_WIN32)
#include <windows.h>
#include <conio.h>
#include <direct.h>
#endif

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define printfe(...) do { fprintf(stderr, __VA_ARGS__); } while (0)

#ifdef __linux__
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

#ifdef _WIN32
static int file_exists(const char *path) {
    DWORD attr = GetFileAttributesA(path);
    return attr != INVALID_FILE_ATTRIBUTES && !(attr & FILE_ATTRIBUTE_DIRECTORY);
}

static int dir_exists(const char *path) {
    DWORD attr = GetFileAttributesA(path);
    return attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_DIRECTORY);
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

    char *last = strrchr(out, '\\');
    if (last == NULL) {
        return 0;
    }

    *last = '\0';
    return 1;
}

static int trim_line(char *line) {
    size_t len = strlen(line);
    while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r' || line[len - 1] == ' ' || line[len - 1] == '\t')) {
        line[len - 1] = '\0';
        len--;
    }

    return 1;
}

static int read_pyvenv_home(const char *cfg_path, char *out, size_t size) {
    FILE *cfg = fopen(cfg_path, "r");
    if (cfg == NULL) {
        return 0;
    }

    char line[2048];
    while (fgets(line, sizeof(line), cfg) != NULL) {
        trim_line(line);

        if (strncmp(line, "home =", 6) == 0) {
            const char *value = line + 6;
            while (*value == ' ' || *value == '\t') {
                value++;
            }

            fclose(cfg);
            return copy_text(out, size, value);
        }
    }

    fclose(cfg);
    return 0;
}

static int find_repo_root(const char *start_dir, char *out, size_t size) {
    char current[1024];
    if (!copy_text(current, sizeof(current), start_dir)) {
        return 0;
    }

    while (1) {
        char pyvenv_cfg[1200];
        int written = snprintf(pyvenv_cfg, sizeof(pyvenv_cfg), "%s\\.venv\\pyvenv.cfg", current);
        if (written > 0 && (size_t)written < sizeof(pyvenv_cfg) && file_exists(pyvenv_cfg)) {
            return copy_text(out, size, current);
        }

        char parent[1024];
        if (!parent_dir_of(current, parent, sizeof(parent))) {
            break;
        }

        if (strcmp(parent, current) == 0) {
            break;
        }

        if (strlen(parent) == 2 && parent[1] == ':') {
            break;
        }

        if (!copy_text(current, sizeof(current), parent)) {
            return 0;
        }
    }

    return 0;
}

static int append_text(char *dest, size_t size, size_t *offset, const char *text) {
    size_t len = strlen(text);

    if (*offset + len >= size) {
        return 0;
    }

    memcpy(dest + *offset, text, len);
    *offset += len;
    dest[*offset] = '\0';
    return 1;
}

static int append_windows_arg(char *dest, size_t size, size_t *offset, const char *arg) {
    if (!append_text(dest, size, offset, "\"")) {
        return 0;
    }

    size_t backslashes = 0;
    for (const char *ch = arg; *ch != '\0'; ch++) {
        if (*ch == '\\') {
            backslashes++;
            continue;
        }

        if (*ch == '"') {
            for (size_t i = 0; i < backslashes * 2 + 1; i++) {
                if (!append_text(dest, size, offset, "\\")) {
                    return 0;
                }
            }
            if (!append_text(dest, size, offset, "\"")) {
                return 0;
            }
            backslashes = 0;
            continue;
        }

        while (backslashes > 0) {
            if (!append_text(dest, size, offset, "\\")) {
                return 0;
            }
            backslashes--;
        }

        char next[2] = {*ch, '\0'};
        if (!append_text(dest, size, offset, next)) {
            return 0;
        }
    }

    while (backslashes > 0) {
        if (!append_text(dest, size, offset, "\\\\")) {
            return 0;
        }
        backslashes--;
    }

    return append_text(dest, size, offset, "\"");
}
#endif

#ifdef __linux__
int run_unix(int argc, char **argv) {
    char base[1024];
    char repo_root[1024];
    char target[1024];
    char python[1024];
    char main_py[1024];
    char setup_py[1024];

    setenv("CELUNE_LAUNCHER", "1", 1);

    if (!get_exe_dir(base, sizeof(base))) {
        printfe("Celune could not determine the launcher location.\n");
        return 1;
    }

    if (!find_repo_root(base, repo_root, sizeof(repo_root))) {
        printfe("Celune could not find the repository root with a Python virtual environment.\n");
        return 1;
    }

    int target_len = snprintf(target, sizeof(target), "%s/celune-bin", base);
    int python_len = snprintf(python, sizeof(python), "%s/.venv/bin/python", repo_root);
    int main_py_len = snprintf(main_py, sizeof(main_py), "%s/main.py", repo_root);
    int setup_py_len = snprintf(setup_py, sizeof(setup_py), "%s/setup.py", repo_root);

    if (target_len < 0 || (size_t)target_len >= sizeof(target) ||
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
int run_windows(int argc, char **argv) {
    char base[1024];
    char target[1024];
    char repo_root[1024];
    char pyvenv_cfg[1200];
    char python_home[1024];
    char python_dlls[1200];
    char python_lib[1200];
    char venv_root[1200];
    char site_packages[1400];
    char setuptools_vendor[1600];
    char nuitka_pythonpath[5200];
    char updated_path[5200];

    SetEnvironmentVariableA("CELUNE_LAUNCHER", "1");

    if (!get_exe_dir(base, sizeof(base))) {
        printfe("Celune could not determine the launcher location.\n");
        return 1;
    }

    int target_len = snprintf(target, sizeof(target), "%s\\celune-bin.exe", base);
    if (target_len < 0 || (size_t)target_len >= sizeof(target)) {
        printfe("Celune cannot start in this location, the path is too long.\n");
        return 1;
    }

    if (!file_exists(target)) {
        printfe("Celune could not find its compiled runtime binary.\n");
        printfe("Expected file: %s\n", target);
        return 1;
    }

    if (!find_repo_root(base, repo_root, sizeof(repo_root))) {
        printfe("Celune could not find the repository root with a Python virtual environment.\n");
        return 1;
    }

    int pyvenv_cfg_len = snprintf(pyvenv_cfg, sizeof(pyvenv_cfg), "%s\\.venv\\pyvenv.cfg", repo_root);
    int venv_root_len = snprintf(venv_root, sizeof(venv_root), "%s\\.venv", repo_root);
    int site_packages_len = snprintf(site_packages, sizeof(site_packages), "%s\\Lib\\site-packages", venv_root);
    int setuptools_vendor_len = snprintf(setuptools_vendor, sizeof(setuptools_vendor), "%s\\setuptools\\_vendor", site_packages);
    if (pyvenv_cfg_len < 0 || (size_t)pyvenv_cfg_len >= sizeof(pyvenv_cfg) ||
        venv_root_len < 0 || (size_t)venv_root_len >= sizeof(venv_root) ||
        site_packages_len < 0 || (size_t)site_packages_len >= sizeof(site_packages) ||
        setuptools_vendor_len < 0 || (size_t)setuptools_vendor_len >= sizeof(setuptools_vendor)) {
        printfe("Celune cannot start in this location, the path is too long.\n");
        return 1;
    }

    if (!file_exists(pyvenv_cfg) || !read_pyvenv_home(pyvenv_cfg, python_home, sizeof(python_home))) {
        printfe("Celune could not determine the base Python installation from .venv\\pyvenv.cfg.\n");
        return 1;
    }

    int python_dlls_len = snprintf(python_dlls, sizeof(python_dlls), "%s\\DLLs", python_home);
    int python_lib_len = snprintf(python_lib, sizeof(python_lib), "%s\\Lib", python_home);
    if (python_dlls_len < 0 || (size_t)python_dlls_len >= sizeof(python_dlls) ||
        python_lib_len < 0 || (size_t)python_lib_len >= sizeof(python_lib)) {
        printfe("Celune cannot start in this location, the path is too long.\n");
        return 1;
    }

    if (!dir_exists(python_home) || !dir_exists(python_lib) || !dir_exists(site_packages)) {
        printfe("Celune could not find the required Python runtime directories.\n");
        return 1;
    }

    int nuitka_pythonpath_len = snprintf(
        nuitka_pythonpath,
        sizeof(nuitka_pythonpath),
        "%s;%s;%s;%s;%s;%s;%s",
        repo_root,
        python_dlls,
        python_lib,
        python_home,
        venv_root,
        site_packages,
        setuptools_vendor
    );
    if (nuitka_pythonpath_len < 0 || (size_t)nuitka_pythonpath_len >= sizeof(nuitka_pythonpath)) {
        printfe("Celune cannot set up its Python path, the path is too long.\n");
        return 1;
    }

    DWORD path_len = GetEnvironmentVariableA("PATH", updated_path, (DWORD)sizeof(updated_path));
    if (path_len == 0 || path_len >= sizeof(updated_path)) {
        updated_path[0] = '\0';
    }

    char path_value[5200];
    int updated_path_len = snprintf(path_value, sizeof(path_value), "%s;%s", python_home, updated_path);
    if (updated_path_len < 0 || (size_t)updated_path_len >= sizeof(path_value)) {
        printfe("Celune cannot set up PATH, the path is too long.\n");
        return 1;
    }

    if (!SetEnvironmentVariableA("PATH", path_value) ||
        !SetEnvironmentVariableA("PYTHONHOME", python_home) ||
        !SetEnvironmentVariableA("NUITKA_PYTHONPATH", nuitka_pythonpath)) {
        printfe("Celune could not configure its Python runtime environment.\n");
        return 1;
    }

    STARTUPINFOA si = {0};
    PROCESS_INFORMATION pi = {0};
    si.cb = sizeof(si);

    si.dwFlags = STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_SHOW;

    char cmd[2200];
    size_t offset = 0;
    cmd[0] = '\0';

    if (!append_windows_arg(cmd, sizeof(cmd), &offset, target)) {
        printfe("Celune cannot start in this location, the command line is too long.\n");
        return 1;
    }

    for (int i = 1; i < argc; i++) {
        if (!append_text(cmd, sizeof(cmd), &offset, " ") ||
            !append_windows_arg(cmd, sizeof(cmd), &offset, argv[i])) {
            printfe("Celune cannot start in this location, the command line is too long.\n");
            return 1;
        }
    }

    BOOL ok = CreateProcessA(
        NULL,
        cmd,
        NULL,
        NULL,
        FALSE,
        0,
        NULL,
        repo_root,
        &si,
        &pi
    );

    if (!ok) {
        printfe("Celune could not launch its compiled runtime.\n%lu\n", GetLastError());
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

int main(int argc, char **argv) {
#ifdef __linux__
    int return_code = run_unix(argc, argv);

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
    int return_code = run_windows(argc, argv);

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
