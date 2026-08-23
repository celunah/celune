#include "launcher_platform.h"

#include <windows.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define printfe(...) do { fprintf(stderr, __VA_ARGS__); } while (0)
#define STATUS_CONTROL_C_EXIT_VALUE 0xC000013AUL
#define LAUNCHER_SEARCH_MAX_DEPTH 8
#define LAUNCHER_SEARCH_MAX_DIRECTORIES 10000
#define LAUNCHER_SEARCH_MAX_MILLISECONDS 5000

static int launcher_child_failed = 0;
static size_t searched_directories = 0;
static ULONGLONG search_deadline = 0;

static const char *windows_exit_detail(DWORD exit_code) {
    switch (exit_code) {
        case 0xC0000005UL:
            return "access violation";
        case 0xC0000006UL:
            return "in-page error";
        case 0xC000001DUL:
            return "illegal instruction";
        case 0xC0000094UL:
            return "integer divide by zero";
        case 0xC0000135UL:
            return "required DLL was not found";
        case 0xC0000139UL:
            return "entry point was not found in a required DLL";
        case 0xC0000374UL:
            return "heap corruption";
        case 0xC0000409UL:
            return "stack buffer overrun";
        default:
            return NULL;
    }
}

static HANDLE create_launcher_pipe(char *name, size_t size) {
    int written = snprintf(
        name,
        size,
        "\\\\.\\pipe\\celune-launcher-%lu",
        (unsigned long)GetCurrentProcessId()
    );
    if (written < 0 || (size_t)written >= size) {
        return INVALID_HANDLE_VALUE;
    }

    return CreateNamedPipeA(
        name,
        PIPE_ACCESS_OUTBOUND | FILE_FLAG_OVERLAPPED,
        PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
        1,
        1,
        1,
        0,
        NULL
    );
}

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

static int read_environment_path(const char *name, char *out, size_t size) {
    char *value = NULL;
    size_t value_size = 0;
    if (_dupenv_s(&value, &value_size, name) != 0 || value == NULL) {
        return 0;
    }

    int copied = copy_text(out, size, value);
    free(value);
    return copied;
}

static int skip_search_directory(const char *name) {
    return _stricmp(name, ".git") == 0 ||
           _stricmp(name, ".venv") == 0 ||
           _stricmp(name, "$Recycle.Bin") == 0 ||
           _stricmp(name, "System Volume Information") == 0 ||
           _stricmp(name, "Windows") == 0 ||
           _stricmp(name, "WindowsApps") == 0 ||
           _stricmp(name, "AppData") == 0 ||
           _stricmp(name, "node_modules") == 0;
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
        GetTickCount64() >= search_deadline) {
        return 0;
    }
    searched_directories++;

    char pattern[1200];
    int pattern_len = snprintf(pattern, sizeof(pattern), "%s\\*", directory);
    if (pattern_len < 0 || (size_t)pattern_len >= sizeof(pattern)) {
        return 0;
    }

    WIN32_FIND_DATAA entry;
    HANDLE search = FindFirstFileA(pattern, &entry);
    if (search == INVALID_HANDLE_VALUE) {
        return 0;
    }

    int found = 0;
    do {
        if (strcmp(entry.cFileName, ".") == 0 ||
            strcmp(entry.cFileName, "..") == 0) {
            continue;
        }

        char candidate[1200];
        int candidate_len = snprintf(
            candidate,
            sizeof(candidate),
            "%s\\%s",
            directory,
            entry.cFileName
        );
        if (candidate_len < 0 || (size_t)candidate_len >= sizeof(candidate)) {
            continue;
        }

        if ((entry.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0) {
            if (depth >= LAUNCHER_SEARCH_MAX_DEPTH ||
                (entry.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0 ||
                skip_search_directory(entry.cFileName)) {
                continue;
            }

            if (search_runtime_directory(
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

        if (_stricmp(entry.cFileName, "celune-bin.exe") != 0 ||
            !find_repo_root(directory, repo_root, repo_root_size) ||
            !copy_text(runtime_dir, runtime_dir_size, directory)) {
            continue;
        }

        found = 1;
        break;
    } while (FindNextFileA(search, &entry));

    FindClose(search);
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
        "%s\\celune-bin.exe",
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
        "%s\\celune-bin.exe",
        base
    );
    if (candidate_len >= 0 && (size_t)candidate_len < sizeof(candidate) &&
        file_exists(candidate) &&
        find_repo_root(base, repo_root, repo_root_size) &&
        copy_text(runtime_dir, sizeof(runtime_dir), base)) {
        return copy_text(target, target_size, candidate);
    }

    searched_directories = 0;
    search_deadline = GetTickCount64() + LAUNCHER_SEARCH_MAX_MILLISECONDS;

    char current_directory[1024];
    DWORD current_length = GetCurrentDirectoryA(
        (DWORD)sizeof(current_directory),
        current_directory
    );
    char environment_root_storage[4][1024];
    const char *environment_roots[4];
    size_t environment_root_count = 0;
    const char *environment_names[] = {
        "USERPROFILE",
        "LOCALAPPDATA",
        "PROGRAMFILES",
        "PROGRAMFILES(X86)"
    };
    for (size_t index = 0; index < sizeof(environment_names) / sizeof(environment_names[0]); index++) {
        if (read_environment_path(
                environment_names[index],
                environment_root_storage[environment_root_count],
                sizeof(environment_root_storage[environment_root_count])
            )) {
            environment_roots[environment_root_count] =
                environment_root_storage[environment_root_count];
            environment_root_count++;
        }
    }

    if (current_length > 0 && current_length < sizeof(current_directory) &&
        search_runtime_directory(
            current_directory,
            0,
            runtime_dir,
            sizeof(runtime_dir),
            repo_root,
            repo_root_size
        )) {
        return set_runtime_target(runtime_dir, target, target_size);
    }

    for (size_t index = 0; index < environment_root_count; index++) {
        if (!search_runtime_directory(
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

    DWORD drives = GetLogicalDrives();
    for (char drive = 'A'; drive <= 'Z'; drive++) {
        DWORD mask = 1UL << (drive - 'A');
        if ((drives & mask) == 0) {
            continue;
        }

        char drive_root[] = "A:\\";
        drive_root[0] = drive;
        if (GetDriveTypeA(drive_root) != DRIVE_FIXED ||
            !search_runtime_directory(
                drive_root,
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

        if (*ch == '\"') {
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

static int spawn_update_helper_windows(
    const char *python,
    const char *main_py,
    const char *launcher_path,
    const char *repo_root,
    int argc,
    char **argv
) {
    STARTUPINFOA si = {0};
    PROCESS_INFORMATION pi = {0};
    char cmd[5200];
    char pid_text[32];
    size_t offset = 0;

    si.cb = sizeof(si);
    cmd[0] = '\0';
    snprintf(pid_text, sizeof(pid_text), "%lu", (unsigned long)GetCurrentProcessId());

    if (!append_windows_arg(cmd, sizeof(cmd), &offset, python) ||
        !append_text(cmd, sizeof(cmd), &offset, " ") ||
        !append_windows_arg(cmd, sizeof(cmd), &offset, main_py) ||
        !append_text(cmd, sizeof(cmd), &offset, " ") ||
        !append_windows_arg(cmd, sizeof(cmd), &offset, "__apply_update") ||
        !append_text(cmd, sizeof(cmd), &offset, " ") ||
        !append_windows_arg(cmd, sizeof(cmd), &offset, pid_text) ||
        !append_text(cmd, sizeof(cmd), &offset, " ") ||
        !append_windows_arg(cmd, sizeof(cmd), &offset, launcher_path)) {
        return 0;
    }

    for (int i = 1; i < argc; i++) {
        if (!append_text(cmd, sizeof(cmd), &offset, " ") ||
            !append_windows_arg(cmd, sizeof(cmd), &offset, argv[i])) {
            return 0;
        }
    }

    if (!CreateProcessA(
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
        )) {
        return 0;
    }

    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return 1;
}

static int get_exe_dir(char *out, size_t size) {
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

int launcher_run(int argc, char **argv) {
    char base[1024];
    char launcher[1024];
    char target[1024];
    char repo_root[1024];
    char pyvenv_cfg[1200];
    char python_home[1024];
    char python_dlls[1200];
    char python_lib[1200];
    char venv_root[1200];
    char venv_python[1400];
    char main_py[1400];
    char site_packages[1400];
    char setuptools_vendor[1600];
    char nuitka_pythonpath[5200];

    char launcher_pid[32];
    snprintf(launcher_pid, sizeof(launcher_pid), "%lu", (unsigned long)GetCurrentProcessId());

    if (!SetEnvironmentVariableA("CELUNE_LAUNCHER", "1") ||
        !SetEnvironmentVariableA("CELUNE_LAUNCHER_PID", launcher_pid)) {
        printfe("Celune could not configure launcher environment variables.\n");
        return 1;
    }

    if (!get_exe_dir(base, sizeof(base))) {
        printfe("Celune could not determine the launcher location.\n");
        return 1;
    }

    int launcher_len = snprintf(launcher, sizeof(launcher), "%s\\celune.exe", base);
    if (launcher_len < 0 || (size_t)launcher_len >= sizeof(launcher)) {
        printfe("Celune cannot start in this location, the path is too long.\n");
        return 1;
    }

    if (!resolve_runtime_location(
            base,
            target,
            sizeof(target),
            repo_root,
            sizeof(repo_root)
        )) {
        printfe("Celune could not find her compiled runtime or repository.\n");
        printfe("Searched beside the launcher and the available filesystem roots.\n");
        return 1;
    }

    int pyvenv_cfg_len = snprintf(pyvenv_cfg, sizeof(pyvenv_cfg), "%s\\.venv\\pyvenv.cfg", repo_root);
    int venv_root_len = snprintf(venv_root, sizeof(venv_root), "%s\\.venv", repo_root);
    int venv_python_len = snprintf(venv_python, sizeof(venv_python), "%s\\Scripts\\python.exe", venv_root);
    int main_py_len = snprintf(main_py, sizeof(main_py), "%s\\main.py", repo_root);
    int site_packages_len = snprintf(site_packages, sizeof(site_packages), "%s\\Lib\\site-packages", venv_root);
    int setuptools_vendor_len = snprintf(setuptools_vendor, sizeof(setuptools_vendor), "%s\\setuptools\\_vendor", site_packages);
    if (pyvenv_cfg_len < 0 || (size_t)pyvenv_cfg_len >= sizeof(pyvenv_cfg) ||
        venv_root_len < 0 || (size_t)venv_root_len >= sizeof(venv_root) ||
        venv_python_len < 0 || (size_t)venv_python_len >= sizeof(venv_python) ||
        main_py_len < 0 || (size_t)main_py_len >= sizeof(main_py) ||
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
        printfe("Celune cannot set up her Python path, the path is too long.\n");
        return 1;
    }

    DWORD path_capacity = GetEnvironmentVariableA("PATH", NULL, 0);
    char *updated_path = NULL;
    if (path_capacity == 0) {
        updated_path = (char *)malloc(1);
        if (updated_path == NULL) {
            printfe("Celune could not allocate memory for %%PATH%%.\n");
            return 1;
        }
        updated_path[0] = '\0';
    } else {
        updated_path = (char *)malloc(path_capacity);
        if (updated_path == NULL) {
            printfe("Celune could not allocate memory for %%PATH%%.\n");
            return 1;
        }

        DWORD path_len = GetEnvironmentVariableA(
            "PATH",
            updated_path,
            path_capacity
        );
        if (path_len >= path_capacity) {
            free(updated_path);
            printfe("Celune could not read %%PATH%%.\n");
            return 1;
        }
    }

    size_t path_value_size = strlen(python_home) + strlen(updated_path) + 2;
    char *path_value = (char *)malloc(path_value_size);
    if (path_value == NULL) {
        free(updated_path);
        printfe("Celune could not allocate memory for %%PATH%%.\n");
        return 1;
    }

    int updated_path_len = snprintf(path_value, path_value_size, "%s;%s", python_home, updated_path);
    if (updated_path_len < 0 || (size_t)updated_path_len >= path_value_size) {
        free(path_value);
        free(updated_path);
        printfe("Celune cannot set up %%PATH%%, the path is too long.\n");
        return 1;
    }

    if (!SetEnvironmentVariableA("PATH", path_value) ||
        !SetEnvironmentVariableA("PYTHONHOME", python_home) ||
        !SetEnvironmentVariableA("NUITKA_PYTHONPATH", nuitka_pythonpath)) {
        free(path_value);
        free(updated_path);
        printfe("Celune could not configure her Python runtime environment.\n");
        return 1;
    }
    free(path_value);
    free(updated_path);

    STARTUPINFOA si = {0};
    PROCESS_INFORMATION pi = {0};
    OVERLAPPED pipe_connect = {0};
    HANDLE launcher_pipe = INVALID_HANDLE_VALUE;
    HANDLE pipe_connect_event = NULL;
    char launcher_pipe_name[256];
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

    launcher_pipe = create_launcher_pipe(launcher_pipe_name, sizeof(launcher_pipe_name));
    if (launcher_pipe == INVALID_HANDLE_VALUE) {
        printfe("Celune could not create the launcher connection pipe.\n");
        return 1;
    }

    pipe_connect_event = CreateEventA(NULL, TRUE, FALSE, NULL);
    if (pipe_connect_event == NULL) {
        CloseHandle(launcher_pipe);
        printfe("Celune could not create the launcher connection event.\n");
        return 1;
    }
    pipe_connect.hEvent = pipe_connect_event;

    BOOL connect_pending = ConnectNamedPipe(launcher_pipe, &pipe_connect);
    DWORD connect_error = connect_pending ? ERROR_SUCCESS : GetLastError();
    if (connect_pending || connect_error == ERROR_PIPE_CONNECTED) {
        SetEvent(pipe_connect_event);
    }
    else if (connect_error != ERROR_IO_PENDING) {
        CloseHandle(pipe_connect_event);
        CloseHandle(launcher_pipe);
        printfe("Celune could not prepare the launcher connection pipe.\n");
        return 1;
    }

    if (!SetEnvironmentVariableA("CELUNE_LAUNCHER_PIPE", launcher_pipe_name)) {
        CancelIoEx(launcher_pipe, &pipe_connect);
        CloseHandle(pipe_connect_event);
        CloseHandle(launcher_pipe);
        printfe("Celune could not configure the launcher connection pipe.\n");
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
        repo_root,
        &si,
        &pi
    );

    if (!ok) {
        CancelIoEx(launcher_pipe, &pipe_connect);
        CloseHandle(pipe_connect_event);
        CloseHandle(launcher_pipe);
        printfe("Celune could not launch her compiled runtime.\nExit code: %lu\n", GetLastError());
        return 1;
    }

    HANDLE wait_handles[2] = {pi.hProcess, pipe_connect_event};
    DWORD first_wait = WaitForMultipleObjects(2, wait_handles, FALSE, INFINITE);
    if (first_wait == WAIT_OBJECT_0) {
        CancelIoEx(launcher_pipe, &pipe_connect);
    }
    CloseHandle(pipe_connect_event);
    WaitForSingleObject(pi.hProcess, INFINITE);

    DWORD exit_code = 1;
    GetExitCodeProcess(pi.hProcess, &exit_code);

    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    CloseHandle(launcher_pipe);

    if ((int)exit_code == CELUNE_EXIT_PENDING_UPDATE) {
        printfe("%s\n", launcher_exit_reason(CELUNE_EXIT_PENDING_UPDATE));
        if (!file_exists(venv_python) || !file_exists(main_py)) {
            printfe("Celune could not find the Python helper needed to apply updates.\n");
            return 1;
        }
        if (!spawn_update_helper_windows(venv_python, main_py, launcher, repo_root, argc, argv)) {
            printfe("Celune could not start her update helper.\n");
            return 1;
        }
        return 0;
    }

    launcher_child_failed = exit_code != 0;
    return (int)exit_code;
}

void launcher_report_failure(int return_code) {
    if (launcher_startup_was_interrupted() ||
        return_code == 130 ||
        (DWORD)return_code == STATUS_CONTROL_C_EXIT_VALUE) {
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

    DWORD exit_code = (DWORD)return_code;
    printfe(
        "Exit code: 0x%08lX (%lu)\n",
        (unsigned long)exit_code,
        (unsigned long)exit_code
    );

    const char *detail = windows_exit_detail(exit_code);
    if (detail != NULL) {
        printfe("Windows exception: %s.\n", detail);
    }
}
