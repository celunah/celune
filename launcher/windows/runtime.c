#include "launcher_platform.h"

#include <windows.h>

#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define printfe(...) do { fprintf(stderr, __VA_ARGS__); } while (0)
#define STATUS_CONTROL_C_EXIT_VALUE 0xC000013AUL
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
static ULONGLONG search_deadline = 0;
static ULONGLONG search_started = 0;
static ULONGLONG next_status_update = 0;
static int search_status_started = 0;
static int search_status_console = 0;
static int search_status_level = 0;
static COORD search_status_origin;
static enum search_limit_reason search_limit = SEARCH_LIMIT_NONE;

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

static int lookup_file_exists(const char *path) {
    DWORD attr = GetFileAttributesA(path);
    return attr != INVALID_FILE_ATTRIBUTES &&
           !(attr & FILE_ATTRIBUTE_DIRECTORY) &&
           !(attr & FILE_ATTRIBUTE_REPARSE_POINT);
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

static int trim_line(char *line);

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
        "%s\\.celune-root",
        directory
    );
    if (marker_len < 0 || (size_t)marker_len >= sizeof(marker_path)) {
        return 0;
    }

    DWORD attributes = GetFileAttributesA(marker_path);
    if (attributes == INVALID_FILE_ATTRIBUTES ||
        (attributes & FILE_ATTRIBUTE_DIRECTORY) != 0 ||
        (attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0) {
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

static void report_lookup_failure(void) {
    printfe("Could not find Celune.\n");
    const char *limit = search_limit_text();
    if (limit != NULL) {
        printfe("%s\n", limit);
    }
}

static int write_console_status_line(
    HANDLE output,
    SHORT row,
    const char *line,
    const CONSOLE_SCREEN_BUFFER_INFO *info
) {
    if (search_status_origin.X >= info->dwSize.X || row < 0 || row >= info->dwSize.Y) {
        return 0;
    }

    COORD position = {search_status_origin.X, row};
    DWORD width = (DWORD)(info->dwSize.X - search_status_origin.X);
    DWORD written;
    if (!FillConsoleOutputCharacterA(output, ' ', width, position, &written) ||
        !FillConsoleOutputAttribute(output, info->wAttributes, width, position, &written) ||
        !SetConsoleCursorPosition(output, position)) {
        return 0;
    }

    DWORD length = (DWORD)strlen(line);
    if (length >= width) {
        length = width - 1;
    }
    return WriteConsoleA(output, line, length, &written, NULL) != 0;
}

static int render_console_status(
    HANDLE output,
    const char *file_line,
    const char *level_line,
    const CONSOLE_SCREEN_BUFFER_INFO *info
) {
    if (search_status_origin.Y > info->dwSize.Y - 3) {
        return 0;
    }

    return write_console_status_line(
               output,
               search_status_origin.Y,
               "Looking for Celune...",
               info
           ) &&
           write_console_status_line(
               output,
               search_status_origin.Y + 1,
               file_line,
               info
           ) &&
           write_console_status_line(
               output,
               search_status_origin.Y + 2,
               level_line,
               info
           );
}

static void show_search_status(const char *path, int level) {
    ULONGLONG now = GetTickCount64();
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

    char file_line[1400];
    char level_line[256];
    snprintf(file_line, sizeof(file_line), "File: %s", path);
    snprintf(
        level_line,
        sizeof(level_line),
        "Level: %d/%d | Folders: %zu/%d | Time: %.2fs",
        level,
        LAUNCHER_SEARCH_MAX_DEPTH,
        searched_directories,
        LAUNCHER_SEARCH_MAX_DIRECTORIES,
        (double)(now - search_started) / 1000.0
    );

    if (!search_status_started) {
        search_status_started = 1;
        search_status_console = 0;
        HANDLE output = GetStdHandle(STD_ERROR_HANDLE);
        CONSOLE_SCREEN_BUFFER_INFO info;
        if (output != INVALID_HANDLE_VALUE && GetConsoleScreenBufferInfo(output, &info)) {
            search_status_origin = info.dwCursorPosition;
            if (render_console_status(output, file_line, level_line, &info)) {
                search_status_console = 1;
                return;
            }
        }

        printfe("Looking for Celune...\n");
        printfe("%s\n%s\n", file_line, level_line);
        fflush(stderr);
        return;
    }

    if (search_status_console) {
        HANDLE output = GetStdHandle(STD_ERROR_HANDLE);
        CONSOLE_SCREEN_BUFFER_INFO info;
        if (output != INVALID_HANDLE_VALUE && GetConsoleScreenBufferInfo(output, &info) &&
            render_console_status(output, file_line, level_line, &info)) {
            return;
        }
        search_status_console = 0;
    }

    printfe("%s\n%s\n", file_line, level_line);
    fflush(stderr);
}

static void clear_search_status(void) {
    if (!search_status_started) {
        return;
    }

    if (search_status_console) {
        HANDLE output = GetStdHandle(STD_ERROR_HANDLE);
        CONSOLE_SCREEN_BUFFER_INFO info;
        if (output != INVALID_HANDLE_VALUE && GetConsoleScreenBufferInfo(output, &info) &&
            search_status_origin.X < info.dwSize.X &&
            search_status_origin.Y <= info.dwSize.Y - 1) {
            DWORD width = (DWORD)(info.dwSize.X - search_status_origin.X);
            DWORD written;
            for (SHORT offset = 0; offset < 3; offset++) {
                SHORT row = search_status_origin.Y + offset;
                if (row >= info.dwSize.Y) {
                    break;
                }
                COORD position = {search_status_origin.X, row};
                FillConsoleOutputCharacterA(output, ' ', width, position, &written);
                FillConsoleOutputAttribute(output, info.wAttributes, width, position, &written);
            }
            SetConsoleCursorPosition(output, search_status_origin);
        }
    } else {
        printfe("\n");
    }
    fflush(stderr);
    search_status_started = 0;
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
        if (read_celune_root(current)) {
            return copy_text(out, size, current);
        }

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
    ULONGLONG now = GetTickCount64();
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

    ULONGLONG read_deadline = GetTickCount64() + LAUNCHER_SEARCH_MAX_READ_MILLISECONDS;
    int found = 0;
    do {
        ULONGLONG read_now = GetTickCount64();
        if (read_now >= search_deadline || read_now >= read_deadline) {
            search_limit = SEARCH_LIMIT_TIME;
            break;
        }
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

        if ((entry.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0) {
            continue;
        }

        if ((entry.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0) {
            if (depth >= LAUNCHER_SEARCH_MAX_DEPTH) {
                search_limit = SEARCH_LIMIT_DEPTH;
                continue;
            }
            if (skip_search_directory(entry.cFileName)) {
                continue;
            }

            search_queue_add(queue, candidate, depth + 1);
            continue;
        }

        if (_stricmp(entry.cFileName, "celune-bin.exe") != 0 ||
            !find_repo_root(directory, repo_root, repo_root_size) ||
            !copy_text(runtime_dir, runtime_dir_size, directory)) {
            continue;
        }

        found = 1;
        break;
    } while (FindNextFileA(search, &entry) &&
             GetTickCount64() < read_deadline &&
             GetTickCount64() < search_deadline);

    if (!found && search_limit == SEARCH_LIMIT_NONE) {
        ULONGLONG finished_at = GetTickCount64();
        if (finished_at >= search_deadline || finished_at >= read_deadline) {
            search_limit = SEARCH_LIMIT_TIME;
        }
    }

    FindClose(search);
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
        "%s\\celune-bin.exe",
        runtime_dir
    );
    if (candidate_len >= 0 && (size_t)candidate_len < sizeof(candidate) &&
        lookup_file_exists(candidate)) {
        return copy_text(target, target_size, candidate);
    }

    candidate_len = snprintf(
        candidate,
        sizeof(candidate),
        "%s\\bin\\celune-bin.exe",
        runtime_dir
    );
    if (candidate_len >= 0 && (size_t)candidate_len < sizeof(candidate) &&
        lookup_file_exists(candidate)) {
        return copy_text(target, target_size, candidate);
    }

    int target_len = snprintf(
        target,
        target_size,
        "%s\\bin\\celune-bin.exe",
        runtime_dir
    );
    return target_len >= 0 && (size_t)target_len < target_size;
}

static int resolve_forced_runtime_location(
    const char *root,
    char *target,
    size_t target_size,
    char *repo_root,
    size_t repo_root_size
) {
    if (!find_repo_root(root, repo_root, repo_root_size) ||
        !set_runtime_target(root, target, target_size) ||
        !lookup_file_exists(target)) {
        return 0;
    }

    return 1;
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
        lookup_file_exists(candidate) &&
        find_repo_root(base, repo_root, repo_root_size) &&
        copy_text(runtime_dir, sizeof(runtime_dir), base)) {
        return copy_text(target, target_size, candidate);
    }

    searched_directories = 0;
    search_deadline = GetTickCount64() + LAUNCHER_SEARCH_MAX_MILLISECONDS;
    search_started = GetTickCount64();
    next_status_update = search_started;
    search_status_started = 0;
    search_status_level = 0;
    search_limit = SEARCH_LIMIT_NONE;

    char current_directory[1024];
    DWORD current_length = GetCurrentDirectoryA(
        (DWORD)sizeof(current_directory),
        current_directory
    );

    if (current_length > 0 && current_length < sizeof(current_directory) &&
        find_repo_root(current_directory, repo_root, repo_root_size) &&
        set_runtime_target(current_directory, target, target_size) &&
        lookup_file_exists(target)) {
        return 1;
    }

    char environment_root_storage[3][1024];
    const char *environment_roots[3];
    size_t environment_root_count = 0;
    const char *environment_names[] = {
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

    char user_profile[1024];
    int found = 0;
    if (read_environment_path("USERPROFILE", user_profile, sizeof(user_profile))) {
        found = search_runtime_directory(
            user_profile,
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
    }
    if (found) {
        clear_search_status();
        return set_runtime_target(runtime_dir, target, target_size);
    }

    if (current_length > 0 && current_length < sizeof(current_directory)) {
        found = search_runtime_directory(
            current_directory,
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
    }
    if (found) {
        clear_search_status();
        return set_runtime_target(runtime_dir, target, target_size);
    }

    for (size_t index = 0; index < environment_root_count; index++) {
        found = search_runtime_directory(
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

    DWORD drives = GetLogicalDrives();
    for (char drive = 'A'; drive <= 'Z'; drive++) {
        DWORD mask = 1UL << (drive - 'A');
        if ((drives & mask) == 0) {
            continue;
        }

        char drive_root[] = "A:\\";
        drive_root[0] = drive;
        if (GetDriveTypeA(drive_root) != DRIVE_FIXED) {
            continue;
        }
        found = search_runtime_directory(
                drive_root,
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
    char root_override[1024];

    char launcher_pid[32];
    snprintf(launcher_pid, sizeof(launcher_pid), "%lu", (unsigned long)GetCurrentProcessId());

    if (!SetEnvironmentVariableA("CELUNE_LAUNCHER", "1") ||
        !SetEnvironmentVariableA("CELUNE_LAUNCHER_PID", launcher_pid)) {
        printfe("Celune could not configure launcher environment variables.\n");
        return 1;
    }

    int root_override_status = launcher_read_root_override(
        &argc,
        argv,
        root_override,
        sizeof(root_override)
    );
    if (root_override_status < 0) {
        printfe("Celune received an invalid --root or CELUNE_ROOT override.\n");
        return 1;
    }

    if (!get_exe_dir(base, sizeof(base))) {
        report_lookup_failure();
        return 1;
    }

    int launcher_len = snprintf(launcher, sizeof(launcher), "%s\\celune.exe", base);
    if (launcher_len < 0 || (size_t)launcher_len >= sizeof(launcher)) {
        report_lookup_failure();
        return 1;
    }

    if (root_override_status > 0) {
        if (!resolve_forced_runtime_location(
                root_override,
                target,
                sizeof(target),
                repo_root,
                sizeof(repo_root)
            )) {
            printfe("Celune could not use the requested root: %s\n", root_override);
            return 1;
        }
    } else if (!resolve_runtime_location(
                   base,
                   target,
                   sizeof(target),
                   repo_root,
                   sizeof(repo_root)
               )) {
        report_lookup_failure();
        return 1;
    }

    if (!lookup_file_exists(target)) {
        report_lookup_failure();
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
        report_lookup_failure();
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
        report_lookup_failure();
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
        report_lookup_failure();
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
        report_lookup_failure();
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
