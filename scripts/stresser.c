#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <workers> <seconds>\n", argv[0]);
        return 2;
    }

    char *end;
    long workers = strtol(argv[1], &end, 10);
    if (*end != '\0' || workers < 1 || workers > omp_get_num_procs()) {
        fprintf(stderr, "workers must be between 1 and %d\n", omp_get_num_procs());
        return 2;
    }

    double seconds = strtod(argv[2], &end);
    if (*end != '\0' || seconds <= 0) {
        fprintf(stderr, "seconds must be greater than zero\n");
        return 2;
    }

    double deadline = omp_get_wtime() + seconds;

    #pragma omp parallel num_threads(workers)
    {
        volatile uint64_t work = 0;

        while (omp_get_wtime() < deadline) {
            for (int i = 0; i < 8192; ++i) {
                work = work * 1664525u + 1013904223u;
            }
        }
    }

    return 0;
}
