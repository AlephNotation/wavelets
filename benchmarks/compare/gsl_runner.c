#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_version.h>
#include <gsl/gsl_wavelet.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SCHEMA_VERSION 1
#define MAX_BATCH_ITERATIONS 100000000ULL

typedef struct {
    const char *case_id;
    const char *direction;
    size_t length;
    size_t samples;
    double sample_time_ms;
    size_t warmup_batches;
} config;

typedef struct {
    gsl_wavelet *wavelet;
    gsl_wavelet_workspace *workspace;
    double *data;
    size_t length;
    int inverse;
} operation;

static void fail(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(EXIT_FAILURE);
}

static size_t parse_size(const char *text, const char *name) {
    char *end = NULL;
    errno = 0;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value > SIZE_MAX) {
        fprintf(stderr, "invalid %s: %s\n", name, text);
        exit(EXIT_FAILURE);
    }
    return (size_t)value;
}

static double parse_positive_double(const char *text, const char *name) {
    char *end = NULL;
    errno = 0;
    double value = strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !isfinite(value) || value <= 0.0) {
        fprintf(stderr, "invalid %s: %s\n", name, text);
        exit(EXIT_FAILURE);
    }
    return value;
}

static config parse_config(int argc, char **argv) {
    if (argc != 7) {
        fail("usage: gsl_runner CASE_ID DIRECTION LENGTH SAMPLES SAMPLE_MS WARMUP_BATCHES");
    }
    config parsed = {
        .case_id = argv[1],
        .direction = argv[2],
        .length = parse_size(argv[3], "length"),
        .samples = parse_size(argv[4], "samples"),
        .sample_time_ms = parse_positive_double(argv[5], "sample time"),
        .warmup_batches = parse_size(argv[6], "warmup batches"),
    };
    if (strcmp(parsed.direction, "forward") != 0 && strcmp(parsed.direction, "inverse") != 0) {
        fail("direction must be forward or inverse");
    }
    if (*parsed.case_id == '\0') {
        fail("case id cannot be empty");
    }
    for (const char *character = parsed.case_id; *character != '\0'; ++character) {
        if (!(('a' <= *character && *character <= 'z') ||
              ('A' <= *character && *character <= 'Z') ||
              ('0' <= *character && *character <= '9') ||
              *character == '_' || *character == '-' || *character == '/')) {
            fail("case id contains a character that is unsafe in JSON output");
        }
    }
    if (parsed.length == 0 || (parsed.length & (parsed.length - 1)) != 0 ||
        parsed.length > SIZE_MAX / sizeof(double)) {
        fail("length must be a positive power of two");
    }
    if (parsed.samples < 3) {
        fail("at least three samples are required");
    }
    return parsed;
}

static uint64_t monotonic_ns(void) {
    struct timespec timestamp;
    if (clock_gettime(CLOCK_MONOTONIC, &timestamp) != 0) {
        fail("clock_gettime failed");
    }
    return (uint64_t)timestamp.tv_sec * 1000000000ULL + (uint64_t)timestamp.tv_nsec;
}

static double signal_value(size_t index) {
    int primary = (int)((index * 17) % 257) - 128;
    int secondary = (int)(index % 11) - 5;
    return (double)primary / 64.0 + (double)secondary / 16.0;
}

static int transform(operation *op) {
    if (op->inverse) {
        return gsl_wavelet_transform_inverse(
            op->wavelet, op->data, 1, op->length, op->workspace
        );
    }
    return gsl_wavelet_transform_forward(
        op->wavelet, op->data, 1, op->length, op->workspace
    );
}

static uint64_t run_batch(operation *op, size_t iterations) {
    uint64_t start = monotonic_ns();
    for (size_t iteration = 0; iteration < iterations; ++iteration) {
        if (transform(op) != GSL_SUCCESS) {
            fail("GSL transform failed");
        }
    }
    return monotonic_ns() - start;
}

static size_t calibrate(operation *op, double target_ns) {
    double minimum_ns = target_ns / 4.0;
    size_t iterations = 1;
    for (;;) {
        uint64_t elapsed_ns = run_batch(op, iterations);
        if ((double)elapsed_ns >= minimum_ns || iterations >= MAX_BATCH_ITERATIONS) {
            double denominator = elapsed_ns == 0 ? 1.0 : (double)elapsed_ns;
            double estimate = ceil((double)iterations * target_ns / denominator);
            if (estimate < 1.0) {
                return 1;
            }
            if (estimate > (double)MAX_BATCH_ITERATIONS) {
                return (size_t)MAX_BATCH_ITERATIONS;
            }
            return (size_t)estimate;
        }
        if (iterations > MAX_BATCH_ITERATIONS / 2) {
            iterations = MAX_BATCH_ITERATIONS;
        } else {
            iterations *= 2;
        }
    }
}

static double checksum(const double *values, size_t length) {
    double total = 0.0;
    for (size_t index = 0; index < length; ++index) {
        total += fabs(values[index]);
    }
    if (!isfinite(total)) {
        fail("benchmark output checksum is not finite");
    }
    return total;
}

int main(int argc, char **argv) {
    config parsed = parse_config(argc, argv);
    gsl_set_error_handler_off();

    double *data = malloc(parsed.length * sizeof(*data));
    if (data == NULL) {
        fail("failed to allocate signal buffer");
    }
    for (size_t index = 0; index < parsed.length; ++index) {
        data[index] = signal_value(index);
    }

    gsl_wavelet *wavelet = gsl_wavelet_alloc(gsl_wavelet_haar, 2);
    gsl_wavelet_workspace *workspace = gsl_wavelet_workspace_alloc(parsed.length);
    if (wavelet == NULL || workspace == NULL) {
        fail("failed to allocate GSL wavelet state");
    }

    operation op = {
        .wavelet = wavelet,
        .workspace = workspace,
        .data = data,
        .length = parsed.length,
        .inverse = strcmp(parsed.direction, "inverse") == 0,
    };
    if (op.inverse && gsl_wavelet_transform_forward(wavelet, data, 1, parsed.length, workspace) != GSL_SUCCESS) {
        fail("failed to prepare inverse coefficients");
    }
    if (transform(&op) != GSL_SUCCESS) {
        fail("GSL checksum transform failed");
    }
    double output_checksum = checksum(data, parsed.length);

    double target_ns = parsed.sample_time_ms * 1.0e6;
    size_t batch_iterations = calibrate(&op, target_ns);
    for (size_t batch = 0; batch < parsed.warmup_batches; ++batch) {
        (void)run_batch(&op, batch_iterations);
    }

    printf(
        "{\"schema\":%d,\"engine\":{\"name\":\"gsl\",\"version\":\"%s\","
        "\"language\":\"C\",\"clock\":\"CLOCK_MONOTONIC\",\"target\":\"native\","
        "\"comparison_scope\":\"complete periodic f64 Haar transforms\"},"
        "\"results\":[{\"case_id\":\"%s\",\"api\":\"into\","
        "\"batch_iterations\":%zu,\"samples_ns\":[",
        SCHEMA_VERSION,
        GSL_VERSION,
        parsed.case_id,
        batch_iterations
    );
    for (size_t sample = 0; sample < parsed.samples; ++sample) {
        uint64_t elapsed_ns = run_batch(&op, batch_iterations);
        if (sample != 0) {
            putchar(',');
        }
        printf("%.17g", elapsed_ns / (double)batch_iterations);
    }
    printf("],\"checksum\":%.17g}]}\n", output_checksum);

    gsl_wavelet_workspace_free(workspace);
    gsl_wavelet_free(wavelet);
    free(data);
    return EXIT_SUCCESS;
}
