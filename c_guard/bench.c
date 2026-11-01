#define _GNU_SOURCE
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include "guard.h"

#define HIST_MAX_NS 200000

static uint64_t hist_ns[HIST_MAX_NS + 1];

static inline uint64_t diff_ns(struct timespec a, struct timespec b) {
    return (uint64_t)(b.tv_sec - a.tv_sec) * 1000000000ULL +
           (uint64_t)(b.tv_nsec - a.tv_nsec);
}

static uint64_t percentile_from_hist(uint64_t total, double q) {
    uint64_t target = (uint64_t)(q * (double)total);
    if (target == 0) target = 1;
    uint64_t acc = 0;
    for (uint64_t i = 0; i <= HIST_MAX_NS; ++i) {
        acc += hist_ns[i];
        if (acc >= target) return i;
    }
    return HIST_MAX_NS;
}

static int run_self_tests(void) {
    mcaf_shield_state_t st;
    mcaf_shield_init(&st);

    mcaf_sensors_t s1 = {210.0, 10.0, 88.0, CMD_CONTINUE, false};
    mcaf_verdict_t v1 = mcaf_shield_step(&st, &s1, ACT_CONTINUE);
    if (v1.executed_action != ACT_ESTOP || v1.blocked_by_idx != 0) return 1;

    mcaf_sensors_t s2 = {90.0, 8.5, 88.0, CMD_REVERSE, false};
    mcaf_verdict_t v2 = mcaf_shield_step(&st, &s2, ACT_CONTINUE);
    if (v2.executed_action != ACT_ESTOP || v2.blocked_by_idx != 1) return 2;

    mcaf_sensors_t s3 = {90.0, 26.0, 88.0, CMD_CONTINUE, false};
    mcaf_verdict_t v3 = mcaf_shield_step(&st, &s3, ACT_CONTINUE);
    if (v3.executed_action != ACT_ESTOP || v3.blocked_by_idx != 2) return 3;

    mcaf_sensors_t s4 = {90.0, 10.0, 112.0, CMD_CONTINUE, false};
    mcaf_verdict_t v4 = mcaf_shield_step(&st, &s4, ACT_CONTINUE);
    if (v4.executed_action != ACT_COOLING || v4.blocked_by_idx != 3) return 4;

    mcaf_sensors_t s5 = {90.0, 10.0, 88.0, CMD_OP_RISK, false};
    mcaf_verdict_t v5 = mcaf_shield_step(&st, &s5, ACT_CONTINUE);
    if (v5.executed_action != ACT_REFUSE || v5.blocked_by_idx != 4) return 5;

    return 0;
}

int main(int argc, char **argv) {
    uint64_t iters = 10000000ULL;
    int pin_cpu = -1;
    int rt_prio = 0;
    int stdin_mode = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            iters = strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--cpu") == 0 && i + 1 < argc) {
            pin_cpu = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--rt-prio") == 0 && i + 1 < argc) {
            rt_prio = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--stdin-verify") == 0) {
            stdin_mode = 1;
        }
    }

    int selftest_rc = run_self_tests();
    if (selftest_rc != 0) {
        fprintf(stderr, "Self-test failed at case %d\n", selftest_rc);
        return 2;
    }

    if (stdin_mode) {
        mcaf_shield_state_t st;
        mcaf_shield_init(&st);
        double brk, spd, eng;
        int cmd_i, auth_i, prop_i;
        while (scanf("%lf %lf %lf %d %d %d", &brk, &spd, &eng, &cmd_i, &auth_i, &prop_i) == 6) {
            mcaf_sensors_t s = {brk, spd, eng, (mcaf_cmd_t)cmd_i, auth_i != 0};
            mcaf_verdict_t v = mcaf_shield_step(&st, &s, (mcaf_action_t)prop_i);
            printf("%d %d\n", (int)v.executed_action, v.blocked_by_idx);
        }
        return 0;
    }

    if (pin_cpu >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(pin_cpu, &set);
        if (sched_setaffinity(0, sizeof(set), &set) != 0) {
            perror("sched_setaffinity");
        }
    }

    bool locked_mem = false;
    bool rt_active = false;
    if (rt_prio > 0) {
        if (mlockall(MCL_CURRENT | MCL_FUTURE) == 0) {
            locked_mem = true;
        }
        struct sched_param sp;
        memset(&sp, 0, sizeof(sp));
        sp.sched_priority = rt_prio;
        if (sched_setscheduler(0, SCHED_FIFO, &sp) == 0) {
            rt_active = true;
        } else {
            perror("sched_setscheduler(SCHED_FIFO)");
        }
    }

    memset(hist_ns, 0, sizeof(hist_ns));
    mcaf_shield_state_t st;
    mcaf_shield_init(&st);

    uint32_t rng = 0xC0FFEE42u;
    uint64_t sum_ns = 0, max_ns = 0, interventions = 0;

    for (int w = 0; w < 10000; ++w) {
        mcaf_sensors_t ws = {90.0, 12.0, 92.0, CMD_CONTINUE, false};
        (void)mcaf_shield_step(&st, &ws, ACT_CONTINUE);
    }
    mcaf_shield_init(&st);

    for (uint64_t k = 0; k < iters; ++k) {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;

        mcaf_sensors_t s;
        s.brake_temp_c = 60.0 + (double)(rng & 0xFFu);
        s.speed_kmh = (double)((rng >> 8) & 0x1Fu);
        s.engine_temp_c = 75.0 + (double)((rng >> 13) & 0x3Fu);
        s.operator_command = (mcaf_cmd_t)((rng >> 19) % 3u);
        s.auth_token = ((rng >> 22) & 1u) != 0u;
        mcaf_action_t prop = (mcaf_action_t)((rng >> 24) % 6u);

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC_RAW, &t0);
        mcaf_verdict_t v = mcaf_shield_step(&st, &s, prop);
        clock_gettime(CLOCK_MONOTONIC_RAW, &t1);

        uint64_t dt = diff_ns(t0, t1);
        sum_ns += dt;
        if (dt > max_ns) max_ns = dt;
        uint64_t bucket = (dt > HIST_MAX_NS) ? HIST_MAX_NS : dt;
        hist_ns[bucket]++;
        if (v.blocked_by_idx >= 0) interventions++;
    }

    printf("{\n");
    printf("  \"iterations\": %llu,\n", (unsigned long long)iters);
    printf("  \"rt_sched_fifo\": %s,\n", rt_active ? "true" : "false");
    printf("  \"mlockall_active\": %s,\n", locked_mem ? "true" : "false");
    printf("  \"interventions\": %llu,\n", (unsigned long long)interventions);
    printf("  \"mean_ns\": %.2f,\n", (double)sum_ns / (double)iters);
    printf("  \"p50_ns\": %llu,\n", (unsigned long long)percentile_from_hist(iters, 0.50));
    printf("  \"p99_ns\": %llu,\n", (unsigned long long)percentile_from_hist(iters, 0.99));
    printf("  \"p999_ns\": %llu,\n", (unsigned long long)percentile_from_hist(iters, 0.999));
    printf("  \"p9999_ns\": %llu,\n", (unsigned long long)percentile_from_hist(iters, 0.9999));
    printf("  \"max_observed_ns\": %llu\n", (unsigned long long)max_ns);
    printf("}\n");
    return 0;
}
