#ifndef MCAF_GUARD_H
#define MCAF_GUARD_H

#include <stdbool.h>
#include <stdint.h>
#include "shield_tables.h"

typedef struct {
    double brake_temp_c;
    double speed_kmh;
    double engine_temp_c;
    mcaf_cmd_t operator_command;
    bool auth_token;
} mcaf_sensors_t;

typedef struct {
    uint8_t states[N_SHIELD_RULES];
    bool prev_auth;
} mcaf_shield_state_t;

typedef struct {
    mcaf_action_t executed_action;
    int blocked_by_idx;
} mcaf_verdict_t;

void mcaf_shield_init(mcaf_shield_state_t *st);
mcaf_verdict_t mcaf_shield_step(
    mcaf_shield_state_t *st,
    const mcaf_sensors_t *sens,
    mcaf_action_t proposed
);

#endif
