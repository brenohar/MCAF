/* AUTO-GENERATED FROM specs/compiled_shield.json (MONA DFA) - DO NOT EDIT */
#ifndef MCAF_SHIELD_TABLES_H
#define MCAF_SHIELD_TABLES_H

#include <stdint.h>

#define N_SHIELD_RULES 5
#define MAX_DFA_STATES 4
#define MAX_VALUATIONS 16

typedef enum {
    ACT_CONTINUE = 0,
    ACT_COOLING = 1,
    ACT_ESTOP = 2,
    ACT_REFUSE = 3,
    ACT_MAINT = 4,
    ACT_REDUCE_SPEED = 5,
    ACT_REVERSE = 6,
    ACT_EXEC_RISK = 7
} mcaf_action_t;

typedef enum {
    CMD_CONTINUE = 0,
    CMD_REVERSE = 1,
    CMD_OP_RISK = 2
} mcaf_cmd_t;

typedef struct {
    const char *rule_id;
    uint8_t initial_state;
    uint32_t accepting_mask;
    mcaf_action_t corrective_action;
    uint8_t transitions[MAX_DFA_STATES][MAX_VALUATIONS];
} dfa_rule_table_t;

static const dfa_rule_table_t MONA_COMPILED_RULES[N_SHIELD_RULES] = {
    {
        .rule_id = "phi_brake",
        .initial_state = 0u,
        .accepting_mask = 0x01u,
        .corrective_action = ACT_ESTOP,
        .transitions = {
            { 0u, 0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u },
            { 1u, 1u, 1u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u },
        }
    },
    {
        .rule_id = "phi_gear",
        .initial_state = 0u,
        .accepting_mask = 0x01u,
        .corrective_action = ACT_ESTOP,
        .transitions = {
            { 0u, 0u, 0u, 0u, 0u, 1u, 0u, 1u, 1u, 1u, 0u, 0u, 1u, 1u, 0u, 1u },
            { 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u },
        }
    },
    {
        .rule_id = "phi_speed",
        .initial_state = 0u,
        .accepting_mask = 0x01u,
        .corrective_action = ACT_ESTOP,
        .transitions = {
            { 0u, 0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u },
            { 1u, 1u, 1u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u },
        }
    },
    {
        .rule_id = "phi_temp",
        .initial_state = 0u,
        .accepting_mask = 0x01u,
        .corrective_action = ACT_COOLING,
        .transitions = {
            { 0u, 0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u },
            { 1u, 1u, 1u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u },
        }
    },
    {
        .rule_id = "phi_auth",
        .initial_state = 0u,
        .accepting_mask = 0x03u,
        .corrective_action = ACT_REFUSE,
        .transitions = {
            { 0u, 1u, 2u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u },
            { 1u, 1u, 1u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u },
            { 2u, 2u, 2u, 2u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u },
        }
    },
};

#endif /* MCAF_SHIELD_TABLES_H */
