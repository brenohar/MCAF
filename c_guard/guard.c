#include "guard.h"

static inline uint8_t compute_valuation_key(
    int rule_idx, const mcaf_sensors_t *s, mcaf_action_t act
) {
    switch (rule_idx) {
        case 0: {
            uint8_t b0 = (s->brake_temp_c > 200.0) ? 1u : 0u;
            uint8_t b1 = (act == ACT_ESTOP) ? 1u : 0u;
            return (uint8_t)((b0 << 1) | b1);
        }
        case 1: {
            uint8_t b0 = (s->speed_kmh > 0.0 && s->operator_command == CMD_REVERSE) ? 1u : 0u;
            uint8_t b1 = (s->speed_kmh > 0.0) ? 1u : 0u;
            uint8_t b2 = (act == ACT_ESTOP) ? 1u : 0u;
            uint8_t b3 = (act == ACT_REVERSE) ? 1u : 0u;
            return (uint8_t)((b0 << 3) | (b1 << 2) | (b2 << 1) | b3);
        }
        case 2: {
            uint8_t b0 = (s->speed_kmh > 20.0) ? 1u : 0u;
            uint8_t b1 = (act == ACT_ESTOP) ? 1u : 0u;
            return (uint8_t)((b0 << 1) | b1);
        }
        case 3: {
            uint8_t b0 = (s->engine_temp_c > 105.0) ? 1u : 0u;
            uint8_t b1 = (act == ACT_COOLING || act == ACT_ESTOP) ? 1u : 0u;
            return (uint8_t)((b0 << 1) | b1);
        }
        default: {
            bool risk_viol = (act == ACT_EXEC_RISK) || (
                s->operator_command == CMD_OP_RISK &&
                !s->auth_token &&
                act != ACT_REFUSE &&
                act != ACT_ESTOP &&
                act != ACT_COOLING
            );
            uint8_t b0 = risk_viol ? 1u : 0u;
            uint8_t b1 = s->auth_token ? 1u : 0u;
            return (uint8_t)((b0 << 1) | b1);
        }
    }
}

void mcaf_shield_init(mcaf_shield_state_t *st) {
    for (int i = 0; i < N_SHIELD_RULES; ++i) {
        st->states[i] = MONA_COMPILED_RULES[i].initial_state;
    }
    st->prev_auth = false;
}

mcaf_verdict_t mcaf_shield_step(
    mcaf_shield_state_t *st,
    const mcaf_sensors_t *sens,
    mcaf_action_t proposed
) {
    if (st->prev_auth && !sens->auth_token) {
        st->states[4] = MONA_COMPILED_RULES[4].initial_state;
    }
    st->prev_auth = sens->auth_token;

    mcaf_action_t exec_act = proposed;
    int blocked_idx = -1;

    for (int i = 0; i < N_SHIELD_RULES; ++i) {
        uint8_t cur_q = st->states[i];
        uint8_t key = compute_valuation_key(i, sens, exec_act);
        uint8_t nxt_q = MONA_COMPILED_RULES[i].transitions[cur_q][key];
        if (((MONA_COMPILED_RULES[i].accepting_mask >> nxt_q) & 1u) == 0u) {
            if (blocked_idx < 0) {
                blocked_idx = i;
            }
            exec_act = MONA_COMPILED_RULES[i].corrective_action;
        }
    }

    for (int i = 0; i < N_SHIELD_RULES; ++i) {
        uint8_t cur_q = st->states[i];
        uint8_t key = compute_valuation_key(i, sens, exec_act);
        st->states[i] = MONA_COMPILED_RULES[i].transitions[cur_q][key];
    }

    mcaf_verdict_t v = { .executed_action = exec_act, .blocked_by_idx = blocked_idx };
    return v;
}
