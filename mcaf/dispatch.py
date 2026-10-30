"""Typed in-process JSON-Schema validation and dispatch for every action."""
import json
import time
from typing import Dict, Tuple

from jsonschema import Draft202012Validator


class TypedInProcessDispatcher:
    """Validates 100% of executed actions (including continue_operation) via JSON Schema."""

    def __init__(self, allowed_actions: Tuple[str, ...]):
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "required": ["jsonrpc", "id", "method", "params"],
            "additionalProperties": False,
            "properties": {
                "jsonrpc": {"type": "string", "const": "2.0"},
                "id": {"type": "integer", "minimum": 0},
                "method": {"type": "string", "const": "tools/call"},
                "params": {
                    "type": "object",
                    "required": ["name", "arguments"],
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string", "enum": list(allowed_actions)},
                        "arguments": {
                            "type": "object",
                            "required": ["equipment_id", "cycle_idx"],
                            "additionalProperties": False,
                            "properties": {
                                "equipment_id": {
                                    "type": "string",
                                    "pattern": "^EQ-[0-9]{2}$",
                                },
                                "cycle_idx": {"type": "integer", "minimum": 0},
                            },
                        },
                    },
                },
            },
        }
        self._validator = Draft202012Validator(schema)

    def dispatch(self, action: str, equipment_id: str, cycle_idx: int) -> Tuple[bool, float]:
        t0 = time.perf_counter_ns()
        payload: Dict[str, object] = {
            "jsonrpc": "2.0",
            "id": int(cycle_idx),
            "method": "tools/call",
            "params": {
                "name": action,
                "arguments": {
                    "equipment_id": equipment_id,
                    "cycle_idx": int(cycle_idx),
                },
            },
        }
        wire_json = json.dumps(payload)
        decoded = json.loads(wire_json)
        self._validator.validate(decoded)
        elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
        return True, elapsed_ms
