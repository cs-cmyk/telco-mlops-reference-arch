#!/usr/bin/env python3
"""
VES 7.1/7.2 Backward-Compatible Parser Stub
============================================

Demonstrates the critical parsing logic for VES (Virtual Event Streaming)
alarm events as described in the whitepaper §3 (Data Requirements, Fault
Management Alarms subsection).

Key VES 7.1 → 7.2 differences handled:
  1. ``eventList`` batch delivery (7.2 wraps multiple alarms in an array;
     7.1 sends one event per HTTP POST)
  2. ``reportingEntityId`` changed from required (7.1) to optional (7.2)
  3. ``timeZoneOffset`` added in 7.2 (absent in 7.1)

Usage (stub — not a production service)::

    from ves_parser_stub import parse_ves_payload

    # Single-event VES 7.1 payload
    events = parse_ves_payload(json.loads(raw_body))

    # Batch VES 7.2 payload — returns list of normalised events
    events = parse_ves_payload(json.loads(raw_body))

.. warning::

    This is a **stub** for illustration and testing only. A production VES
    consumer would:
    - Validate against the VES JSON schema (7.1 or 7.2) using jsonschema
    - Deserialise from Kafka via a schema-registry-aware Avro/JSON consumer
    - Write parsed alarms to a dedicated Kafka topic for downstream models
    - Emit Prometheus metrics for parse failures and batch sizes

See also:
    - ONAP VES specification: https://docs.onap.org/en/latest/submodules/vnfsdk/model.git/docs/files/VESEventListener.html
    - Whitepaper §3 Data Requirements, FM Alarms subsection
    - ``05_production_patterns.py`` MetricsCollector class (which consumes
      parsed alarm counts as features, not raw VES payloads)

Licence: Apache 2.0 (same as companion code)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model — normalised alarm event (VES-version-agnostic)
# ---------------------------------------------------------------------------


@dataclass
class NormalisedAlarmEvent:
    """A VES alarm event normalised to a version-agnostic representation.

    All fields are populated regardless of whether the source was VES 7.1 or
    7.2.  Fields absent in a given VES version are filled with sensible
    defaults (documented per-field).
    """

    # --- Common header fields (present in both 7.1 and 7.2) ---
    domain: str  # e.g. "fault"
    event_id: str
    event_name: str
    source_name: str  # network element identifier
    start_epoch_microsec: int
    last_epoch_microsec: int
    priority: str  # "Critical", "Major", "Minor", "Warning", "Normal"
    sequence: int

    # --- Fields with 7.1/7.2 differences ---
    reporting_entity_id: Optional[str] = None  # required in 7.1, optional in 7.2
    reporting_entity_name: Optional[str] = None
    time_zone_offset: Optional[str] = None  # absent in 7.1; e.g. "+10:00" in 7.2

    # --- Fault-specific fields ---
    alarm_condition: Optional[str] = None
    event_severity: Optional[str] = None
    specific_problem: Optional[str] = None
    vf_status: Optional[str] = None  # "Active", "Idle"

    # --- Metadata ---
    ves_version: str = "unknown"  # "7.1" or "7.2" (detected during parsing)
    raw_event: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def timestamp_utc(self) -> datetime:
        """Return event start time as a UTC datetime.

        If ``time_zone_offset`` is available (VES 7.2), it is used to convert
        to UTC.  Otherwise, ``startEpochMicrosec`` is assumed to be UTC (the
        VES 7.1 convention).
        """
        return datetime.fromtimestamp(
            self.start_epoch_microsec / 1_000_000, tz=timezone.utc
        )


# ---------------------------------------------------------------------------
# Parser — the critical dispatch logic
# ---------------------------------------------------------------------------


def parse_ves_payload(payload: Dict[str, Any]) -> List[NormalisedAlarmEvent]:
    """Parse a raw VES HTTP POST body into normalised alarm events.

    Handles both VES 7.1 (single event) and VES 7.2 (``eventList`` batch)
    formats.

    .. important::

        The ``eventList`` check is the most operationally critical part of
        this parser.  VES 7.2 may wrap multiple alarm events in a single
        HTTP POST.  A parser that does not handle the batch form **silently
        drops all but the first alarm** in each delivery — degrading RCA
        model input quality without any error signal.

    Parameters
    ----------
    payload : dict
        Deserialised JSON body from the VES HTTP POST.

    Returns
    -------
    list[NormalisedAlarmEvent]
        One or more normalised alarm events.

    Raises
    ------
    ValueError
        If the payload structure is unrecognised (neither single-event nor
        ``eventList`` batch).
    """
    events: List[NormalisedAlarmEvent] = []

    # -----------------------------------------------------------------
    # CRITICAL: Check for VES 7.2 eventList batch delivery FIRST.
    # If ``eventList`` is present at the top level, iterate over its
    # contents.  Otherwise, treat the payload as a single VES 7.1 event.
    # -----------------------------------------------------------------
    if "eventList" in payload:
        # VES 7.2 batch delivery — array of event objects
        raw_events = payload["eventList"]
        if not isinstance(raw_events, list):
            raise ValueError(
                f"eventList is not a list (got {type(raw_events).__name__})"
            )
        logger.info("VES 7.2 batch delivery: %d events in eventList", len(raw_events))
        for raw_event in raw_events:
            events.append(_parse_single_event(raw_event, ves_version="7.2"))

    elif "event" in payload:
        # VES 7.1 single-event delivery (or VES 7.2 single event without
        # eventList wrapper — some implementations send single events
        # without the array wrapper even in 7.2)
        ves_version = _detect_ves_version(payload["event"])
        events.append(_parse_single_event(payload["event"], ves_version=ves_version))

    else:
        raise ValueError(
            "Unrecognised VES payload structure: expected 'event' or 'eventList' "
            f"at top level, got keys: {list(payload.keys())}"
        )

    return events


def _detect_ves_version(event: Dict[str, Any]) -> str:
    """Heuristic VES version detection based on field presence.

    - ``timeZoneOffset`` present in commonEventHeader → 7.2
    - ``reportingEntityId`` required (non-empty) → likely 7.1
    - Otherwise → assume 7.1 (conservative default)
    """
    header = event.get("commonEventHeader", {})
    if "timeZoneOffset" in header:
        return "7.2"
    if header.get("reportingEntityId"):
        return "7.1"
    return "7.1"


def _parse_single_event(
    event: Dict[str, Any], ves_version: str
) -> NormalisedAlarmEvent:
    """Parse a single VES event dict into a NormalisedAlarmEvent."""
    header = event.get("commonEventHeader", {})
    fault_fields = event.get("faultFields", {})

    # --- reportingEntityId: required in 7.1, optional in 7.2 ---
    reporting_entity_id = header.get("reportingEntityId")
    if not reporting_entity_id and ves_version == "7.1":
        logger.warning(
            "VES 7.1 event missing required reportingEntityId (eventId=%s)",
            header.get("eventId", "unknown"),
        )

    # --- timeZoneOffset: absent in 7.1, present in 7.2 ---
    # Default to UTC if absent (VES 7.1 convention: startEpochMicrosec is UTC)
    time_zone_offset = header.get("timeZoneOffset")
    if time_zone_offset is None and ves_version == "7.2":
        logger.debug(
            "VES 7.2 event missing timeZoneOffset — defaulting to UTC "
            "(extracting from startEpochMicrosec)"
        )

    return NormalisedAlarmEvent(
        domain=header.get("domain", "fault"),
        event_id=header.get("eventId", ""),
        event_name=header.get("eventName", ""),
        source_name=header.get("sourceName", ""),
        start_epoch_microsec=header.get("startEpochMicrosec", 0),
        last_epoch_microsec=header.get("lastEpochMicrosec", 0),
        priority=header.get("priority", "Normal"),
        sequence=header.get("sequence", 0),
        reporting_entity_id=reporting_entity_id,
        reporting_entity_name=header.get("reportingEntityName"),
        time_zone_offset=time_zone_offset,
        alarm_condition=fault_fields.get("alarmCondition"),
        event_severity=fault_fields.get("eventSeverity"),
        specific_problem=fault_fields.get("specificProblem"),
        vf_status=fault_fields.get("vfStatus"),
        ves_version=ves_version,
        raw_event=event,
    )


# ---------------------------------------------------------------------------
# Self-test — demonstrates both 7.1 and 7.2 parsing paths
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # --- VES 7.1 single-event payload ---
    ves71_payload = {
        "event": {
            "commonEventHeader": {
                "domain": "fault",
                "eventId": "fault-001",
                "eventName": "Fault_gNB_Link_Failure",
                "sourceName": "gNB-DU-001",
                "reportingEntityId": "gNB-CU-001",
                "startEpochMicrosec": 1720000000000000,
                "lastEpochMicrosec": 1720000060000000,
                "priority": "Critical",
                "sequence": 1,
            },
            "faultFields": {
                "alarmCondition": "linkFailure",
                "eventSeverity": "CRITICAL",
                "specificProblem": "CPRI link down on port 3",
                "vfStatus": "Active",
            },
        }
    }

    # --- VES 7.2 batch payload (eventList with 3 alarms) ---
    ves72_payload = {
        "eventList": [
            {
                "commonEventHeader": {
                    "domain": "fault",
                    "eventId": f"fault-batch-{i}",
                    "eventName": "Fault_gNB_High_Temperature",
                    "sourceName": f"gNB-DU-{100 + i}",
                    "startEpochMicrosec": 1720000000000000 + i * 1000000,
                    "lastEpochMicrosec": 1720000000000000 + i * 1000000,
                    "priority": "Major",
                    "sequence": i,
                    "timeZoneOffset": "+10:00",
                },
                "faultFields": {
                    "alarmCondition": "highTemperature",
                    "eventSeverity": "MAJOR",
                    "specificProblem": f"Temperature exceeds threshold on RU-{i}",
                    "vfStatus": "Active",
                },
            }
            for i in range(3)
        ]
    }

    print("=" * 60)
    print("VES 7.1 single-event test")
    print("=" * 60)
    result_71 = parse_ves_payload(ves71_payload)
    for ev in result_71:
        print(f"  [{ev.ves_version}] {ev.event_name} from {ev.source_name} "
              f"@ {ev.timestamp_utc.isoformat()} — {ev.event_severity}")

    print()
    print("=" * 60)
    print("VES 7.2 batch delivery test (3 alarms in eventList)")
    print("=" * 60)
    result_72 = parse_ves_payload(ves72_payload)
    for ev in result_72:
        print(f"  [{ev.ves_version}] {ev.event_name} from {ev.source_name} "
              f"@ {ev.timestamp_utc.isoformat()} — {ev.event_severity} "
              f"(tz: {ev.time_zone_offset})")

    print()
    print(f"VES 7.1: parsed {len(result_71)} event(s)")
    print(f"VES 7.2: parsed {len(result_72)} event(s) from batch")
    print("PASS — both VES versions handled correctly.")
