"""Find complete trace events and host-side evidence from torch async flows."""
import argparse
from bisect import bisect_left, bisect_right
from collections import defaultdict
import json
from pathlib import Path


def lane(event):
    return event.get("pid"), event.get("tid")


def evidence(event):
    fields = ("name", "cat", "pid", "tid", "ts", "dur")
    item = {field: event[field] for field in fields if field in event}
    item["args"] = {k: v for k, v in event.get("args", {}).items()
                    if any(word in k.lower() for word in ("stack", "shape", "module", "stream", "type"))}
    return item


def locate(events, keyword):
    spans = defaultdict(list)
    starts = defaultdict(list)
    ends = defaultdict(list)
    for event in events:
        if "ts" not in event:
            continue
        phase = event.get("ph")
        if phase == "X":
            spans[lane(event)].append(event)
        elif phase in ("s", "f") and "id" in event and event.get("cat") == "async_npu":
            key = (event.get("scope"), event["id"])
            if phase == "s":
                starts[key].append(event)
            else:
                ends[lane(event)].append((event["ts"], key))
    launch_times = defaultdict(set)
    for launches in starts.values():
        for launch in launches:
            launch_times[lane(launch)].add(launch["ts"])
    contexts = defaultdict(list)
    for host_lane, stamps in launch_times.items():
        ordered = sorted(stamps)
        for span in spans.get(host_lane, ()):
            left = bisect_left(ordered, span["ts"])
            right = bisect_right(ordered, span["ts"] + span.get("dur", 0))
            for stamp in ordered[left:right]:
                contexts[host_lane, stamp].append(span)

    for device_lane, records in spans.items():

        endpoints = sorted(ends[device_lane], key=lambda pair: pair[0])
        times = [pair[0] for pair in endpoints]
        for event in records:
            if keyword.casefold() not in event.get("name", "").casefold():
                continue
            host = []
            left = bisect_left(times, event["ts"])
            right = bisect_right(times, event["ts"] + event.get("dur", 0))
            for _, key in endpoints[left:right]:
                for start in starts[key]:
                    if start["ts"] <= event["ts"]:
                        host.extend(evidence(span) for span in contexts[lane(start), start["ts"]])
            yield {"event": evidence(event), "host_candidates": host}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("keyword", help="case-insensitive event name substring")
    args = parser.parse_args()
    with args.trace.open(encoding="utf-8-sig") as stream:
        data = json.load(stream)
    events = data["traceEvents"] if isinstance(data, dict) else data
    for result in locate(events, args.keyword):
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
