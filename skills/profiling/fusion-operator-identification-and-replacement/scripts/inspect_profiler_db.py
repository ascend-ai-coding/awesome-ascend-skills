#!/usr/bin/env python3
"""Extract environment evidence from an ascend_pytorch_profiler_*.db (read-only).

Answers: which NPU model? how many ranks/devices? what env vars were set?
what was the capture session window? Does NOT contain CANN/torch_npu versions.

Usage:
    python inspect_profiler_db.py <ascend_pytorch_profiler_*.db>
"""
import json
import sqlite3
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    db = sys.argv[1]
    con = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
    cur = con.cursor()
    tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    out = {'db': db, 'tables': tables}

    def safe(q):
        try:
            return cur.execute(q).fetchall()
        except sqlite3.Error as e:
            return f'ERR {e}'

    if 'NPU_INFO' in tables:
        out['npu_info'] = safe('SELECT * FROM NPU_INFO')
    if 'RANK_DEVICE_MAP' in tables:
        out['rank_device_map'] = safe('SELECT * FROM RANK_DEVICE_MAP')
    if 'META_DATA' in tables:
        md = safe('SELECT name, value FROM META_DATA')
        parsed = {}
        for k, v in md:
            try:
                v = json.loads(v)
            except (ValueError, TypeError):
                pass
            parsed[k] = v
        out['meta_data'] = parsed
    if 'SESSION_TIME_INFO' in tables:
        rows = safe('SELECT startTimeNs, endTimeNs FROM SESSION_TIME_INFO')
        if rows and not isinstance(rows, str):
            s, e = rows[0]
            out['session_window'] = {'start_ns': s, 'end_ns': e,
                                     'duration_s': round((e - s) / 1e9, 3)}
    for t in ('TASK', 'PYTORCH_API', 'CANN_API'):
        if t in tables:
            r = safe(f'SELECT COUNT(*) FROM {t}')
            out[f'{t}_rows'] = r[0][0] if r and not isinstance(r, str) else r

    con.close()
    print(json.dumps(out, ensure_ascii=False, indent=2))
    p = Path('profiler_db_env.json')
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'JSON written to {p}')


if __name__ == '__main__':
    main()
