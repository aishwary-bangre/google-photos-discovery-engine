"""Orchestrator.

    python -m pipeline.run                    # all stages
    python -m pipeline.run --stages gate extract cluster
    python -m pipeline.run --sources play_store app_store
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

from . import config as C

STAGES = ["collect", "gate", "extract", "cluster"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="*", default=STAGES, choices=STAGES)
    ap.add_argument("--sources", nargs="*", default=None)
    a = ap.parse_args()

    log = {"started": datetime.now(timezone.utc).isoformat(), "stages": {}}
    for stage in STAGES:
        if stage not in a.stages:
            continue
        t0 = time.time()
        print(f"\n===== {stage.upper()} =====")
        if stage == "collect":
            from . import collect
            res = collect.run(a.sources)
        elif stage == "gate":
            from . import gate
            res = len(gate.run())
        elif stage == "extract":
            from . import extract
            res = len(extract.run())
        else:
            from . import cluster
            res = cluster.run()
        log["stages"][stage] = {"seconds": round(time.time() - t0, 1), "result": res}
    log["finished"] = datetime.now(timezone.utc).isoformat()
    (C.PROC_DIR / "run_log.json").write_text(json.dumps(log, indent=2, default=str))


if __name__ == "__main__":
    main()
