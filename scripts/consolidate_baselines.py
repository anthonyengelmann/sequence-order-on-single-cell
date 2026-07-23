#!/usr/bin/env python
"""Copy finished max500 baselines into outputs/report/ (no retraining)."""
import glob, json, os, shutil
try:
    import yaml
except Exception:
    yaml = None


def _cfg(run_dir):
    c = os.path.join(run_dir, ".hydra", "config.yaml")
    if yaml and os.path.exists(c):
        try:
            return yaml.safe_load(open(c))
        except Exception:
            pass
    return {}


def _find(model_sub, want_ordering, want_pe):
    """seed -> newest run matching (model, ordering, PE) at max_cells=500."""
    best = {}
    for mj in glob.glob("outputs/**/metrics.json", recursive=True):
        if "/report/" in mj or "/smoke/" in mj:
            continue
        rd = os.path.dirname(mj)
        try:
            m = json.load(open(mj))
        except Exception:
            continue
        if model_sub not in str(m.get("model", "")):
            continue
        c = _cfg(rd)
        data = c.get("data", {}) if isinstance(c.get("data"), dict) else {}
        rep = c.get("representation", {}) if isinstance(c.get("representation"), dict) else {}
        mdl = c.get("model", {}) if isinstance(c.get("model"), dict) else {}
        if data.get("max_cells_per_sample") != 500:
            continue
        if want_ordering is not None and rep.get("ordering") != want_ordering:
            continue
        if want_pe is not None and mdl.get("positional_encoding", "none") != want_pe:
            continue
        s, t = m.get("seed"), os.path.getmtime(mj)
        if s is not None and (s not in best or t > best[s][0]):
            best[s] = (t, rd)
    return {s: rd for s, (t, rd) in best.items()}


def _consolidate(dst, runs):
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst, exist_ok=True)
    for s, rd in sorted(runs.items()):
        d = os.path.join(dst, str(s))
        shutil.rmtree(d, ignore_errors=True)
        shutil.copytree(rd, d, dirs_exist_ok=True)
    print(f"  {dst}: {len(runs)} seeds -> {sorted(runs)}")


if __name__ == "__main__":
    print("Consolidating clean max500 baselines into outputs/report/ (no retraining) ...")
    _consolidate("outputs/report/ladder_fnn", _find("FNN", None, None))
    _consolidate("outputs/report/ladder_lstm", _find("LSTM", "rank", None))
    _consolidate("outputs/report/ladder_transformer", _find("Transformer", "rank", "none"))
    print("done.")
