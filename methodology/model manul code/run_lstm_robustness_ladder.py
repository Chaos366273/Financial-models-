"""Execute the LSTM robustness ladder (Phase 1–4) sequentially.

Writes under saved_data/lstm_robustness/<experiment_id>/ and appends
saved_data/lstm_robustness/robustness_ledger.csv after each run.

Usage:
  python run_lstm_robustness_ladder.py
  python run_lstm_robustness_ladder.py --from R3a_liq20
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
SCRIPT = ROOT / "lstm_return.py"
LEDGER = ROOT / "saved_data" / "lstm_robustness" / "robustness_ledger.csv"

# Frozen baseline reference (honest WF)
BASE_IC = 0.0069

STEPS = [
    # (id, argv extras, default_decision_hint)
    (
        "R1a_contiguous",
        ["--baseline-only", "--require-contiguous", "--out-tag", "R1a_contiguous",
         "--ledger-decision", "pending", "--notes", "P1 contiguous warm-start ON"],
        "adopt_if_alive",
    ),
    (
        "R1b_contiguous_no_warmstart",
        ["--baseline-only", "--require-contiguous", "--no-warm-start",
         "--out-tag", "R1b_contiguous_no_warmstart",
         "--ledger-decision", "diagnostic-only",
         "--notes", "P1 contiguous + warm-start OFF"],
        "diagnostic-only",
    ),
    (
        "R3a_liq20",
        ["--baseline-only", "--liquidity-drop", "0.20", "--out-tag", "R3a_liq20",
         "--ledger-decision", "pending", "--notes", "P3a liquidity drop 20%"],
        "pending",
    ),
    (
        "R2w_ref",
        ["--single-val", "--winsor", "0.01,0.99", "--out-tag", "R2w_ref",
         "--no-stability", "--ledger-decision", "pending", "--notes", "winsor ref"],
        "pending",
    ),
    (
        "R2w_tight",
        ["--single-val", "--winsor", "0.005,0.995", "--out-tag", "R2w_tight",
         "--no-stability", "--ledger-decision", "pending", "--notes", "winsor tight"],
        "pending",
    ),
    (
        "R2w_none",
        ["--single-val", "--winsor", "none", "--out-tag", "R2w_none",
         "--no-stability", "--ledger-decision", "pending", "--notes", "winsor none"],
        "pending",
    ),
    (
        "R2s_mkt_neutral",
        ["--baseline-only", "--no-sector-neutral", "--out-tag", "R2s_mkt_neutral",
         "--ledger-decision", "pending", "--notes", "P2 sector-neutral OFF"],
        "pending",
    ),
    (
        "R3b_all_names",
        ["--baseline-only", "--no-members-filter", "--out-tag", "R3b_all_names",
         "--ledger-decision", "pending", "--notes", "P3b members filter OFF"],
        "pending",
    ),
    # P4 pilots — always run; gate decision recorded in summary note
    (
        "R4h_48",
        ["--single-val", "--hidden", "48", "--out-tag", "R4h_48",
         "--no-stability", "--ledger-decision", "pending", "--notes", "P4 hidden 48"],
        "pending",
    ),
    (
        "R4h_64",
        ["--single-val", "--hidden", "64", "--out-tag", "R4h_64",
         "--no-stability", "--ledger-decision", "pending", "--notes", "P4 hidden 64"],
        "pending",
    ),
    (
        "R4h_96",
        ["--single-val", "--hidden", "96", "--out-tag", "R4h_96",
         "--no-stability", "--ledger-decision", "pending", "--notes", "P4 hidden 96"],
        "pending",
    ),
    (
        "R4d_0.1",
        ["--single-val", "--dropout", "0.1", "--out-tag", "R4d_0.1",
         "--no-stability", "--ledger-decision", "pending", "--notes", "P4 dropout 0.1"],
        "pending",
    ),
    (
        "R4d_0.2",
        ["--single-val", "--dropout", "0.2", "--out-tag", "R4d_0.2",
         "--no-stability", "--ledger-decision", "pending", "--notes", "P4 dropout 0.2"],
        "pending",
    ),
    (
        "R4lr_5e-4",
        ["--single-val", "--lr", "0.0005", "--out-tag", "R4lr_5e-4",
         "--no-stability", "--ledger-decision", "pending", "--notes", "P4 lr 5e-4"],
        "pending",
    ),
    (
        "R4lr_2e-3",
        ["--single-val", "--lr", "0.002", "--out-tag", "R4lr_2e-3",
         "--no-stability", "--ledger-decision", "pending", "--notes", "P4 lr 2e-3"],
        "pending",
    ),
]


def _done_ids() -> set[str]:
    if not LEDGER.is_file():
        return set()
    with LEDGER.open(encoding="utf-8") as f:
        return {r["experiment_id"] for r in csv.DictReader(f) if r.get("experiment_id")}


def _latest_ic(exp_id: str) -> float | None:
    if not LEDGER.is_file():
        return None
    last = None
    with LEDGER.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("experiment_id") == exp_id:
                try:
                    last = float(r["mean_ic"]) if r.get("mean_ic") not in ("", None) else None
                except ValueError:
                    last = None
    return last


def _write_gate_note(path: Path, alive: bool, detail: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join([
            "LSTM robustness signal gate",
            "===========================",
            f"alive={alive}",
            detail,
            "",
            "Criteria: mean IC > 0 and usable vs frozen baseline, or TopK hit>=55% with non-neg edge.",
            "If not alive: stop architecture search for dissertation; keep robustness tables.",
            "",
        ]),
        encoding="utf-8",
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="from_id", default="", help="Resume from this experiment id")
    ap.add_argument("--only", default="", help="Run a single experiment id")
    ap.add_argument("--skip-done", action="store_true", default=True)
    args = ap.parse_args(argv)

    start = False if args.from_id else True
    only = args.only.strip()
    done = _done_ids()

    for exp_id, extras, _hint in STEPS:
        if only and exp_id != only:
            continue
        if args.from_id and not start:
            if exp_id == args.from_id:
                start = True
            else:
                continue
        if args.skip_done and exp_id in done and not only:
            print(f"[skip] {exp_id} already in ledger")
            continue

        extras_run = list(extras)
        # If R1a was adopted, carry --require-contiguous into later WF stresses
        # (one-change vs adopted base). Skip for R1a itself and pure single-val
        # winsor/P4 pilots unless already specified.
        r1a_dec = ROOT / "saved_data" / "lstm_robustness" / "R1a_decision.txt"
        adopted_contiguous = False
        if r1a_dec.is_file() and "adopt" in r1a_dec.read_text(encoding="utf-8"):
            adopted_contiguous = True
        if (
            adopted_contiguous
            and exp_id in ("R3a_liq20", "R2s_mkt_neutral", "R3b_all_names")
            and "--require-contiguous" not in extras_run
        ):
            extras_run.append("--require-contiguous")

        cmd = [PY, str(SCRIPT), *extras_run]
        print("=" * 72)
        print("RUN:", " ".join(cmd))
        print("=" * 72)
        rc = subprocess.call(cmd, cwd=str(ROOT))
        if rc != 0:
            print(f"FAILED {exp_id} exit={rc}")
            return rc

        # After R1a: adopt contiguous if IC still positive
        if exp_id == "R1a_contiguous":
            ic = _latest_ic(exp_id)
            decision = "adopt" if (ic is not None and ic > 0) else "reject"
            note = ROOT / "saved_data" / "lstm_robustness" / "R1a_decision.txt"
            note.parent.mkdir(parents=True, exist_ok=True)
            note.write_text(f"decision={decision}\nic={ic}\n", encoding="utf-8")
            print(f"R1a decision={decision} ic={ic}")

        # After core WF stresses, write gate before P4 (R4h_48 is first P4)
        if exp_id == "R3b_all_names":
            # signal from best of baseline-like WF runs in ledger
            best_ic = None
            for cand in ("R1a_contiguous", "R3a_liq20", "R2s_mkt_neutral", "R3b_all_names"):
                ic = _latest_ic(cand)
                if ic is None:
                    continue
                if best_ic is None or ic > best_ic:
                    best_ic = ic
            # also compare to frozen baseline id implicitly
            alive = best_ic is not None and best_ic > 0
            gate_path = ROOT / "saved_data" / "lstm_robustness" / "SIGNAL_GATE.txt"
            _write_gate_note(
                gate_path,
                alive,
                f"best_wf_ic={best_ic} frozen_baseline_ic={BASE_IC}",
            )
            print(f"Signal gate alive={alive} best_ic={best_ic} -> {gate_path}")

    # Preferred recipe note
    pref = ROOT / "saved_data" / "lstm_robustness" / "PREFERRED_ROBUST_RECIPE.txt"
    r1a_ic = _latest_ic("R1a_contiguous")
    if r1a_ic is not None and r1a_ic > 0:
        recipe = (
            "preferred=R1a_contiguous\n"
            "flags=--require-contiguous (warm-start ON, members ON, sector-neutral ON, winsor 0.01/0.99)\n"
            f"mean_ic={r1a_ic}\n"
            "note=Adopt contiguous sequences if IC held; else keep original frozen baseline.\n"
        )
    else:
        recipe = (
            "preferred=frozen_baseline\n"
            "flags=(none — original champion)\n"
            f"r1a_ic={r1a_ic}\n"
            "note=Contiguous stress did not retain usable IC; keep original recipe for thesis.\n"
        )
    pref.parent.mkdir(parents=True, exist_ok=True)
    pref.write_text(recipe, encoding="utf-8")
    print("Wrote", pref)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
