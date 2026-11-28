"""Run Lab 2 benchmark configurations sequentially and persist a progress table."""

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "outputs" / "experiment_summary.csv"

# README section 9 (6A), followed by the two hyperparameter comparisons (6B).
EXPERIMENTS = [
    ("EXP_01", "resnet18", "freeze_all", 64, 1e-3, "adamw"),
    ("EXP_02", "resnet18", "unfreeze_last_block", 64, 1e-4, "adamw"),
    ("EXP_03", "resnet50", "freeze_all", 64, 1e-3, "adamw"),
    ("EXP_04", "vgg16_bn", "freeze_all", 64, 1e-3, "adamw"),
    ("EXP_05", "densenet121", "freeze_all", 64, 1e-3, "adamw"),
    ("EXP_06", "densenet121", "unfreeze_last_block", 64, 1e-4, "adamw"),
    ("EXT_LR_OPT", "resnet18", "freeze_all", 64, 1e-2, "sgd"),
    ("EXT_BATCH", "resnet18", "freeze_all", 32, 1e-3, "adamw"),
]
FIELDS = ["experiment", "model", "strategy", "batch_size", "learning_rate", "optimizer",
          "epochs", "status", "best_epoch", "best_val_accuracy_pct", "test_accuracy_pct",
          "test_macro_f1", "avg_epoch_time_s", "run_id", "checkpoint", "tensorboard",
          "report", "summary", "predictions", "log", "diagnostic"]


def persist(rows):
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    temporary = SUMMARY.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(SUMMARY)


def valid_completed(report_path, expected):
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        config = report["config"]
        artifacts = report["artifacts"]
        if (config["model"], config["strategy"], config["batch_size"],
                round(config["lr"], 12), config["optimizer"], config["subset_ratio"],
                config["dry_run"], config["epochs"]) != (
                expected[1], expected[2], expected[3], round(expected[4], 12), expected[5], 1.0,
                False, 15):
            return None
        for name in ("checkpoint", "tensorboard", "report", "summary", "predictions",
                     "training_history", "class_accuracy", "confusion_matrix"):
            path = Path(artifacts[name])
            if not path.exists() or (path.is_file() and path.stat().st_size == 0):
                return None
        with artifacts_path(artifacts["predictions"]).open(encoding="utf-8", newline="") as stream:
            predictions = list(csv.DictReader(stream))
        with artifacts_path(artifacts["summary"]).open(encoding="utf-8-sig", newline="") as stream:
            summary = next(csv.DictReader(stream))
        if len(predictions) != 10000 or len({row["sample_index"] for row in predictions}) != 10000:
            return None
        if summary["run_id"] != report["run_id"]:
            return None
        return report, artifacts, summary
    except (OSError, ValueError, KeyError, StopIteration, TypeError):
        return None


def artifacts_path(value):
    return Path(value)


def existing_for(expected):
    for path in sorted((ROOT / "outputs").rglob("report.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True):
        result = valid_completed(path, expected)
        if result:
            return path, result
    return None


def row_for(exp, status, log, data=None, diagnostic=""):
    exp_id, model, strategy, batch, lr, optimizer = exp
    row = dict.fromkeys(FIELDS, "")
    row.update(experiment=exp_id, model=model, strategy=strategy, batch_size=batch,
               learning_rate=lr, optimizer=optimizer, epochs=15, status=status,
               log=str(log), diagnostic=diagnostic)
    if data:
        report, artifacts, summary = data
        row.update(best_epoch=summary["best_epoch"],
                   best_val_accuracy_pct=summary["best_val_accuracy_pct"],
                   test_accuracy_pct=summary["test_accuracy_pct"],
                   test_macro_f1=summary["test_macro_f1"],
                   avg_epoch_time_s=summary["avg_epoch_time_s"], run_id=report["run_id"],
                   checkpoint=artifacts["checkpoint"], tensorboard=artifacts["tensorboard"],
                   report=artifacts["report"], summary=artifacts["summary"],
                   predictions=artifacts["predictions"])
    return row


def main():
    rows = []
    for exp in EXPERIMENTS:
        exp_id, model, strategy, batch, lr, optimizer = exp
        log = ROOT / "outputs" / f"{exp_id}_{model}_{strategy}_b{batch}_lr{lr:g}_{optimizer}.log"
        old = existing_for(exp)
        if old:
            report_path, data = old
            status = "SKIPPED_VALID" if exp_id == "EXP_01" else "COMPLETED"
            rows.append(row_for(exp, status, log, data,
                                "Matched full 15-epoch run; all required artifacts and 10,000 predictions validated."))
            persist(rows)
            print(f"{exp_id}: existing valid run {data[0]['run_id']}", flush=True)
            continue

        rows.append(row_for(exp, "RUNNING", log))
        persist(rows)
        command = [sys.executable, "main.py", "--model", model, "--strategy", strategy,
                   "--batch-size", str(batch), "--lr", str(lr), "--optimizer", optimizer,
                   "--epochs", "15"]
        print(f"Starting {exp_id}: {' '.join(command)}", flush=True)
        started = time.time()
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.run(command, cwd=ROOT, stdout=stream,
                                     stderr=subprocess.STDOUT, text=True)
        elapsed = time.time() - started
        if process.returncode == 0:
            matches = existing_for(exp)
            if matches:
                report_path, data = matches
                row = row_for(exp, "COMPLETED", log, data,
                              f"Exit 0; independently validated artifacts. Wall time {elapsed:.1f}s.")
                rows[-1] = row
                persist(rows)
                print(f"Completed {exp_id}: {data[0]['run_id']}; updated {SUMMARY}", flush=True)
                continue
            diagnosis = "Process exited 0 but outputs failed independent validity checks; halting for diagnosis."
        else:
            diagnosis = f"Process exited {process.returncode}; see log. Halting to protect scientific comparability."
        rows[-1] = row_for(exp, "FAILED_HALTED", log, diagnostic=diagnosis)
        persist(rows)
        print(f"{exp_id} failed; stopped subsequent runs. See {log}", flush=True)
        return process.returncode or 2
    print(f"All requested configurations accounted for. Summary: {SUMMARY}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
