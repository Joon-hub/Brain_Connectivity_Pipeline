#!/usr/bin/env python3
import sys
import argparse
import shutil
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import numpy as np

FILE_MAPPINGS = {
    'predictions/train_predictions.csv': 'predictions_train.csv',
    'predictions/cv_validation_predictions.csv': 'predictions_cv_validation.csv',
    'predictions/test_predictions.csv': 'predictions_task.csv',
}

def validate_experiment_exists(exp_name: str, base_dir: Path) -> Tuple[bool, Path]:
    p = base_dir / exp_name
    if not p.exists() or not p.is_dir():
        return False, p
    return True, p

def validate_required_files(exp_path: Path) -> Tuple[bool, List[str], List[str]]:
    existing, missing = [], []
    for src in FILE_MAPPINGS.keys():
        (existing if (exp_path / src).exists() else missing).append(src)
    return len(missing) == 0, existing, missing

def copy_file_with_validation(source: Path, target: Path, label: str) -> bool:
    try:
        if not source.exists():
            print(f"ERROR: {label}: source not found: {source}")
            return False
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if not target.exists() or source.stat().st_size != target.stat().st_size:
            print(f"ERROR: {label}: copy/size mismatch")
            return False
        print(f"  {label}: {source.name} → {target.name}")
        return True
    except Exception as ex:
        print(f"ERROR: {label}: {ex}")
        return False

def copy_experiment_files(exp_path: Path, target_dir: Path, force: bool = False) -> Tuple[int, int]:
    existing_targets = [t for t in FILE_MAPPINGS.values() if (target_dir / t).exists()]
    if existing_targets and not force:
        print(f"WARNING: will overwrite {len(existing_targets)} file(s) in {target_dir}")
        if input("Overwrite? (y/n): ").lower() != 'y':
            print("Aborted.")
            return 0, 0

    n_ok = n_fail = 0
    print("Copying files...")
    for src, tgt in FILE_MAPPINGS.items():
        ok = copy_file_with_validation(exp_path / src, target_dir / tgt, tgt)
        n_ok += ok
        n_fail += (not ok)
    return n_ok, n_fail


def validate_processed_directory(target_dir: Path) -> Tuple[bool, List[str], List[str]]:
    required = [
        'predictions_train.csv',
        'predictions_cv_validation.csv',
        'predictions_task.csv'    
    ]
    existing, missing = [], []
    for f in required:
        ((existing if (target_dir / f).exists() else missing).append(f))
    return len(missing) == 0, existing, missing

def print_summary(exp_name: str, exp_path: Path, target_dir: Path, n_ok: int, n_fail: int):
    print(f"\n{'='*60}\nBRIDGE SUMMARY\n{'='*60}\n")
    print(f"Experiment: {exp_name}")
    print(f"Source:    {exp_path}")
    print(f"Target:    {target_dir}\n")
    print(f"Copied OK: {n_ok}")
    print(f"Failed:    {n_fail}\n")

    all_exist, existing, missing = validate_processed_directory(target_dir)
    if all_exist:
        print("OK: All required files present in data/processed")
        for f in existing:
            sz = (target_dir / f).stat().st_size
            print(f"  {f:35s} ({sz:10,d} bytes)")
        print("\nRun:")
        print("  ./sh_files/DAG_AdvanceAnalysis.sh")
    else:
        print("ERROR: Missing required files:")
        for f in missing:
            print(f"  {f}")
        print(f"\nRe-run main pipeline if needed:\n  python run.py --experiment-name '{exp_name}'")

def list_experiments(base_dir: Path) -> int:
    if not base_dir.exists():
        print(f"ERROR: experiments directory not found: {base_dir}")
        return 1
    exps = [d for d in base_dir.iterdir() if d.is_dir()]
    if not exps:
        print(f"No experiments in {base_dir}")
        return 0
    print(f"Experiments in {base_dir}:")
    for d in sorted(exps):
        mark = "✓" if (d / 'predictions').exists() else "·"
        print(f"  {mark} {d.name}")
    return 0

def main() -> int:
    parser = argparse.ArgumentParser(
        description='Bridge run.py outputs to AdvanceAnalysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('-e', '--experiment', type=str, help='Experiment name (in results/experiments/)')
    parser.add_argument('-f', '--force', action='store_true', help='Overwrite without asking')
    parser.add_argument('-l', '--list', action='store_true', help='List experiments and exit')
    args = parser.parse_args()

    base_dir = Path('results/experiments')
    target_dir = Path('data/processed')

    print(f"\n{'='*60}\nBRIDGE TO ADVANCE ANALYSIS\n{'='*60}\n")

    if args.list:
        return list_experiments(base_dir)

    if not args.experiment:
        print("ERROR: no experiment specified")
        print("Use: --experiment <name> or --list")
        return 1

    exp_name = args.experiment
    exists, exp_path = validate_experiment_exists(exp_name, base_dir)
    if not exists:
        print(f"ERROR: experiment not found: {exp_name}")
        if base_dir.exists():
            print("\nAvailable experiments:")
            for d in base_dir.iterdir():
                if d.is_dir():
                    print(f"  - {d.name}")
        return 1

    print(f"OK: Using experiment: {exp_name} ({exp_path})")

    all_exist, existing, missing = validate_required_files(exp_path)
    print(f"\nRequired files in experiment: {len(existing)}/{len(FILE_MAPPINGS)}")
    if existing:
        print("  found:")
        for f in existing:
            print(f"    {f}")
    if missing:
        print("  missing:")
        for f in missing:
            print(f"    {f}")

    n_ok, n_fail = copy_experiment_files(exp_path, target_dir, args.force)

    print_summary(exp_name, exp_path, target_dir, n_ok, n_fail)
    return 1 if n_fail else 0

if __name__ == '__main__':
    sys.exit(main())
