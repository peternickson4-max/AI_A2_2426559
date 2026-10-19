
"""Single-command reproducible pipeline entry point."""
import argparse, sys
from src.pipeline import run_all
def main():
    parser = argparse.ArgumentParser(description="Recreate personalized harvest ML pipeline.")
    parser.add_argument("--student-id", required=True, help="Institutional email")
    parser.add_argument("--seed", type=int, required=True, help="Last four registration digits")
    args = parser.parse_args()
    try:
        run_all(args.student_id, args.seed)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
if __name__ == "__main__":
    main()
