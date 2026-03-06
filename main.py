from pathlib import Path

from src.experiments.runner import build_parser, run_experiments


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent
    run_experiments(args, project_root)


if __name__ == "__main__":
    main()

