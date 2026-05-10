"""CLI entry point.

Examples:
    python main.py --mode stat                 # CPU baseline
    python main.py --mode logprobs             # GPU — main signal
    python main.py --mode all
    python main.py --mode logprobs --upload    # train, save CSV, send to server
"""
from __future__ import annotations
import argparse
import subprocess
import sys

import pipeline


def main():
    parser = argparse.ArgumentParser(description="LLM watermark detection pipeline")
    parser.add_argument("--mode", choices=["stat", "logprobs", "all"], default="stat",
                        help="which feature set(s) to use")
    parser.add_argument("--upload", action="store_true",
                        help="upload the resulting submission CSV to the server")
    args = parser.parse_args()

    pipeline.run(args.mode)

    if args.upload:
        print("\n[main] running upload.py on the latest submission ...")
        subprocess.check_call([sys.executable, "upload.py"])


if __name__ == "__main__":
    main()
