#!/usr/bin/env python3

import re
import sys
import os
import shutil
from pathlib import Path
from subprocess import check_call
from argparse import ArgumentParser


def install():
    parser = ArgumentParser()
    parser.add_argument(
        "graphgym_path", type=str, help="Path to the GraphGym folder.", default=None, nargs="?"
    )
    args = parser.parse_args()

    # obtain the graphgym path #####################################################################
    GRAPHGYM_PATH = args.graphgym_path
    if GRAPHGYM_PATH is None:
        GRAPHGYM_PATH = input("Please enter the path to the GraphGym folder: ")
    GRAPHGYM_PATH = Path(GRAPHGYM_PATH).absolute()
    if not GRAPHGYM_PATH.exists():
        raise ValueError(f"GraphGym path {GRAPHGYM_PATH} does not exist.")
    if not (GRAPHGYM_PATH / "graphgym").exists():
        # the user might have provided a path to the graphgym source folder
        GRAPHGYM_PATH = GRAPHGYM_PATH.parent
        assert (GRAPHGYM_PATH / "graphgym").exists(), "GraphGym path does not look right."

    # install the contrib files from the contrib source folder #####################################
    contrib_source_folder = Path(__file__).parent / "contrib"
    file_list = sum(
        [
            [
                (Path(root) / fname).relative_to(contrib_source_folder)
                for fname in fnames
                if re.fullmatch(".*?\.py", fname)
            ]
            for (root, _, fnames) in os.walk(contrib_source_folder)
        ],
        [],
    )
    for file in file_list:
        file_path = contrib_source_folder / file
        new_path = GRAPHGYM_PATH / "graphgym" / "contrib" / file
        new_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Copied {file_path} to {new_path}.")
        shutil.copy(file_path, new_path)

    # install the modified graphgym ################################################################
    check_call([sys.executable, "-m", "pip", "install", str(GRAPHGYM_PATH)])

if __name__ == "__main__":
    install()