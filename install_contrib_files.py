import sys
import os
import shutil
import warnings
from pathlib import Path
from subprocess import check_call


if __name__ == "__main__":
    # obtain the graphgym path #####################################################################
    GRAPHGYM_PATH = os.getenv("GRAPHGYM", None)
    if GRAPHGYM_PATH is None:
        GRAPHGYM_PATH = Path(input("Please enter the path to the GraphGym folder:")).absolute()
    if not GRAPHGYM_PATH.exists():
        raise ValueError(f"GraphGym path {GRAPHGYM_PATH} does not exist.")
    if not (GRAPHGYM_PATH / "graphgym").exists():
        # the user might have provided a path to the graphgym source folder
        GRAPHGYM_PATH = GRAPHGYM_PATH.parent
        assert (GRAPHGYM_PATH / "graphgym").exists(), "GraphGym path does not look right."

    # install the contrib files from the contrib source folder #####################################
    contrib_source_folder = Path(__file__).parents[1] / "contrib"
    file_list = [(root, dir, fname) in os.walk(contrib_source_folder)
    for file in file_list:
        file_path = contrib_source_folder / file
        if file_path.exists():
            shutil.copy(file_path, GRAPHGYM_PATH / file_list[file])
        else:
            warnings.warn(f'Path file "{file}" not found in {contrib_source_folder}')

    # install the modified graphgym ################################################################
    check_call([sys.executable, "-m", "pip", "install", str(GRAPHGYM_PATH)])
