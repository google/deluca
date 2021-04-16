#!/usr/bin/env python3

import sys
import argparse
import pathlib
import shutil
import fileinput

parser = argparse.ArgumentParser(
    description="A script that generates the template for a new plugin"
)
parser.add_argument("name", type=str, help="Name of the package, formatted as deluca-name")
parser.add_argument(
    "-d",
    "--dst",
    type=pathlib.Path,
    help="Parent directory to place new plugin (default: 'PATH/deluca/..')",
    default=pathlib.Path(__file__).absolute().parent.parent.parent,
)

args = parser.parse_args()


def err_and_die(message):
    print(f"{message}\n")
    parser.print_help()
    sys.exit(1)


if not args.dst.exists() or not args.dst.is_dir():
    err_and_die(f"ERROR: '{args.dst}' either does not exist or is not a directory!")
    
path = args.dst.absolute() / f"deluca-{args.name}"

if path.exists():
    err_and_die(f"ERROR: '{path}' already exists!")

src = pathlib.Path(__file__).absolute().parent.parent / "plugin"

shutil.copytree(src, path)

for file_path in path.glob("**/*"):
    if file_path.is_file():
        with fileinput.FileInput(file_path.absolute(), inplace=True) as file:
            for line in file:
                line = line.replace("deluca.plugin", f"deluca.{args.name}")
                line = line.replace("deluca-plugin", f"deluca-{args.name}")
                line = line.replace("deluca/plugin", f"deluca/{args.name}")
                print(line, end="")

lib_path = path / "deluca" / args.name
lib_path.mkdir()

with open(lib_path / "__init__.py", "w") as init_file:
    pass

with open(lib_path / "_version.py", "w") as version_file:
    version_file.write("__version__ = 0.0.1")