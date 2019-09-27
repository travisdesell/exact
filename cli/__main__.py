#!/usr/bin/env python3
import subprocess
import sys

import toml

from examm_task import ExammTask

USAGE = """
Usage:
python3 cli [config_file_path]
"""

def main():
    args = sys.argv

    if len(args) < 2:
        print(USAGE)
        print("You must provide a configuration file")
        return

    config_file_path = args[1]
    try:
        with open(config_file_path) as f:
            text = f.read()
    except FileNotFoundError as e:
        print(f"Could not find configuration file '{config_file_path}'.")
        print(f"Encountered error: {str(e)}")
        return
    task = ExammTask(toml.loads(text), "junk:")
    command = task.to_command()
    print(command)
    subprocess.call(command)

if __name__ == "__main__":
    main()
