#!/usr/bin/env python3
import subprocess
import argparse
import sys

import toml

from examm_task import ExammTask

CONFIG_FILE_HELP = """the EXAMM configurations that are to be ran. Each file will be ran the number of times that is specified by the '--n_runs' argument, the default is 1 time."""

N_RUNS_HELP = """how many times each EXAMM configuration should be ran. The default is 1 time. If this argument is n > 1, the output_directory will contain folders 0 through n - 1 which contain the outputs for each run."""

NO_RUN_HELP = """use this flag to view the generated EXAMM commands without running them."""

def main():
    args = sys.argv

    parser = argparse.ArgumentParser(description='Utility for running EXAMM.', add_help=True)
    parser.add_argument(
        '-n', '--n_runs',
        metavar='n',
        action="store",
        default=1,
        type=int,
        help=N_RUNS_HELP)
    parser.add_argument(
        'config_files', 
        metavar='config_file', 
        type=str, 
        nargs='+', 
        help=CONFIG_FILE_HELP)
    parser.add_argument(
        '--no_run', '-nr',
        action="store_const",
        const=True,
        default=False,
        help=NO_RUN_HELP)

    args = parser.parse_args()
    should_run = not args.no_run   
    if args.n_runs == 1:
        configs = args.config_files
        for config in configs:
            parsed = get_config(config)
            task = ExammTask(parsed, config, None)
            command = task.to_command()
            print("Generated the following command:")
            print(command)
            if should_run:
                subprocess.call(command)
    else:
        configs = args.config_files
        for config in configs:
            parsed = get_config(config)
            for i in range(args.n_runs):
                task = ExammTask(parsed, config, i)
                command = task.to_command()
                print("Generated the following command:")
                print(command)
                if should_run:
                    subprocess.call(command)

def get_config(config_file_path):
    try:
        with open(config_file_path) as f:
            text = f.read()
    except FileNotFoundError as e:
        print(f"Could not find configuration file '{config_file_path}'.")
        print(f"Encountered error: {str(e)}")
        exit(-1)
    try:
        parsed = toml.loads(text)
        return parsed
    except Exception as e:
        print(f"Failed to parse configuration file '{config_file_path}', encountered the following exception:\n{str(e)}")
        exit(-1)

if __name__ == "__main__":
    main()
