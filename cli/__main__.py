#!/usr/bin/env python3
import subprocess

import toml

from examm_task import ExammTask


with open('cli/example_config.toml') as f:
    text = f.read()

task = ExammTask(toml.loads(text), "junk:")
command = task.to_command()
print(command)
subprocess.call(command)
