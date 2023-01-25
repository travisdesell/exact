#!/bin/bash
find . -type f -name "*.*xx" -exec clang-format -style=file -i {} \;
