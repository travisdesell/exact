#!/bin/bash
for folder in common examm mpi multithreaded rnn rnn_examples rnn_tests time_series weights word_series; do
  find $folder -type f -name "*.*xx" -exec clang-format -style=file -i {} \;
done
