# Genome Analysis Tools

## analyze_global_bests.cxx

This tool analyzes all global best genomes from an experiment's output directory and calculates statistics.

### Usage

```bash
./rnn_examples/analyze_global_bests \
  --std_message_level info \
  --file_message_level none \
  --output_directory /path/to/output/directory
```

### What it does

1. **Finds all global best genome files**: Searches for files matching `global_best_genome_*.bin` in the output directory
2. **Extracts fitness metrics**: Gets the best validation MSE for each genome
3. **Counts enabled weights**: Gets the number of enabled weights in each genome
4. **Calculates statistics**:
   - Min/Average/Max fitness (validation MSE)
   - Min/Average/Max number of enabled weights
5. **Generates CSV report**: Creates `global_bests_analysis.csv` with detailed results

### Output

The script generates a CSV file (`global_bests_analysis.csv`) with:
- Summary statistics (min/max/average for fitness and weights)
- Per-run details with filename, fitness, and enabled weights

### Example Results

From the coal_mpi experiment with 16 runs:
- **Fitness (Validation MSE)**: Min=0.001410, Max=0.003284, Avg=0.002473
- **Enabled Weights**: Min=46, Max=104, Avg=69.69

## finetune_rnn.cxx

This tool extends `evaluate_rnn.cxx` to add fine-tuning/post-training capabilities.

### Usage

```bash
./rnn_examples/finetune_rnn \
  --std_message_level info \
  --file_message_level none \
  --output_directory /path/to/output \
  --genome_file /path/to/genome.bin \
  --training_filenames training_data.csv \
  --testing_filenames testing_data.csv \
  --finetune_iterations 100 \
  --finetune_stochastic true
```

### Arguments

- **--genome_file**: Path to the genome binary file to fine-tune
- **--training_filenames**: CSV file(s) with training data
- **--testing_filenames**: CSV file(s) with testing data  
- **--finetune_iterations**: Number of backpropagation iterations (default: 100)
- **--finetune_stochastic**: Use stochastic backpropagation (default: true)
- **--save_predictions**: Optional flag to save prediction outputs

### What it does

1. **Loads the genome**: Reads a pre-trained genome from file
2. **Loads data**: Loads training and testing datasets
3. **Evaluates BEFORE**: Computes initial MSE on training and testing data
4. **Fine-tunes**: Runs additional backpropagation iterations
5. **Evaluates AFTER**: Computes final MSE after fine-tuning
6. **Reports improvement**: Shows the change in performance
7. **Saves genome**: Writes the fine-tuned genome to `finetuned_genome.bin`

### Example Output

```
=== BEFORE FINE-TUNING ===
Initial Training MSE: 0.050000
Initial Testing MSE: 0.060000

Starting fine-tuning with 100 iterations, stochastic: true

=== AFTER FINE-TUNING ===
Final Training MSE: 0.045000
Final Testing MSE: 0.055000
Improvement: 0.005000
```

The fine-tuned genome is saved for further use.

