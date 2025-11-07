import subprocess


def run_examm_mt(
        train_file: str, val_file: str, num_threads: str, offset: int, in_params: str,
        out_params: str, num_islands: int, island_size: int, max_genomes: int,
        bp_iter: int, output_dir: str
    ):
    """
    Runs multithreaded version of examm as a seprate os process
    
    Parameters:
        train_file (str)
        val_file (str)
        num_threads (str)
        offset (int)
        in_params (str)
        out_params (str)
        num_islands (int)
        island_size (int)
        max_genomes (int)
        bp_iter (int)
        output_dir (str)
    """

    cmd = [
        'cd ../build;',
        './multithreaded/examm_mt',
        '--number_threads', num_threads,
        '--training_filenames', train_file,
        '--validation_filenames', val_file,
        '--time_offset', offset,
        '--input_parameter_names', in_params,
        '--output_parameter_names', out_params,
        '--number_islands', num_islands,
        '--island_size', island_size,
        '--max_genomes', max_genomes,
        'bp_iterations', bp_iter,
        'output_directory', output_dir,
        '--possible_node_types', 'simple UGRNN MGU GRU delta LSTM',
        '--std_message_level INFO --file_message_level INFO'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Process the output
    if result.returncode != 0:
        print("EXAMM failed:", result.stderr)
        return None

    return result.stdout