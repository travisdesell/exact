import os

class ExammTask:
    
    # Maps required config option name to a lambda which takes the value set of that config option
    # and returns a list of command line arguments.
    # Some type checking should be done somewhere but it's probably not worth the effort.
    REQUIRED_CONFIG_OPTIONS = {
            "training_files":       lambda x: ['--training_filenames', str(x)],
            "test_files":           lambda x: ['--test_filenames', str(x)],
            "time_offset":          lambda x: ['--time_offset', str(x)],
            "input_parameters":     lambda x: ['--input_parameter_names'] + list(map(str, x)),
            "output_parameters":    lambda x: ['--output_parameter_names'] + list(map(str, x)),
            "n_islands":            lambda x: ['--number_islands', str(x)],
            "population_size":      lambda x: ['--population_size', str(x)],
            "max_genomes":          lambda x: ['--max_genomes', str(x)],
            "bp_iterations":        lambda x: ['--bp_iterations', str(x)],
            "output_directory":     lambda x: ['--output_directory', str(x)],
    }

    # Maps optional config option name to a lambda which returns the appropriate command arguments
    # for that option.
    OPTIONAL_CONFIG_OPTIONS = {
            "parallelism": lambda x: (_ for _ in ()).throw(Exception("This shouldn't happen")), 
            "node_types": lambda x: ['--possible_node_types'] + list(map(str, x)),
    }

    # A dictionary mapping optional parameter name to a function which will calculate
    # the default value
    OPTIONAL_CONFIG_OPTION_DEFAULTS = {
            # https://stackoverflow.com/questions/1006289/how-to-find-out-the-number-of-cpus-using-python
            "parallelism": lambda: len(os.sched_getaffinity(0)) + 1,
            "node_types": lambda: [ 'simple', 'UGRNN', 'MGU', 'GRU', 'delta', 'LSTM' ]
    }

    ALL_CONFIG_OPTIONS = set(REQUIRED_CONFIG_OPTIONS.keys()).union(OPTIONAL_CONFIG_OPTIONS.keys())

    def __init__(self, toml_object, file_path):
        if 'examm' not in toml_object:
            raise Exception(f"There is no [examm] section in configuration file '{file_path}'.") 
        self.examm_config = toml_object['examm']

        # Version is a required parameter but is a special case.
        if 'version' not in self.examm_config:
            raise Exception(f"Configuration option 'version' is required but was not found in {file_path}")
        
        self.version = str(self.examm_config['version']).lower()
        if self.version in ['mt', 'multithreaded']:
            self.version = 'mt'
            ExammTask.OPTIONAL_CONFIG_OPTIONS['parallelism'] = lambda x: ['--number_threads', x]
        elif self.version == 'mpi':
            self.version = 'mpi'
            ExammTask.OPTIONAL_CONFIG_OPTIONS['parallelism'] = lambda x: ['-np', x]

        # Dynamically create fields for all configuration options
        for config_option in ExammTask.ALL_CONFIG_OPTIONS:
            setattr(self, config_option, None)
        
        for config_option in ExammTask.REQUIRED_CONFIG_OPTIONS.keys():
            if config_option not in self.examm_config:
                raise Exception(f"Configuration option '{config_option}' is required," + \
                                f" but it was not found in '{file_path}'.")
            self.__dict__[config_option] = self.examm_config[config_option]

        for config_option in ExammTask.OPTIONAL_CONFIG_OPTIONS.keys():
            if config_option not in self.examm_config:
                self.__dict__[config_option] = ExammTask.OPTIONAL_CONFIG_OPTION_DEFAULTS[config_option]()
                continue
            self.__dict__[config_option] = self.examm_config[config_option]

        for config_option in ExammTask.ALL_CONFIG_OPTIONS:
            assert self.__dict__[config_option] is not None

    def to_command(self):
        if self.version == 'mpi':
            command = ['mpirun']
            command = command + ['-np', str(self.parallelism)]
            command.append('./build/mpi/examm_mpi')
        else:
            command = ['./build/multithreaded/examm_mt', '--number_threads', str(self.parallelism)]
        
        for config_option, to_arg_fn in ExammTask.REQUIRED_CONFIG_OPTIONS.items():
            command = command + to_arg_fn(self.__dict__[config_option])
        
        for config_option, to_arg_fn in ExammTask.OPTIONAL_CONFIG_OPTIONS.items():
            # Since this is a special case / format changes based on which version you use,
            # it gets added manually beforehand
            if config_option == 'parallelism':
                continue
            command = command + to_arg_fn(self.__dict__[config_option])

        return command
