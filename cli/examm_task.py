import os
import glob

from rec_args import RecArgs
from island_purging_args import IslandPurgingArgs
from config_to_arg import ConfigToArg

class ExammTask(ConfigToArg):
    @staticmethod
    def glob_to_all(paths):
        a = []
        for path in paths:
            a = a + glob.glob(path)
        return list(set(a))

    ALL_NODE_TYPES = [ 'simple', 'UGRNN', 'MGU', 'GRU', 'delta', 'LSTM' , 'ENARC' ]
    CONFIG_OPTIONS = {
            "training_files":       lambda self, x: ['--training_filenames'] + ExammTask.glob_to_all(x),
            "test_files":           lambda self, x: ['--test_filenames'] + ExammTask.glob_to_all(x),
            "time_offset":          lambda self, x: ['--time_offset', str(x)],
            "input_parameters":     lambda self, x: ['--input_parameter_names'] + list(map(str, x)),
            "output_parameters":    lambda self, x: ['--output_parameter_names'] + list(map(str, x)),
            "n_islands":            lambda self, x: ['--number_islands', str(x)],
            "population_size":      lambda self, x: ['--population_size', str(x)],
            "max_genomes":          lambda self, x: ['--max_genomes', str(x)],
            "bp_iterations":        lambda self, x: ['--bp_iterations', str(x)],
            "output_directory":     lambda self, x: ['--output_directory', str(x)],
            "node_types":           lambda self, x: ['--possible_node_types'] + list(map(str, x)),
            "rec":                  lambda self, x: RecArgs(x, self.filename).to_args(),
            "island_purging":       lambda self, x: IslandPurgingArgs(x, self.filename).to_args()
    }    

    # A dictionary mapping optional parameter name to a function which will calculate
    # the default value
    DEFAULTS = {
            "node_types": lambda: ExammTask.ALL_NODE_TYPES,
            "rec": lambda: dict(),
            "island_purging": lambda: dict()
    }

    TYPES = {
            "training_files":       {list},
            "test_files":           {list},
            "time_offset":          {int},
            "input_parameters":     {list},
            "output_parameters":    {list},
            "n_islands":            {int},
            "population_size":      {int},
            "max_genomes":          {int},
            "bp_iterations":        {int},
            "output_directory":     {str},
            "node_types":           {list},
            # Subsections should be of type dict
            "rec":                  {dict},
            "island_purging":       {dict}
 
    }

    CONSTRAINTS = {
            "training_files":       (lambda self: self.all_strings(self.training_files),
                                    "must be a list of strings"),
            "test_files":           (lambda self: self.all_strings(self.test_files),
                                    "must be a list of strings"),
            "time_offset":          (lambda self: self.time_offset > 0, "must a positive integer"),
            "input_parameters":     (lambda self: self.all_strings(self.input_parameters),
                                    "must be a list of strings"),
            "output_parameters":    (lambda self: self.all_strings(self.output_parameters),
                                    "must be a list of strings"),
            "n_islands":            (lambda self: self.n_islands > 0, "must be a positive integer"),
            "population_size":      (lambda self: self.population_size > 0, "must be a positive integer"),
            "max_genomes":          (lambda self: self.max_genomes > 0, "must be a positive integer"),
            "bp_iterations":        (lambda self: self.bp_iterations >= 0, "must be a non-negative integer"),
            "output_directory":     (lambda self: True, "must be a valid path"),
            "node_types":           (lambda self: self.all_strings(self.node_types) \
                                                and set(self.node_types).issubset(set(ExammTask.ALL_NODE_TYPES)),
                                    "must be a subset of " + str(ALL_NODE_TYPES)),
            # Subsections should be of type dict
            "rec":                  (lambda self: True, ""),
            "island_purging":       (lambda self: True, "")
 
    }


    def __init__(self, toml_object, file_path, run_number):
        if 'examm' not in toml_object:
            raise Exception(f"There is no [examm] section in configuration file '{file_path}'.") 
        self.examm_config = toml_object['examm']
        self.filename = file_path

        # Version is a required parameter but is a special case.
        if 'version' not in self.examm_config:
            raise Exception(f"Configuration option 'version' is required but was not found in {file_path}")
        
        self.version = str(self.examm_config['version']).lower()
        if self.version in ['mt', 'multithreaded']:
            self.version = 'mt'
        elif self.version == 'mpi':
            self.version = 'mpi'

        # https://stackoverflow.com/questions/1006289/how-to-find-out-the-number-of-cpus-using-python
        # "parallelism": lambda: len(os.sched_getaffinity(0)) + 1,
        # Since the master process is usually idle, it is good to use one more thread than the number of threads on the cpu.
        self.parallelism = len(os.sched_getaffinity(0)) + 1
        
        if 'parallelism' in self.examm_config:
            if type(self.examm_config['parallelism']) != int:
                raise Exception("'examm.parallelism' must be an integer")
            self.parallelism = self.examm_config['parallelism']

        ConfigToArg.__init__(self, self.examm_config, 'examm', file_path, config_options=ExammTask.CONFIG_OPTIONS,
                            defaults=ExammTask.DEFAULTS, types=ExammTask.TYPES, constraints=ExammTask.CONSTRAINTS)
        
        if run_number is not None:
            self.output_directory = os.path.join(self.output_directory, f"{run_number}")
            from pathlib import Path
            p = Path(self.output_directory)
            p.mkdir(parents=True, exist_ok=True)

    def to_command(self):
        if self.version == 'mpi':
            command = ['mpirun']
            command = command + ['-np', str(self.parallelism)]
            command.append('./build/mpi/examm_mpi')
        else:
            command = ['./build/multithreaded/examm_mt', '--number_threads', str(self.parallelism)]
         
        command = command + self.to_args()
        return command

    def all_strings(self, some_list):
        return {True} == set(map(lambda s: type(s) == str, some_list))
