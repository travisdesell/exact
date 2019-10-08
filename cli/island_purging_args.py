from config_to_arg import ConfigToArg

class IslandPurgingArgs:
    
    SELECTION_METHOD_MAP = {
            'worst_best': 'clear_island_with_worst_best_genome'
    }

    CONFIG_OPTIONS = {
            'selection_method': lambda self, x: ['--check_on_island_method', IslandPurgingArgs.SELECTION_METHOD_MAP[x]],
            'period':           lambda self, x: ['--num_genomes_check_worst_fit', str(x)],
    }

    DEFAULTS = {
            'selection_method': lambda: None,
            'period': lambda: None,
    }

    TYPES = {
            'period': {int},
            'selection_method': {str}
    }

    CONSTRAINTS = {
            'period': (lambda self: self.period > 0,
                    "must be a positive integer"),
            'selection_method': (lambda self: self.selection_method in {'worst_best'},
                    "must be unset or 'worst_best'")
    }

    def __init__(self, rec, filename):
        self.c2a = ConfigToArg(rec, 'examm.island_purging', filename, config_options=IslandPurgingArgs.CONFIG_OPTIONS,
                defaults=IslandPurgingArgs.DEFAULTS, types=IslandPurgingArgs.TYPES, constraints=IslandPurgingArgs.CONSTRAINTS)

    def to_args(self):
        return self.c2a.to_args()
