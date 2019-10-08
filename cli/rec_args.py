from config_to_arg import ConfigToArg

class RecArgs:
    
    CONFIG_OPTIONS = {
            'min': lambda self, x: ['--rec_delay_min', str(x)],
            'max': lambda self, x: ['--rec_delay_max', str(x)],
            'dist': lambda self, x: ['--rec_sampling_distribution', str(x)],
            'population': lambda self, x: ['--rec_sampling_population', str(x)],
    }

    DEFAULTS = {
            'min': lambda: 1,
            'max': lambda: 50,
            'dist': lambda: 'uniform',
            'population': lambda: 'global',
    }

    TYPES = {
            'min': {int},
            'max': {int},
            'population': {str},
            'dist': {str},
    }

    CONSTRAINTS = {
            'min': (lambda self: self.min < self.max and self.min >= 1,
                    "must be less than examm.rec.max and at least 1"),
            'max': (lambda self: self.max > self.min and self.max >= 2,
                    "must be greater than examm.rec.min and at least 2"),
            'dist': (lambda self: self.dist in {'uniform', 'histogram', 'normal'},
                    "must be one of 'uniform', 'histogram', or 'normal'"),
            'population': (lambda self: self.population in {'global', 'island'},
                    "must be either 'global' or 'island'")
    }

    def __init__(self, rec, filename):
        self.c2a = ConfigToArg(rec, 'examm.rec', filename, config_options=RecArgs.CONFIG_OPTIONS,
                defaults=RecArgs.DEFAULTS, types=RecArgs.TYPES, constraints=RecArgs.CONSTRAINTS)

    def to_args(self):
        return self.c2a.to_args()
