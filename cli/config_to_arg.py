class ConfigToArg:
    """
    A easily configurable class meant to be used for different configuration sections.
    """

    def __init__(self, config, name, filename, config_options={}, defaults={}, 
            types={}, constraints={}):
        """
        :param config a dictionary from the parsed config file
        :param name the name of this configuration section (e.g. examm.<section_name>)
        :param filename name / path of the configuration file

        :param config_options a map from config option name to a lambda used to turn
        the parameter into a list of command line arguments. The lambda takes self and the
        value of the parameter as arguments.
        
        :param defaults a map from config option name to a lambda that
        returns the default value of a parameter. If a config option does not appear
        in this map then it is not optional. If the value of an option is None,
        it won't be added to the argument list / it will remain unset.
        
        :param types maps config option name to a set of the valid types for that config option.
        
        :param constraints maps config option name to a tuple, the first element is a lambda that
        takes the config option value as an argument and returns True if it meets all constraints.
        The second value of the tuple is a short description of the constraints.
        """
        assert type(config) == dict
        self.config_options = config_options
        self.optional_config_option_defaults = defaults
        self.types = types
        self.constraints = constraints

        for config_option in config_options.keys():
            if config_option not in config:
                if config_option in defaults:
                    df = defaults[config_option]()
                    setattr(self, config_option, df)
                    continue
                else:
                    err = f"Required config option '{name}.{config_option}' was not supplied in " + \
                            f"configuration file '{filename}'"
                    raise Exception(err)
            setattr(self, config_option, config[config_option])

        for config_option, types in types.items():
            v = self.__dict__[config_option]
            if v is None:
                continue
            if type(v) not in types:
                err = f"Value for '{name}.{config_option}' is of type {type(v)}, but must " + \
                        f"be one of the following types: {str(types)}"
                raise Exception(err)
        
        for config_option, (constraints, desc) in constraints.items():
            val = self.__dict__[config_option] 
            if val is not None and not constraints(self):
                err = f"Value {val} for '{name}.{config_option}' is invalid, it must meet the " + \
                        f"following constraint(s): {desc}"
                raise Exception(err)

    def to_args(self):
        args = []
        for option, to_args_fn in self.config_options.items():
            if self.__dict__[option] is None:
                continue

            args = args + to_args_fn(self, self.__dict__[option])
        return args
