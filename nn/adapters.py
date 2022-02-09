

class DefaultGenomeAdapter(object):
    def __init__(self, nodes, connections):
        self.nodes = nodes
        self.connections = connections


class ConfigAdapter(object):
    def __init__(self, genome_config_adapter):
        self.genome_config = genome_config_adapter


class DefaultGenomeConfigAdapter(object):
    def __init__(self, input_keys, output_keys, agg_funcs, act_funcs):
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.aggregation_function_defs = agg_funcs
        self.activation_defs = act_funcs