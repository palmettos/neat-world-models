from __future__ import annotations
from typing import Tuple
from neat.activations import ActivationFunctionSet
from neat.aggregations import AggregationFunctionSet
from neat.genes import DefaultConnectionGene, DefaultNodeGene
from neat.genome import DefaultGenome, DefaultGenomeConfig
from neat.config import ConfigParameter, write_pretty_params
from itertools import count
import sys
from random import choice, shuffle, random

from neat.graphs import creates_cycle


class WorldModelGenomeConfig(object):
    """Sets up and holds configuration information for the DefaultGenome class."""
    allowed_connectivity = [
        'full_nodirect', 'full', 'full_direct',
        'partial_nodirect', 'partial', 'partial_direct',
        'unconnected'
    ]

    def __init__(self, params):

        # Create full set of available activation functions.
        self.activation_defs = ActivationFunctionSet()
        # ditto for aggregation functions - name difference for backward compatibility
        self.aggregation_function_defs = AggregationFunctionSet()
        self.aggregation_defs = self.aggregation_function_defs

        self._params = [
            # "Inherited"
            ConfigParameter('compatibility_disjoint_coefficient', float),
            ConfigParameter('compatibility_weight_coefficient', float),
            ConfigParameter('conn_add_prob', float),
            ConfigParameter('conn_delete_prob', float),
            ConfigParameter('node_add_prob', float),
            ConfigParameter('node_delete_prob', float),
            ConfigParameter('single_structural_mutation', bool, 'false'),
            ConfigParameter('structural_mutation_surer', str, 'default'),
            ConfigParameter('initial_connection', str, 'unconnected'),

            # Extensions
            ConfigParameter('encoder_num_inputs', int),
            ConfigParameter('encoder_num_hidden', int),
            ConfigParameter('encoder_num_outputs', int),
            
            ConfigParameter('memory_num_hidden', int),

            ConfigParameter('controller_num_hidden', int),
            ConfigParameter('controller_num_outputs', int),

            ConfigParameter('compatibility_disjoint_coefficient', float),
            ConfigParameter('compatibility_weight_coefficient', float),
            ConfigParameter('conn_add_prob', float),
            ConfigParameter('conn_delete_prob', float),
            ConfigParameter('node_add_prob', float),
            ConfigParameter('node_delete_prob', float),
            ConfigParameter('single_structural_mutation', bool, 'false'),
            ConfigParameter('structural_mutation_surer', str, 'default'),
            ConfigParameter('initial_connection', str, 'unconnected')
        ]

        self.node_gene_type = params['node_gene_type']
        self._params += self.node_gene_type.get_config_params()
        self.connection_gene_type = params['connection_gene_type']
        self._params += self.connection_gene_type.get_config_params()

        for p in self._params:
            setattr(self, p.name, p.interpret(params))

        self.encoder_input_keys = [-i - 1 for i in range(self.encoder_num_inputs)]
        self.encoder_output_keys = [i for i in range(self.encoder_num_outputs)]

        self.controller_input_keys = [-i - 1 for i in range(self.encoder_num_outputs * 2)]
        self.controller_output_keys = [i for i in range(self.controller_num_outputs)]

        self.memory_input_keys = [-i - 1 for i in range(self.controller_num_outputs + self.encoder_num_outputs * 2)]
        self.memory_output_keys = [i for i in range(self.encoder_num_outputs)]

        self.connection_fraction = None

        # Verify that initial connection type is valid.
        # pylint: disable=access-member-before-definition
        if 'partial' in self.initial_connection:
            c, p = self.initial_connection.split()
            self.initial_connection = c
            self.connection_fraction = float(p)
            if not (0 <= self.connection_fraction <= 1):
                raise RuntimeError(
                    "'partial' connection value must be between 0.0 and 1.0, inclusive.")

        assert self.initial_connection in self.allowed_connectivity

        # Verify structural_mutation_surer is valid.
        # pylint: disable=access-member-before-definition
        if self.structural_mutation_surer.lower() in ['1', 'yes', 'true', 'on']:
            self.structural_mutation_surer = 'true'
        elif self.structural_mutation_surer.lower() in ['0', 'no', 'false', 'off']:
            self.structural_mutation_surer = 'false'
        elif self.structural_mutation_surer.lower() == 'default':
            self.structural_mutation_surer = 'default'
        else:
            error_string = "Invalid structural_mutation_surer {!r}".format(
                self.structural_mutation_surer)
            raise RuntimeError(error_string)

        self.node_indexer = None


    def add_activation(self, name, func):
        self.activation_defs.add(name, func)


    def add_aggregation(self, name, func):
        self.aggregation_function_defs.add(name, func)


    def save(self, f):
        if 'partial' in self.initial_connection:
            if not (0 <= self.connection_fraction <= 1):
                raise RuntimeError(
                    "'partial' connection value must be between 0.0 and 1.0, inclusive.")
            f.write('initial_connection      = {0} {1}\n'.format(self.initial_connection,
                                                                 self.connection_fraction))
        else:
            f.write('initial_connection      = {0}\n'.format(self.initial_connection))

        assert self.initial_connection in self.allowed_connectivity

        write_pretty_params(f, self, [p for p in self._params
                                      if 'initial_connection' not in p.name])


    def get_new_node_key(self, node_dict):
        if self.node_indexer is None:
            self.node_indexer = count(max(list(node_dict)) + 1)

        new_id = next(self.node_indexer)

        assert new_id not in node_dict

        return new_id


    def check_structural_mutation_surer(self):
        if self.structural_mutation_surer == 'true':
            return True
        elif self.structural_mutation_surer == 'false':
            return False
        elif self.structural_mutation_surer == 'default':
            return self.single_structural_mutation
        else:
            error_string = "Invalid structural_mutation_surer {!r}".format(
                self.structural_mutation_surer)
            raise RuntimeError(error_string)


class WorldModelGenome(DefaultGenome):

    def __init__(self, key):
        super().__init__(key)

        self.encoder_connections = {}
        self.encoder_nodes = {}

        self.memory_connections = {}
        self.memory_nodes = {}

        self.controller_connections = {}
        self.controller_nodes = {}

        del self.connections
        del self.nodes


    @classmethod
    def parse_config(cls, param_dict):
        param_dict['node_gene_type'] = DefaultNodeGene
        param_dict['connection_gene_type'] = DefaultConnectionGene
        return WorldModelGenomeConfig(param_dict)

    
    def configure_new(self, config: WorldModelGenomeConfig):
        
        for node_key in config.encoder_output_keys:
            self.encoder_nodes[node_key] = self.create_node(config, node_key)

        for node_key in config.memory_output_keys:
            self.memory_nodes[node_key] = self.create_node(config, node_key)

        for node_key in config.controller_output_keys:
            self.controller_nodes[node_key] = self.create_node(config, node_key)

        if config.encoder_num_hidden > 0:
            for i in range(config.encoder_num_hidden):
                node_key = config.get_new_node_key({**self.encoder_nodes, **self.memory_nodes, **self.controller_nodes})
                assert node_key not in self.encoder_nodes
                node = self.create_node(config, node_key)
                self.encoder_nodes[node_key] = node

        if config.memory_num_hidden > 0:
            for i in range(config.memory_num_hidden):
                node_key = config.get_new_node_key({**self.encoder_nodes, **self.memory_nodes, **self.controller_nodes})
                assert node_key not in self.memory_nodes
                node = self.create_node(config, node_key)
                self.memory_nodes[node_key] = node

        if config.controller_num_hidden > 0:
            for i in range(config.controller_num_hidden):
                node_key = config.get_new_node_key({**self.encoder_nodes, **self.memory_nodes, **self.controller_nodes})
                assert node_key not in self.controller_nodes
                node = self.create_node(config, node_key)
                self.controller_nodes[node_key] = node

        if 'full' in config.initial_connection:
            if config.initial_connection == 'full_nodirect':
                self.connect_full_nodirect(config)
            elif config.initial_connection == 'full_direct':
                self.connect_full_direct(config)
            else:
                if config.encoder_num_hidden > 0 or config.memory_num_hidden > 0 or config.controller_num_hidden > 0:
                    print(
                        "Warning: initial_connection = full with hidden nodes will not do direct input-output connections;",
                        "\tif this is desired, set initial_connection = full_nodirect;",
                        "\tif not, set initial_connection = full_direct",
                        sep='\n', file=sys.stderr)

                self.connect_full_nodirect(config)
        elif 'partial' in config.initial_connection:
            if config.initial_connection == 'partial_nodirect':
                self.connect_partial_nodirect(config)
            elif config.initial_connection == 'partial_direct':
                self.connect_partial_direct(config)
            else:
                if config.encoder_num_hidden > 0 or config.memory_num_hidden > 0 or config.controller_num_hidden > 0:
                    print(
                        "Warning: initial_connection = partial with hidden nodes will not do direct input-output connections;",
                        "\tif this is desired, set initial_connection = partial_nodirect {0};".format(
                            config.connection_fraction),
                        "\tif not, set initial_connection = partial_direct {0}".format(
                            config.connection_fraction),
                        sep='\n', file=sys.stderr)

                self.connect_partial_nodirect(config)

    
    def compute_full_connections(self, config: WorldModelGenomeConfig, direct: bool):
        """
        Compute connections for a fully-connected feed-forward genome--each
        input connected to all hidden nodes
        (and output nodes if ``direct`` is set or there are no hidden nodes),
        each hidden node connected to all output nodes.
        (Recurrent genomes will also include node self-connections.)
        """
        encoder_hidden = [i for i in self.encoder_nodes if i not in config.encoder_output_keys]
        encoder_output = [i for i in self.encoder_nodes if i in config.encoder_output_keys]

        memory_hidden = [i for i in self.memory_nodes if i not in config.memory_output_keys]
        memory_output = [i for i in self.memory_nodes if i in config.memory_output_keys]

        controller_hidden = [i for i in self.controller_nodes if i not in config.controller_output_keys]
        controller_output = [i for i in self.controller_nodes if i in config.controller_output_keys]

        encoder_connections = []
        memory_connections = []
        controller_connections = []

        if encoder_hidden:
            for input_id in config.encoder_input_keys:
                for h in encoder_hidden:
                    encoder_connections.append((input_id, h))
            for h in encoder_hidden:
                for output_id in encoder_output:
                    encoder_connections.append((h, output_id))
        if direct or (not encoder_hidden):
            for input_id in config.encoder_input_keys:
                for output_id in encoder_output:
                    encoder_connections.append((input_id, output_id))

        if memory_hidden:
            for input_id in config.memory_input_keys:
                for h in memory_hidden:
                    memory_connections.append((input_id, h))
            for h in memory_hidden:
                for output_id in memory_output:
                    memory_connections.append((h, output_id))
        if direct or (not memory_hidden):
            for input_id in config.memory_input_keys:
                for output_id in memory_output:
                    memory_connections.append((input_id, output_id))
        # Include node self-connections for recurrent memory module.
        for i in self.memory_nodes:
            memory_connections.append((i, i))

        if controller_hidden:
            for input_id in config.controller_input_keys:
                for h in controller_hidden:
                    controller_connections.append((input_id, h))
            for h in controller_hidden:
                for output_id in controller_output:
                    controller_connections.append((h, output_id))
        if direct or (not controller_hidden):
            for input_id in config.controller_input_keys:
                for output_id in controller_output:
                    controller_connections.append((input_id, output_id))

        return (encoder_connections, memory_connections, controller_connections)


    def connect_full_nodirect(self, config: WorldModelGenomeConfig):
        """
        Create a fully-connected genome
        (except without direct input-output unless no hidden nodes).
        """

        encoder_connections, memory_connections, controller_connections = self.compute_full_connections(config, False)

        for input_id, output_id in encoder_connections:
            connection = self.create_connection(config, input_id, output_id)
            self.encoder_connections[connection.key] = connection

        for input_id, output_id in memory_connections:
            connection = self.create_connection(config, input_id, output_id)
            self.memory_connections[connection.key] = connection

        for input_id, output_id in controller_connections:
            connection = self.create_connection(config, input_id, output_id)
            self.controller_connections[connection.key] = connection


    def connect_full_direct(self, config):
        """ Create a fully-connected genome, including direct input-output connections. """

        encoder_connections, memory_connections, controller_connections = self.compute_full_connections(config, True)

        for input_id, output_id in encoder_connections:
            connection = self.create_connection(config, input_id, output_id)
            self.encoder_connections[connection.key] = connection

        for input_id, output_id in memory_connections:
            connection = self.create_connection(config, input_id, output_id)
            self.memory_connections[connection.key] = connection

        for input_id, output_id in controller_connections:
            connection = self.create_connection(config, input_id, output_id)
            self.controller_connections[connection.key] = connection


    def connect_partial_nodirect(self, config: WorldModelGenomeConfig):
        """
        Create a partially-connected genome,
        with (unless no hidden nodes) no direct input-output connections."""
        assert 0 <= config.connection_fraction <= 1

        encoder_connections, memory_connections, controller_connections = self.compute_full_connections(config, False)

        shuffle(encoder_connections)
        shuffle(memory_connections)
        shuffle(controller_connections)

        num_to_add = int(round(len(encoder_connections) * config.connection_fraction))
        for input_id, output_id in encoder_connections[:num_to_add]:
            connection = self.create_connection(config, input_id, output_id)
            self.encoder_connections[connection.key] = connection

        num_to_add = int(round(len(memory_connections) * config.connection_fraction))
        for input_id, output_id in memory_connections[:num_to_add]:
            connection = self.create_connection(config, input_id, output_id)
            self.memory_connections[connection.key] = connection

        num_to_add = int(round(len(controller_connections) * config.connection_fraction))
        for input_id, output_id in controller_connections[:num_to_add]:
            connection = self.create_connection(config, input_id, output_id)
            self.controller_connections[connection.key] = connection


    def connect_partial_direct(self, config: WorldModelGenomeConfig):
        """
        Create a partially-connected genome,
        including (possibly) direct input-output connections.
        """
        assert 0 <= config.connection_fraction <= 1

        encoder_connections, memory_connections, controller_connections = self.compute_full_connections(config, True)

        shuffle(encoder_connections)
        shuffle(memory_connections)
        shuffle(controller_connections)

        num_to_add = int(round(len(encoder_connections) * config.connection_fraction))
        for input_id, output_id in encoder_connections[:num_to_add]:
            connection = self.create_connection(config, input_id, output_id)
            self.encoder_connections[connection.key] = connection

        num_to_add = int(round(len(memory_connections) * config.connection_fraction))
        for input_id, output_id in memory_connections[:num_to_add]:
            connection = self.create_connection(config, input_id, output_id)
            self.memory_connections[connection.key] = connection

        num_to_add = int(round(len(controller_connections) * config.connection_fraction))
        for input_id, output_id in controller_connections[:num_to_add]:
            connection = self.create_connection(config, input_id, output_id)
            self.controller_connections[connection.key] = connection


    def configure_crossover(self, genome1: WorldModelGenome, genome2: WorldModelGenome, config: WorldModelGenomeConfig):
        """ Configure a new genome by crossover from two parent genomes. """
        if genome1.fitness > genome2.fitness:
            parent1, parent2 = genome1, genome2
        else:
            parent1, parent2 = genome2, genome1

        # Inherit connection genes
        # Encoder
        for key, cg1 in parent1.encoder_connections.items():
            cg2 = parent2.encoder_connections.get(key)
            if cg2 is None:
                # Excess or disjoint gene: copy from the fittest parent.
                self.encoder_connections[key] = cg1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.encoder_connections[key] = cg1.crossover(cg2)

        # Memory
        for key, cg1 in parent1.memory_connections.items():
            cg2 = parent2.memory_connections.get(key)
            if cg2 is None:
                # Excess or disjoint gene: copy from the fittest parent.
                self.memory_connections[key] = cg1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.memory_connections[key] = cg1.crossover(cg2)

        # Controller
        for key, cg1 in parent1.controller_connections.items():
            cg2 = parent2.controller_connections.get(key)
            if cg2 is None:
                # Excess or disjoint gene: copy from the fittest parent.
                self.controller_connections[key] = cg1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.controller_connections[key] = cg1.crossover(cg2)

        # Inherit node genes
        # Encoder
        parent1_set = parent1.encoder_nodes
        parent2_set = parent2.encoder_nodes

        for key, ng1 in parent1_set.items():
            ng2 = parent2_set.get(key)
            assert key not in self.encoder_nodes
            if ng2 is None:
                # Extra gene: copy from the fittest parent
                self.encoder_nodes[key] = ng1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.encoder_nodes[key] = ng1.crossover(ng2)

        # Memory
        parent1_set = parent1.memory_nodes
        parent2_set = parent2.memory_nodes

        for key, ng1 in parent1_set.items():
            ng2 = parent2_set.get(key)
            assert key not in self.memory_nodes
            if ng2 is None:
                # Extra gene: copy from the fittest parent
                self.memory_nodes[key] = ng1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.memory_nodes[key] = ng1.crossover(ng2)

        # Controller
        parent1_set = parent1.controller_nodes
        parent2_set = parent2.controller_nodes

        for key, ng1 in parent1_set.items():
            ng2 = parent2_set.get(key)
            assert key not in self.controller_nodes
            if ng2 is None:
                # Extra gene: copy from the fittest parent
                self.controller_nodes[key] = ng1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.controller_nodes[key] = ng1.crossover(ng2)


    def mutate(self, config: WorldModelGenomeConfig):
        """ Mutates this genome. """

        modules = [
            (self.encoder_nodes, self.encoder_connections, config.encoder_input_keys, config.encoder_output_keys, False),
            (self.memory_nodes, self.memory_connections, config.memory_input_keys, config.memory_output_keys, True),
            (self.controller_nodes, self.controller_connections, config.controller_input_keys, config.controller_output_keys, False)
        ]

        if config.single_structural_mutation:
            div = max(1, (config.node_add_prob + config.node_delete_prob +
                          config.conn_add_prob + config.conn_delete_prob))
            r = random()
            if r < (config.node_add_prob/div):
                self.mutate_add_node(config)
            elif r < ((config.node_add_prob + config.node_delete_prob)/div):
                self.mutate_delete_node(config)
            elif r < ((config.node_add_prob + config.node_delete_prob +
                       config.conn_add_prob)/div):
                self.mutate_add_connection(config)
            elif r < ((config.node_add_prob + config.node_delete_prob +
                       config.conn_add_prob + config.conn_delete_prob)/div):
                self.mutate_delete_connection()
        else:
            if random() < config.node_add_prob:
                self.mutate_add_node(config, choice(modules))

            if random() < config.node_delete_prob:
                self.mutate_delete_node(config, choice(modules))

            if random() < config.conn_add_prob:
                self.mutate_add_connection(config, choice(modules))

            if random() < config.conn_delete_prob:
                self.mutate_delete_connection(choice(modules))

        # Mutate connection genes.
        for cg in self.encoder_connections.values():
            cg.mutate(config)

        for cg in self.memory_connections.values():
            cg.mutate(config)

        for cg in self.controller_connections.values():
            cg.mutate(config)

        # Mutate node genes (bias, response, etc.).
        for ng in self.encoder_nodes.values():
            ng.mutate(config)

        for ng in self.memory_nodes.values():
            ng.mutate(config)

        for ng in self.controller_nodes.values():
            ng.mutate(config)


    def mutate_add_node(self, config: WorldModelGenomeConfig, module: Tuple[dict, dict, dict, dict, bool]):
        module_nodes, module_connections, _, _, _ = module

        if not module_connections:
            if config.check_structural_mutation_surer():
                self.mutate_add_connection(config, module)
            return

        # Choose a random connection to split
        try:
            conn_to_split = choice(list(module_connections.values()))
        except Exception as e:
            print(e)
            print(module)
            raise e
        new_node_id = config.get_new_node_key({**self.encoder_nodes, **self.memory_nodes, **self.controller_nodes})
        ng = self.create_node(config, new_node_id)
        module_nodes[new_node_id] = ng

        # Disable this connection and create two new connections joining its nodes via
        # the given node.  The new node+connections have roughly the same behavior as
        # the original connection (depending on the activation function of the new node).
        conn_to_split.enabled = False

        i, o = conn_to_split.key
        self.add_connection(config, module, i, new_node_id, 1.0, True)
        self.add_connection(config, module, new_node_id, o, conn_to_split.weight, True)


    def add_connection(self, config: WorldModelGenomeConfig, module: Tuple[dict, dict, dict, dict, bool], input_key, output_key, weight, enabled):
        _, module_connections, _, _, _ = module

        # TODO: Add further validation of this connection addition?
        assert isinstance(input_key, int)
        assert isinstance(output_key, int)
        assert output_key >= 0
        assert isinstance(enabled, bool)
        key = (input_key, output_key)
        connection = config.connection_gene_type(key)
        connection.init_attributes(config)
        connection.weight = weight
        connection.enabled = enabled
        module_connections[key] = connection

    
    def mutate_add_connection(self, config: WorldModelGenomeConfig, module: Tuple[dict, dict, dict, dict, bool]):
        """
        Attempt to add a new connection, the only restriction being that the output
        node cannot be one of the network input pins.
        """
        module_nodes, module_connections, module_input_keys, module_output_keys, is_recurrent = module

        possible_outputs = list(module_nodes)
        out_node = choice(possible_outputs)

        possible_inputs = possible_outputs + module_input_keys
        in_node = choice(possible_inputs)

        # Don't duplicate connections.
        key = (in_node, out_node)
        if key in module_connections:
            # TODO: Should this be using mutation to/from rates? Hairy to configure...
            if config.check_structural_mutation_surer():
                module_connections[key].enabled = True
            return

        # Don't allow connections between two output nodes
        if in_node in module_output_keys and out_node in module_output_keys:
            return

        # No need to check for connections between input nodes:
        # they cannot be the output end of a connection (see above).

        # For feed-forward networks, avoid creating cycles.
        if not is_recurrent and creates_cycle(list(module_connections), key):
            return

        cg = self.create_connection(config, in_node, out_node)
        module_connections[cg.key] = cg


    def mutate_delete_node(self, config: WorldModelGenomeConfig, module: Tuple[dict, dict, dict, dict, bool]):
        module_nodes, module_connections, _, module_output_keys, _ = module

        # Do nothing if there are no non-output nodes.
        available_nodes = [k for k in module_nodes if k not in module_output_keys]
        if not available_nodes:
            return -1

        del_key = choice(available_nodes)

        connections_to_delete = set()
        for k, v in module_connections.items():
            if del_key in v.key:
                connections_to_delete.add(v.key)

        for key in connections_to_delete:
            del module_connections[key]

        del module_nodes[del_key]

        return del_key


    def mutate_delete_connection(self, module: Tuple[dict, dict, dict, dict, bool]):
        _, module_connections, _, _, _ = module

        if module_connections:
            key = choice(list(module_connections.keys()))
            del module_connections[key]


    def distance(self, other, config):
        """
        Returns the genetic distance between this genome and the other. This distance value
        is used to compute genome compatibility for speciation.
        """

        # Compute node gene distance component.
        total_node_distance = 0.

        if self.encoder_nodes or other.encoder_nodes:
            node_distance = 0.
            disjoint_nodes = 0

            for k2 in other.encoder_nodes:
                if k2 not in self.encoder_nodes:
                    disjoint_nodes += 1

            for k1, n1 in self.encoder_nodes.items():
                n2 = other.encoder_nodes.get(k1)
                if n2 is None:
                    disjoint_nodes += 1
                else:
                    # Homologous genes compute their own distance value.
                    node_distance += n1.distance(n2, config)

            max_nodes = max(len(self.encoder_nodes), len(other.encoder_nodes))
            total_node_distance += (node_distance +
                                   (config.compatibility_disjoint_coefficient *
                                    disjoint_nodes)) / max_nodes

        if self.memory_nodes or other.memory_nodes:
            node_distance = 0.
            disjoint_nodes = 0

            for k2 in other.memory_nodes:
                if k2 not in self.memory_nodes:
                    disjoint_nodes += 1

            for k1, n1 in self.memory_nodes.items():
                n2 = other.memory_nodes.get(k1)
                if n2 is None:
                    disjoint_nodes += 1
                else:
                    # Homologous genes compute their own distance value.
                    node_distance += n1.distance(n2, config)

            max_nodes = max(len(self.memory_nodes), len(other.memory_nodes))
            total_node_distance += (node_distance +
                                   (config.compatibility_disjoint_coefficient *
                                    disjoint_nodes)) / max_nodes

        if self.controller_nodes or other.controller_nodes:
            node_distance = 0.
            disjoint_nodes = 0

            for k2 in other.controller_nodes:
                if k2 not in self.controller_nodes:
                    disjoint_nodes += 1

            for k1, n1 in self.controller_nodes.items():
                n2 = other.controller_nodes.get(k1)
                if n2 is None:
                    disjoint_nodes += 1
                else:
                    # Homologous genes compute their own distance value.
                    node_distance += n1.distance(n2, config)

            max_nodes = max(len(self.controller_nodes), len(other.controller_nodes))
            total_node_distance += (node_distance +
                                   (config.compatibility_disjoint_coefficient *
                                    disjoint_nodes)) / max_nodes

        # Compute connection gene differences.
        total_connection_distance = 0.0

        if self.encoder_connections or other.encoder_connections:
            connection_distance = 0.
            disjoint_connections = 0

            for k2 in other.encoder_connections:
                if k2 not in self.encoder_connections:
                    disjoint_connections += 1

            for k1, c1 in self.encoder_connections.items():
                c2 = other.encoder_connections.get(k1)
                if c2 is None:
                    disjoint_connections += 1
                else:
                    # Homologous genes compute their own distance value.
                    connection_distance += c1.distance(c2, config)

            max_conn = max(len(self.encoder_connections), len(other.encoder_connections))
            total_connection_distance += (connection_distance +
                                         (config.compatibility_disjoint_coefficient *
                                          disjoint_connections)) / max_conn

        if self.memory_connections or other.memory_connections:
            connection_distance = 0.
            disjoint_connections = 0

            for k2 in other.memory_connections:
                if k2 not in self.memory_connections:
                    disjoint_connections += 1

            for k1, c1 in self.memory_connections.items():
                c2 = other.memory_connections.get(k1)
                if c2 is None:
                    disjoint_connections += 1
                else:
                    # Homologous genes compute their own distance value.
                    connection_distance += c1.distance(c2, config)

            max_conn = max(len(self.memory_connections), len(other.memory_connections))
            total_connection_distance += (connection_distance +
                                         (config.compatibility_disjoint_coefficient *
                                          disjoint_connections)) / max_conn

        if self.controller_connections or other.controller_connections:
            connection_distance = 0.
            disjoint_connections = 0

            for k2 in other.controller_connections:
                if k2 not in self.controller_connections:
                    disjoint_connections += 1

            for k1, c1 in self.controller_connections.items():
                c2 = other.controller_connections.get(k1)
                if c2 is None:
                    disjoint_connections += 1
                else:
                    # Homologous genes compute their own distance value.
                    connection_distance += c1.distance(c2, config)

            max_conn = max(len(self.controller_connections), len(other.controller_connections))
            total_connection_distance += (connection_distance +
                                         (config.compatibility_disjoint_coefficient *
                                          disjoint_connections)) / max_conn

        distance = total_node_distance + total_connection_distance
        return distance


    def size(self):
        """
        Returns genome 'complexity', taken to be
        (number of nodes, number of enabled connections)
        """
        num_enabled_connections = sum([1 for cg in self.encoder_connections.values() if cg.enabled])
        num_enabled_connections += sum([1 for cg in self.memory_connections.values() if cg.enabled])
        num_enabled_connections += sum([1 for cg in self.controller_connections.values() if cg.enabled])
        return len(self.encoder_nodes) + len(self.memory_nodes) + len(self.controller_nodes), num_enabled_connections

    def __str__(self):
        s = "Key: {0}\nFitness: {1}".format(self.key, self.fitness)

        s += "\nEncoder nodes:"
        for k, ng in self.encoder_nodes.items():
            s += "\n\t{0} {1!s}".format(k, ng)
        s += "\nEncoder connections:"
        connections = list(self.encoder_connections.values())
        connections.sort()
        for c in connections:
            s += "\n\t" + str(c)

        s += "\nMemory nodes:"
        for k, ng in self.memory_nodes.items():
            s += "\n\t{0} {1!s}".format(k, ng)
        s += "\nMemory connections:"
        connections = list(self.memory_connections.values())
        connections.sort()
        for c in connections:
            s += "\n\t" + str(c)

        s += "\nController nodes:"
        for k, ng in self.controller_nodes.items():
            s += "\n\t{0} {1!s}".format(k, ng)
        s += "\nController connections:"
        connections = list(self.controller_connections.values())
        connections.sort()
        for c in connections:
            s += "\n\t" + str(c)

        return s