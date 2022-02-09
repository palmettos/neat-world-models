from neat import Config
from neat.genome import DefaultGenomeConfig
from neat.nn import FeedForwardNetwork, RecurrentNetwork
from .adapters import DefaultGenomeAdapter, DefaultGenomeConfigAdapter, ConfigAdapter
from ..genome.world_model_genome import WorldModelGenome
from typing import List


class WorldModelNetwork(object):
    def __init__(self, encoder: FeedForwardNetwork, memory: RecurrentNetwork, controller: FeedForwardNetwork):
        self.encoder = encoder
        self.memory = memory
        self.controller = controller

        self.memory_state = [0. for _ in range(len(memory.output_nodes))]


    def activate(self, x: List[float]) -> List[float]:
        encoder_out = self.encoder.activate(x)
        controller_out = self.controller.activate(encoder_out + self.memory_state)
        self.memory_state = self.memory.activate(controller_out + encoder_out + self.memory_state)

        return encoder_out, controller_out, self.memory_state


    def reset(self):
        self.memory.reset()


    @staticmethod
    def create(genome: WorldModelGenome, config: Config):
        # Encoder
        #   Create encoder adapters
        encoder_genome_adapter = DefaultGenomeAdapter(genome.encoder_nodes, genome.encoder_connections)
        encoder_genome_config_adapter = DefaultGenomeConfigAdapter(
            config.genome_config.encoder_input_keys,
            config.genome_config.encoder_output_keys,
            config.genome_config.aggregation_function_defs,
            config.genome_config.activation_defs
        )
        encoder_config_adapter = ConfigAdapter(encoder_genome_config_adapter)
        #   Create encoder NN
        encoder = FeedForwardNetwork.create(encoder_genome_adapter, encoder_config_adapter)

        # Memory
        #   Create memory adapters
        memory_genome_adapter = DefaultGenomeAdapter(genome.memory_nodes, genome.memory_connections)
        memory_genome_config_adapter = DefaultGenomeConfigAdapter(
            config.genome_config.memory_input_keys,
            config.genome_config.memory_output_keys,
            config.genome_config.aggregation_function_defs,
            config.genome_config.activation_defs
        )
        memory_config_adapter = ConfigAdapter(memory_genome_config_adapter)
        #    Create memory RNN
        memory = RecurrentNetwork.create(memory_genome_adapter, memory_config_adapter)

        # Controller
        #    Create controller adapters
        controller_genome_adapter = DefaultGenomeAdapter(genome.controller_nodes, genome.controller_connections)
        controller_genome_config_adapter = DefaultGenomeConfigAdapter(
            config.genome_config.controller_input_keys,
            config.genome_config.controller_output_keys,
            config.genome_config.aggregation_function_defs,
            config.genome_config.activation_defs
        )
        controller_config_adapter = ConfigAdapter(controller_genome_config_adapter)
        #   Create controller NN
        controller = FeedForwardNetwork.create(controller_genome_adapter, controller_config_adapter)

        return WorldModelNetwork(encoder, memory, controller)