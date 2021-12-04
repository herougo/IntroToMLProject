from ml.utils.config import process_command_line_and_get_config
from ml.experiments.common.core import MemoryNetAgent
from ml.utils.log import get_stream_debug_logger

logger = get_stream_debug_logger('Memory_Network_bAbI_13')


def main():
    config = process_command_line_and_get_config()
    logger.info(f'Config: {config}')
    agent = MemoryNetAgent(config, logger, dir_include_date=True)
    agent.setup()
    agent.train()
    print('Running on Test Data')
    agent.model.use_interpretability = True
    print(agent.test())


if __name__ == '__main__':
    main()