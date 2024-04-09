import os

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()

    config.result_prefix = 'results/individual_llama2'

    config.tokenizer_paths=["/share/models/llama2/hf/7B-chat"]
    config.model_paths=["/share/models/llama2/hf/7B-chat"]
    config.conversation_templates=['llama-2']

    return config