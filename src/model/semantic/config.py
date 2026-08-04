MODEL_PATH = "src/model/semantic/unsloth_Qwen2.5-3B-Instruct"
MODEL = "unsloth/Qwen2.5-3B-Instruct"
SYSTEM_PROMPT_PATH = "src/model/semantic/prompts/sys_prompt.txt"
FEW_SHOTS_PATH = "src/model/semantic/prompts/few_shots_examples.json"



MAX_SEQ_LENGTH = 2048

# Params de generación 
MAX_NEW_TOKENS = 64
TEMPERATURE = 0.1
REPETITION_PENALTY = 1.2
LOAD_IN_4BIT = True

USE_UNSLOTH = False  # True: usar modelo UnsloTH, False: usar modelo Qwen2.5-3B-Instruct