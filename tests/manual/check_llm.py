from src.models.llm_decider import ask
from src.query.validation import load_settings

cfg = load_settings()
print("model:", cfg["llm"]["model"])
print(ask("Reply with the single word: OK", cfg))