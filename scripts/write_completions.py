from datetime import datetime
from pathlib import Path

from ztnd.generations import (
    get_completions,
    get_ztnd_choices_from_completions,
    save_completions,
    save_ztnd_choices,
)

prompt = """Write a short story starting with "Once upon a time"."""
model_name = "gpt-4o-mini"
temperature = 0.4
n_api_calls = 50
n_choices_per_call = 10

completions = get_completions(
    prompt,
    model_name,
    temperature,
    n_api_calls=n_api_calls,
    n_choices_per_call=n_choices_per_call,
)

cache_path = Path("cache") / datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
cache_path.mkdir(exist_ok=True, parents=True)
save_completions(completions, cache_path / "completions.json")
