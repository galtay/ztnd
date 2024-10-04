import json
from pathlib import Path


from ztnd.generations import (
    load_completions,
    get_ztnd_choices_from_completions,
    save_ztnd_choices,
)

cache_path = Path("cache") / "2024-10-03-21-41-11"
completions = load_completions(cache_path / "completions.json")
ztnd_choices = get_ztnd_choices_from_completions(completions)
save_ztnd_choices(ztnd_choices, cache_path / "ztnd_choices.json")
