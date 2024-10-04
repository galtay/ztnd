from ztnd.generations import (
    get_completions,
    get_ztnd_choices_from_completions,
    save_completions,
    save_ztnd_choices,
)

prompt = """Write a short story starting with "Once upon a time"."""
model_name = "gpt-4o-mini"
temperature = 0.4
n_api_calls = 2
n_choices_per_call = 3

completions = get_completions(
    prompt,
    model_name,
    temperature,
    n_api_calls=n_api_calls,
    n_choices_per_call=n_choices_per_call,
)
save_completions(completions, "completions.json")
ztnd_choices = get_ztnd_choices_from_completions(completions)
save_ztnd_choices(ztnd_choices, "ztnd_choices.json")
