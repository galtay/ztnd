import json
import logging
import os
from pathlib import Path

from openai import OpenAI
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion import ChatCompletion
from pydantic import BaseModel


logger = logging.getLogger(__name__)


def create_completions(
    messages: list[dict[str, str]],
    model: str = "gpt-4o-mini",
    logprobs: bool = True,
    top_logprobs: int = 1,
    max_completion_tokens: int = 64,
    n_choices_per_call: int = 1,
    seed: int = 9237,
    temperature: float = 0.0,
    n_api_calls: int = 1,
) -> list[ChatCompletion]:

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    completions = []
    for ii in range(n_api_calls):
        logger.info(f"n_api_call={ii}")
        completion = client.chat.completions.create(
            messages=messages,
            model=model,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            max_completion_tokens=max_completion_tokens,
            n=n_choices_per_call,
            seed=seed,
            temperature=temperature,
        )
        completions.append(completion)

    return completions


def save_completions(completions: list[ChatCompletion], path: str | Path) -> None:
    path = Path(path)
    dat = [el.dict() for el in completions]
    with path.open("w") as fp:
        fp.write(json.dumps(dat, indent=4))


def load_completions(path: str | Path) -> list[ChatCompletion]:
    path = Path(path)
    with path.open("r") as fp:
        dat = json.load(fp)
    return [ChatCompletion(**el) for el in dat]
