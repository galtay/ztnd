import json
import os
from pathlib import Path

from openai import OpenAI
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion import ChatCompletion
from pydantic import BaseModel


class ZtndToken(BaseModel, frozen=True):
    text: str
    ipos: int
    logprob: float

    def get_node(self) -> tuple[str, int]:
        return (self.text, self.ipos)


class ZtndChoice(BaseModel, frozen=True):
    completion_id: str
    choice_index: int
    choice_id: str
    tokens: list[ZtndToken]

    def __len__(self):
        return len(self.tokens)


def generate_completion(
    model_name: str,
    prompt: str,
    temperature: float = 0.0,
    n: int = 1,
    seed: int = 9237,
) -> ChatCompletion:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model_name,
        temperature=temperature,
        logprobs=True,
        top_logprobs=1,
        max_completion_tokens=64,
        n=n,
        seed=seed,
    )


def get_ztnd_choices_from_completions(
    completions: list[ChatCompletion],
) -> list[ZtndChoice]:
    output = []
    for completion in completions:
        for choice_index, choice in enumerate(completion.choices):
            ztnd_choice = ZtndChoice(
                completion_id=completion.id,
                choice_index=choice_index,
                choice_id=f"{completion.id}-{choice_index}",
                tokens=[
                    ZtndToken(
                        text=clt.token,
                        ipos=ipos,
                        logprob=clt.logprob,
                    )
                    for ipos, clt in enumerate(choice.logprobs.content)
                ],
            )
            output.append(ztnd_choice)
    return output


def get_completions(
    prompt: str,
    model_name: str,
    temperature: float,
    n_api_calls: int = 10,
    n_choices_per_call: int = 8,
) -> list[ChatCompletion]:

    completions = []
    for ii in range(n_api_calls):
        completion = generate_completion(
            model_name,
            prompt,
            temperature=temperature,
            n=n_choices_per_call,
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


def save_ztnd_choices(ztnd_choices: list[ZtndChoice], path: str | Path) -> None:
    path = Path(path)
    dat = [el.dict() for el in ztnd_choices]
    with path.open("w") as fp:
        fp.write(json.dumps(dat, indent=4))


def load_ztnd_choices(path: str | Path) -> list[ZtndChoices]:
    path = Path(path)
    with path.open("r") as fp:
        dat = json.load(fp)
    return [ZtndChoice(**el) for el in dat]
