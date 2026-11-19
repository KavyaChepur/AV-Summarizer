"""Translator agent: translates the final outputs into the chosen language."""
from langchain_core.prompts import ChatPromptTemplate

from agents.common import emit, options
from llm import client

NAME = "translator"

TRANSLATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are the Translator Agent. Translate Markdown study notes into {language}. "
     "Keep the Markdown structure, numbering and every [MM:SS] timestamp exactly as "
     "they are. Keep technical terms, names and product names in their usual form. "
     "Return only the translation."),
    ("human", "{text}"),
])


def translate(state) -> dict:
    opts = options(state)
    language = opts["language"]
    keys = [k for k in opts["outputs"] if state.get(k)]
    emit(NAME, f"Translating {len(keys)} output(s) into {language}…")
    texts = client.ask_many(
        [TRANSLATE_PROMPT.format_messages(language=language, text=state[k]) for k in keys],
        temperature=0.1)
    return {"final_outputs": dict(zip(keys, texts)),
            "log": [f"Translated {len(keys)} output(s) into {language}."]}
