from argparse import ArgumentParser
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import add_messages, END, START, StateGraph
from nltk.tokenize import sent_tokenize
from pydantic import BaseModel, Field

parser = ArgumentParser()
parser.add_argument("FILE", type=str, help="Path to the text file to be translated.")

args = parser.parse_args()

load_dotenv()

LLM = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", api_key=os.getenv("GOOGLE_API_KEY"), temperature=0.7
)


class TranslatedSegment(BaseModel):
    target: str = Field(..., description="The translated segment in Portuguese.")
    glossary: dict = Field(
        ..., description="A glossary of terms used in the translation."
    )


class State(TypedDict):
    messages: Annotated[list, add_messages]
    glossary: dict
    source_segments: list
    target_segments: list


def translate(state):
    source_segments = state["source_segments"]
    glossary = state["glossary"]

    translation_llm = LLM.with_structured_output(TranslatedSegment)

    translated_segments = list()
    prompt = f"""
        You are an expert translator from English to Portuguese.
        Please translate the following segment of sentences.
        If there are any terms that require a glossary entry, please include them in the glossary section of your response.
        The current glossary entries are: {glossary}
    """
    response = translation_llm.invoke(
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "".join(source_segments)},
        ]
    )
    translated_segments.append(response.target)
    glossary = {**glossary, **response.glossary}
    return {"target_segments": translated_segments, "glossary": glossary}


graph_builder = StateGraph(State)


def build_graph():
    graph_builder.add_node("translate", translate)
    graph_builder.add_edge(START, "translate")
    graph_builder.add_edge("translate", END)
    return graph_builder.compile()


if __name__ == "__main__":
    graph = build_graph()
    with open(args.FILE, "r") as f:
        text = f.read()

        sentences = sent_tokenize(text)
    print("<?xml version='1.0' encoding='UTF-8'?>")
    print("<text>")
    glossary = dict()
    for i in range(0, len(sentences), 5):
        batch = sentences[i : i + 5]
        print("<segment>")
        print(f"<source>{''.join(batch)}</source>")
        state = graph.invoke(
            {"source_segments": batch, "glossary": glossary},
            config={"configurable": {"thread_id": "1"}},
        )
        print(f"<target>{''.join(state['target_segments'])}</target>")
        print(f"<glossary>{state['glossary']}</glossary>")
        glossary = state["glossary"]
        print("</segment>")
    print("</text>")
