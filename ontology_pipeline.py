from argparse import ArgumentParser
import os
import re
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import add_messages, END, START, StateGraph

# from langgraph.checkpoint.memory import InMemorySaver
from nltk.tokenize import sent_tokenize
from pydantic import BaseModel, Field
from rdflib import Graph
from yaml import safe_load

with open("prompts.yaml", "r") as f:
    PROMPTS = safe_load(f)

parser = ArgumentParser()
parser.add_argument("FILE", type=str, help="Path to the text file to be translated.")
args = parser.parse_args()

load_dotenv()

parser = ArgumentParser()
parser.add_argument("FILE", type=str, help="Path to the text file to be translated.")

LLM = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
)

G = Graph()
G.parse("ontology.ttl", format="ttl")


class Extraction(BaseModel):
    entities: list[str] = Field(
        default_factory=list, description="List of entities detected in the text."
    )


class TranslatedSegment(BaseModel):
    target: str = Field(..., description="The translated segment in Portuguese.")


class State(TypedDict):
    messages: Annotated[list, add_messages]
    entities: list[str]
    target_segment: str
    graph: Graph


def extract(state):

    def to_pascal_case(s):
        words = re.findall(r"\w+", s)
        return "".join(word.capitalize() for word in words)

    extraction_llm = LLM.with_structured_output(Extraction)

    messages = [
        {
            "role": "system",
            "content": PROMPTS["extraction"],
        },
        {
            "role": "user",
            "content": state["messages"][-1].content,
        },
    ]

    extraction = extraction_llm.invoke(messages)

    normalized_entities = [to_pascal_case(e) for e in extraction.entities]
    print(f"<entities>{normalized_entities}</entities>")

    return {"entities": normalized_entities}


def translate(state):
    entities = state["entities"]
    graph = state["graph"]

    entity_iris = ", ".join(f":{e}" for e in entities)
    query = f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX : <http://example.org/newman#>

    SELECT ?concept ?aspect ?label_en ?label_pt
    WHERE {{

    ?aspect :isSemanticAspectOf ?concept .

    FILTER (
        ?concept IN ({entity_iris}) ||
        ?aspect IN ({entity_iris})
    )

    OPTIONAL {{ ?aspect rdfs:label ?label_en FILTER(lang(?label_en) = "en") }}
    OPTIONAL {{ ?aspect rdfs:label ?label_pt FILTER(lang(?label_pt) = "pt") }}
    }}
    """

    result = graph.query(query)
    result_graph = [
        {
            "concept": str(row["concept"]),
            "aspect": str(row["aspect"]),
            "label_en": str(row["label_en"]),
            "label_pt": str(row["label_pt"]),
        }
        for row in result
    ]

    print(f"<query>{result_graph}</query>")

    translation_llm = LLM.with_structured_output(TranslatedSegment)

    response = translation_llm.invoke(
        [
            {
                "role": "system",
                "content": PROMPTS["translation"].format(result_graph=result_graph),
            },
            {"role": "user", "content": state["messages"][-1].content},
        ]
    )
    return {"target_segment": response.target}


graph_builder = StateGraph(State)


def build_graph():
    graph_builder.add_node("extract", extract)
    graph_builder.add_node("translate", translate)
    graph_builder.add_edge(START, "extract")
    graph_builder.add_edge("extract", "translate")
    graph_builder.add_edge("translate", END)
    return graph_builder.compile()


if __name__ == "__main__":
    with open(args.FILE, "r") as f:
        text = f.read()

    sentences = sent_tokenize(text)

    graph = build_graph()

    print("<text>")
    for i in range(0, len(sentences), 5):
        print("<segment>")
        batch = sentences[i : i + 5]
        print(f"<source>{batch}</source>")
        state = graph.invoke(
            {
                "messages": [{"role": "user", "content": batch}],
                "graph": G,
            }
        )
        print(f"<target>{state['target_segment']}</target>")
        print("</segment>")
    print("</text>")
