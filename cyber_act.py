from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    ToolMessage,
)
from operator import add as add_messages
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from pypdf import PdfReader

load_dotenv()

pdf_path = "act_2018.pdf"
txt_path = "act_2025.txt"
persist_directory = "./vector_db/"
collection_name = "cybercrime_laws"


# 2025 ammendment
if not os.path.exists(txt_path):
    raise FileNotFoundError(f"File not found: {txt_path}")

with open(txt_path, "r", encoding="utf-8") as f:
    amendment_text = f.read().strip()
if not amendment_text:
    raise ValueError(f"File is empty: {txt_path}")

# 2018 principal act
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"File not found: {pdf_path}")

reader = PdfReader(pdf_path)
pdf_text = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        pdf_text += text + "\n"

if not pdf_text.strip():
    raise ValueError(f"No text extracted from {pdf_path}")

combined_text = f"--- Source: Computer Misuse and Cybercrimes Act, 2018 ---\n{pdf_text}\n\n--- Source: Amendment Act, 2025 ---\n{amendment_text}"

# creat chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_text(combined_text)

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

# llm + embeddings
llm = ChatOpenAI(model="gpt-4o", temperature=0)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
)

try:
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    print(f"✅ Created ChromaDB vector store with {len(chunks)} chunks!")
except Exception as e:
    raise RuntimeError(f"Error setting up ChromaDB: {str(e)}")

# retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})


@tool
def retriever_tool(query: str) -> str:
    """
    This tool searches and returns information from the
    Computer Misuse and Cybercrimes Acts (2018 & 2025 Amendment).
    """
    docs = retriever.invoke(query)

    if not docs:
        return "I found no relevant information in the Computer Misuse and Cybercrimes Acts (2018 & 2025 Amendment)."

    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")

    return "\n\n".join(results)


tools = [retriever_tool]
llm = llm.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def should_continue(state: AgentState):
    """Check if the last message contains tool calls."""
    result = state["messages"][-1]
    return hasattr(result, "tool_calls") and len(result.tool_calls) > 0


system_prompt = """
You are a legal research assistant specializing in Kenyan cybersecurity legislation. Your task is to answer questions accurately using information from both The Computer Misuse and Cybercrimes Act, 2018 and The Computer Misuse and Cybercrimes (Amendment) Act, 2025. Always use the retriever_tool to locate and cite relevant sections, and make as many retrieval calls as needed. If no relevant information is found, respond with “I found no relevant information in the Computer Misuse and Cybercrimes Acts (2018 & 2025 Amendment).”

Base every answer strictly on the Acts’ text. When explaining amendments, clearly identify which Act or section the information comes from, describe what change was made, and summarize its effect or intent. Maintain a factual, legal tone and cite retrieved content precisely (e.g., “According to Section 46A of the 2025 Amendment…”). Never speculate or include information not supported by the documents.
"""

tools_dict = {our_tool.name: our_tool for our_tool in tools}


def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    messages = list(state["messages"])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = llm.invoke(messages)
    return {"messages": [message]}


# retriever
def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""

    tool_calls = state["messages"][-1].tool_calls
    results = []
    for t in tool_calls:
        print(
            f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}"
        )

        if not t["name"] in tools_dict:  # Checks if a valid tool is present
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."

        else:
            result = tools_dict[t["name"]].invoke(t["args"].get("query", ""))
            print(f"Result length: {len(str(result))}")

        # Appends the Tool Message
        results.append(
            ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result))
        )

    print("Tools Execution Complete. Back to the model!")
    return {"messages": results}


graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges(
    "llm", should_continue, {True: "retriever_agent", False: END}
)
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()


def running_agent():
    print("\n=== RAG AGENT===")

    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        messages = [
            HumanMessage(content=user_input)
        ]  # converts back to a HumanMessage type

        result = rag_agent.invoke({"messages": messages})

        print("\n=== ANSWER ===")
        print(result["messages"][-1].content)


if __name__ == "__main__":
    running_agent()
