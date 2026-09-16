import logging
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from app.core.config import get_settings
from app.rag.state import AgentState, RouteDecision, EvidenceGrade
from app.rag.vector_store import get_retriever

logger = logging.getLogger(__name__)
settings = get_settings()


_llm = None
_web_search = None


# get the openai LLM object
def llm():
    global _llm
    if _llm is None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is missing")
        _llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0,
            api_key=settings.openai_api_key,
        )
    return _llm


# get the Tavily web search onject
def web_search_tool():
    global _web_search
    if _web_search is None:
        if not settings.tavily_api_key:
            raise RuntimeError("TAVILY_API_KEY is missing")
        _web_search = TavilySearch(
            tavily_api_key=settings.tavily_api_key,
            max_results=5,
            topic="general",
            include_answer=True,
            include_raw_content=False,
        )
    return _web_search


# add trace message to the main state trace list
def add_trace(state: AgentState, message: str):
    return [*state.get("trace", []), message]


# build the token usage state update for a single llm response
def token_usage(node: str, response) -> dict:
    meta = getattr(response, "usage_metadata", None) or {}
    input_tokens = int(meta.get("input_tokens") or 0)
    output_tokens = int(meta.get("output_tokens") or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "llm_calls": 1,
        "token_usage_by_node": [
            {"node": node, "input_tokens": input_tokens, "output_tokens": output_tokens}
        ],
    }


# plain llm call, returns the response message plus its token usage update
def invoke_llm(node: str, prompt: str):
    response = llm().invoke(prompt)
    return response, token_usage(node, response)


# structured llm call, keeps the raw message so token usage stays available
def invoke_structured_llm(node: str, schema, prompt: str):
    response = llm().with_structured_output(schema, method="json_mode", include_raw=True).invoke(prompt)
    parsed = response.get("parsed")
    if parsed is None:
        raise RuntimeError(f"{node}: structured output failed → {response.get('parsing_error')}")
    return parsed, token_usage(node, response.get("raw"))


# route the question to: kb or direct answer and retrun the route in source_used and update the trace 
def route_question(state: AgentState):
    decision, usage = invoke_structured_llm("route_question", RouteDecision, f"""
You route messages for an enterprise company customer support assistant.
Use kb for company related questions and support.
Use direct only for greetings, thanks, or casual chat that needs no company knowledge.
Question: {state['question']}
Return valid JSON like {{"route":"kb"}}.
""")
    return {
        "source_used": decision.route,
        "trace": add_trace(state, f"Router → {decision.route.upper()}"),
        **usage,
    }


# route the node based on the source_used in the state
def route_after_router(state: AgentState) -> Literal["retrieve_kb", "direct_answer"]:
    return "retrieve_kb" if state["source_used"] == "kb" else "direct_answer"


# retrieve the relevant documents (chunks) from the knowledge base (pipecone vectorstore)
def retrieve_kb(state: AgentState):
    docs = get_retriever().invoke(state["current_query"])
    return {"kb_docs": docs, "trace": add_trace(state, f"Private KB retrieval → {len(docs)} chunks")}


# grade documents/chunks are good or weak for answering the question 
def grade_kb(state: AgentState):
    context = "\n\n".join(f"Source: {d.metadata.get('source','unknown')}\n{d.page_content}" for d in state["kb_docs"])
    grade, usage = invoke_structured_llm("grade_kb", EvidenceGrade, f"""
You grade evidence for an enterprise customer support assistant.
Question: {state['question']}
Private company knowledge base evidence:\n{context}
Return good only if the evidence is sufficient to answer confidently and specifically.
Otherwise return weak. JSON: {{"grade":"good"}} or {{"grade":"weak"}}.
""")
    return {
        "kb_grade": grade.grade,
        "trace": add_trace(state, f"KB evidence grade → {grade.grade.upper()}"),
        **usage,
    }



# conditional routing if the kb good, the generate from kb else search web
def after_kb(state: AgentState) -> Literal["generate_from_kb", "search_web"]:
    return "generate_from_kb" if state["kb_grade"] == "good" else "search_web"


# web search result based on the user query
def search_web(state: AgentState):
    result = web_search_tool().invoke({"query": f"for daraz, {state["current_query"]}"})
    lines, citations = [], []
    if isinstance(result, dict):
        if result.get("answer"):
            lines.append("Search answer: " + result["answer"])
        for item in result.get("results", []):
            title, url, content = item.get("title", ""), item.get("url", ""), item.get("content", "")
            lines.append(f"Title: {title}\nURL: {url}\nContent: {content}")
            citations.append({"title": title or url, "url": url, "type": "web"})
    else:
        lines.append(str(result))
    return {
        "web_results": "\n\n".join(lines),
        "citations": citations,
        "source_used": "web",
        "trace": add_trace(state, "Web fallback → Tavily search"),
    }


# grade web good or weak 
def grade_web(state: AgentState):
    grade, usage = invoke_structured_llm("grade_web", EvidenceGrade, f"""
Question: {state['question']}
Web evidence:\n{state['web_results']}
Return good if the evidence is sufficient and directly relevant; otherwise weak.
Return valid JSON like {{"grade":"good"}}.
""")

    return {
        "web_grade": grade.grade,
        "trace": add_trace(state, f"Web evidence grade → {grade.grade.upper()}"),
        **usage,
    }


# routing the the node based on grade web
def after_web(state: AgentState) -> Literal["generate_from_web", "rewrite_query", "insufficient"]:
    if state["web_grade"] == "good":
        return "generate_from_web"
    if state["retry_count"] < settings.max_retries:
        return "rewrite_query"
    return "insufficient"



# rewrite the query or question for beter retriever or web search
def rewrite_query(state: AgentState):
    response, usage = invoke_llm("rewrite_query", f"""
Rewrite this customer-support question for better private knowledge base retrieval and public web search.
Preserve intent, add useful support keywords, do not answer, return only the query.
Question: {state['question']}
""")
    rewritten = response.content.strip()
    return {
        "current_query": rewritten,
        "retry_count": state["retry_count"] + 1,
        "trace": add_trace(state, f"Query rewrite → {rewritten}"),
        **usage,
    }


# generate answer with llm from knowledge base
def generate_from_kb(state: AgentState):
    context = "\n\n".join(f"[Source: {d.metadata.get('source','unknown')}]\n{d.page_content}" for d in state["kb_docs"])
    response, usage = invoke_llm("generate_from_kb", f"""
You are an enterprise HR policy and employee support copilot. Answer ONLY from the private company HR KB below.
Be concise, practical, respectful, and policy-grounded. If steps are present, present them clearly.
Do not invent policy details. Mention that the answer is based on the company's private knowledge base.
Question: {state['question']}\n\nPrivate KB:\n{context}
""")
    answer = response.content
    citations = []
    seen = set()
    for d in state["kb_docs"]:
        src = d.metadata.get("source", "Private KB")
        if src not in seen:
            seen.add(src)
            citations.append({"title": src.split("/")[-1], "url": "", "type": "private_kb"})
    return {
        "answer": answer,
        "source_used": "private_kb",
        "citations": citations,
        "trace": add_trace(state, "Answer generation → PRIVATE KB"),
        **usage,
    }


# generate answer with llm based on web search results and question
def generate_from_web(state: AgentState):
    response, usage = invoke_llm("generate_from_web", f"""
You are an enterprise customer support copilot. The private company Knowledge base was insufficient.
Answer ONLY from the web evidence below. Clearly say this is external public information which you have answered.
Question: {state['question']}\n\nWeb evidence:\n{state['web_results']}
""")
    return {
        "answer": response.content,
        "source_used": "web_search",
        "trace": add_trace(state, "Answer generation → WEB SEARCH"),
        **usage,
    }


# generate answer for normal conversations
def direct_answer(state: AgentState):
    response, usage = invoke_llm("direct_answer", f"Respond briefly and naturally to: {state['question']}")
    return {
        "answer": response.content,
        "source_used": "direct",
        "trace": add_trace(state, "Direct response → no retrieval"),
        **usage,
    }


# if no answer can be found from all the sources
def insufficient(state: AgentState):
    return {
        "answer": "I couldn't find enough reliable evidence in the company knowledge base or external search to answer confidently. Please contact the support team or provide more details.",
        "source_used": "insufficient_evidence",
        "trace": add_trace(state, "Stopped → insufficient reliable evidence"),
    }



# Main Workflow
def build_graph():
    graph = StateGraph(AgentState)
    # all nodes
    for name, fn in {
        "route_question": route_question,
        "retrieve_kb": retrieve_kb,
        "grade_kb": grade_kb,
        "search_web": search_web,
        "grade_web": grade_web,
        "rewrite_query": rewrite_query,
        "generate_from_kb": generate_from_kb,
        "generate_from_web": generate_from_web,
        "direct_answer": direct_answer,
        "insufficient": insufficient,
    }.items():
        graph.add_node(name, fn)

    # all edges
    graph.add_edge(START, "route_question")
    graph.add_conditional_edges("route_question", route_after_router, {
        "retrieve_kb": "retrieve_kb", "direct_answer": "direct_answer"
    })
    graph.add_edge("retrieve_kb", "grade_kb")
    graph.add_conditional_edges("grade_kb", after_kb, {
        "generate_from_kb": "generate_from_kb", "search_web": "search_web"
    })
    graph.add_edge("search_web", "grade_web")
    graph.add_conditional_edges("grade_web", after_web, {
        "generate_from_web": "generate_from_web", "rewrite_query": "rewrite_query", "insufficient": "insufficient"
    })
    graph.add_edge("rewrite_query", "retrieve_kb")
    graph.add_edge("generate_from_kb", END)
    graph.add_edge("generate_from_web", END)
    graph.add_edge("direct_answer", END)
    graph.add_edge("insufficient", END)
    return graph.compile()


# build the langgraph state graph
agent_graph = build_graph()


# function to invoke the graph workflow
def ask(question: str):
    initial: AgentState = {
        "question": question,
        "current_query": question,
        "kb_docs": [],
        "web_results": "",
        "kb_grade": "",
        "web_grade": "",
        "answer": "",
        "source_used": "",
        "retry_count": 0,
        "trace": [],
        "citations": [],
        "input_tokens": 0,
        "output_tokens": 0,
        "llm_calls": 0,
        "token_usage_by_node": [],
    }
    result = agent_graph.invoke(initial)

    
    total_in = result.get("input_tokens", 0)
    total_out = result.get("output_tokens", 0)
    result["trace"] = [
        *result.get("trace", []),
        f"Token usage → {result.get('llm_calls', 0)} LLM calls, "
        f"input {total_in}, output {total_out}, total {total_in + total_out}",
    ]
    return result