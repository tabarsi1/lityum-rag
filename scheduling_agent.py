from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, List
from dotenv import load_dotenv
import json

load_dotenv()
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Define the state — what the agent remembers between steps
class AgentState(TypedDict):
    messages: List
    jobs: List[dict]
    machines: List[dict]
    schedule: dict
    next_action: str

# Sample job shop data
JOBS = [
    {"id": "J001", "name": "Aluminium Housing", "operations": ["milling", "drilling", "inspection"], "priority": "HIGH", "due_hours": 8},
    {"id": "J002", "name": "Steel Bracket", "operations": ["turning", "milling", "inspection"], "priority": "MEDIUM", "due_hours": 12},
    {"id": "J003", "name": "Copper Connector", "operations": ["drilling", "inspection"], "priority": "LOW", "due_hours": 24},
    {"id": "J004", "name": "Titanium Shaft", "operations": ["turning", "grinding", "inspection"], "priority": "HIGH", "due_hours": 6},
]

MACHINES = [
    {"id": "M001", "name": "CNC Mill DMG MORI", "type": "milling", "available": True, "efficiency": 0.95},
    {"id": "M002", "name": "CNC Lathe Mazak", "type": "turning", "available": True, "efficiency": 0.90},
    {"id": "M003", "name": "Drill Press", "type": "drilling", "available": False, "efficiency": 0.85},
    {"id": "M004", "name": "Surface Grinder", "type": "grinding", "available": True, "efficiency": 0.88},
    {"id": "M005", "name": "CMM Inspection", "type": "inspection", "available": True, "efficiency": 0.99},
]

# Node 1 — analyse the job queue
def analyse_jobs(state: AgentState) -> AgentState:
    print("Agent Step 1: Analysing job queue...")
    jobs_summary = json.dumps(state["jobs"], indent=2)
    response = llm.invoke([
        HumanMessage(content=f"""You are a manufacturing scheduling agent.
Analyse these jobs and identify:
1. Which jobs are most urgent based on priority and due time
2. Which jobs are at risk of missing their deadline
3. Recommended processing order

Jobs:
{jobs_summary}

Respond concisely with your analysis.""")
    ])
    state["messages"].append(AIMessage(content=f"Job Analysis: {response.content}"))
    state["next_action"] = "check_machines"
    print(f"Analysis complete: {response.content[:100]}...")
    return state

# Node 2 — check machine availability
def check_machines(state: AgentState) -> AgentState:
    print("Agent Step 2: Checking machine availability...")
    available = [m for m in state["machines"] if m["available"]]
    unavailable = [m for m in state["machines"] if not m["available"]]
    summary = f"Available: {[m['name'] for m in available]}\nUnavailable: {[m['name'] for m in unavailable]}"
    state["messages"].append(AIMessage(content=f"Machine Status: {summary}"))
    state["next_action"] = "create_schedule"
    print(f"Machines checked: {len(available)} available, {len(unavailable)} unavailable")
    return state

# Node 3 — create optimised schedule
def create_schedule(state: AgentState) -> AgentState:
    print("Agent Step 3: Creating optimised schedule...")
    context = "\n".join([m.content for m in state["messages"]])
    response = llm.invoke([
        HumanMessage(content=f"""Based on this analysis:
{context}

Create an optimised production schedule that:
1. Prioritises HIGH priority jobs first
2. Assigns jobs to available machines only
3. Minimises total completion time
4. Flags any jobs at risk of missing deadlines

Format as a clear schedule with time slots.""")
    ])
    state["schedule"] = {"optimised_schedule": response.content}
    state["messages"].append(AIMessage(content=f"Schedule: {response.content}"))
    state["next_action"] = "end"
    print("Schedule created successfully")
    return state

# Decision function — what to do next
def decide_next(state: AgentState) -> str:
    return state["next_action"]

# Build the agent graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("analyse_jobs", analyse_jobs)
workflow.add_node("check_machines", check_machines)
workflow.add_node("create_schedule", create_schedule)

# Add edges
workflow.set_entry_point("analyse_jobs")
workflow.add_conditional_edges(
    "analyse_jobs",
    decide_next,
    {"check_machines": "check_machines"}
)
workflow.add_conditional_edges(
    "check_machines",
    decide_next,
    {"create_schedule": "create_schedule"}
)
workflow.add_conditional_edges(
    "create_schedule",
    decide_next,
    {"end": END}
)

# Compile and run
agent = workflow.compile()

if __name__ == "__main__":
    print("Starting AI Scheduling Agent...")
    print("=" * 50)

    initial_state = {
        "messages": [],
        "jobs": JOBS,
        "machines": MACHINES,
        "schedule": {},
        "next_action": "analyse_jobs"
    }

    result = agent.invoke(initial_state)

    print("\n" + "=" * 50)
    print("FINAL SCHEDULE:")
    print("=" * 50)
    print(result["schedule"]["optimised_schedule"])