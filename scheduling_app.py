import streamlit as st
import json
from scheduling_agent import agent
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="AI Scheduling Agent", layout="wide")
st.markdown("### Lityum Engineering — AI Scheduling Agent")
st.caption("Autonomous production scheduling powered by AI")
st.divider()

# Default sample data
DEFAULT_JOBS = [
    {"id": "J001", "name": "Aluminium Housing", "operations": ["milling", "drilling", "inspection"], "priority": "HIGH", "due_hours": 8},
    {"id": "J002", "name": "Steel Bracket", "operations": ["turning", "milling", "inspection"], "priority": "MEDIUM", "due_hours": 12},
    {"id": "J003", "name": "Copper Connector", "operations": ["drilling", "inspection"], "priority": "LOW", "due_hours": 24},
    {"id": "J004", "name": "Titanium Shaft", "operations": ["turning", "grinding", "inspection"], "priority": "HIGH", "due_hours": 6},
]

DEFAULT_MACHINES = [
    {"id": "M001", "name": "CNC Mill DMG MORI", "type": "milling", "available": True, "efficiency": 0.95},
    {"id": "M002", "name": "CNC Lathe Mazak", "type": "turning", "available": True, "efficiency": 0.90},
    {"id": "M003", "name": "Drill Press", "type": "drilling", "available": False, "efficiency": 0.85},
    {"id": "M004", "name": "Surface Grinder", "type": "grinding", "available": True, "efficiency": 0.88},
    {"id": "M005", "name": "CMM Inspection", "type": "inspection", "available": True, "efficiency": 0.99},
]

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Job Queue")
    jobs_input = st.text_area(
        "Edit jobs (JSON format)",
        value=json.dumps(DEFAULT_JOBS, indent=2),
        height=300
    )

with col2:
    st.subheader("Machine Status")
    machines_input = st.text_area(
        "Edit machines (JSON format)",
        value=json.dumps(DEFAULT_MACHINES, indent=2),
        height=300
    )

st.divider()

if st.button("Run Scheduling Agent", type="primary"):
    try:
        jobs = json.loads(jobs_input)
        machines = json.loads(machines_input)
    except json.JSONDecodeError:
        st.error("Invalid JSON in jobs or machines. Please check the format.")
        st.stop()

    with st.spinner("Agent is analysing jobs and optimising schedule..."):
        initial_state = {
            "messages": [],
            "jobs": jobs,
            "machines": machines,
            "schedule": {},
            "next_action": "analyse_jobs"
        }

        result = agent.invoke(initial_state)

    st.success("Schedule generated successfully!")
    st.divider()

    # Show agent reasoning steps
    st.subheader("Agent Reasoning")
    for i, msg in enumerate(result["messages"]):
        with st.expander(f"Step {i+1} — {msg.content[:50]}..."):
            st.write(msg.content)

    # Show final schedule
    st.subheader("Optimised Production Schedule")
    st.markdown(result["schedule"]["optimised_schedule"])

    # Show risk summary
    high_priority = [j for j in jobs if j["priority"] == "HIGH"]
    at_risk = [j for j in jobs if j["due_hours"] <= 8]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Jobs", len(jobs))
    c2.metric("High Priority", len(high_priority))
    c3.metric("At Risk", len(at_risk))