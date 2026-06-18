import streamlit as st
import pandas as pd
import tempfile, os, json
from fmea_rag import generate_fmea_with_context
from fmea_export import export_fmea_to_excel
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="FMEA Generator", layout="wide")
st.markdown("### Lityum Engineering — FMEA Generator")
st.caption("AI-assisted Process FMEA generation grounded in your documents")
st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    process_name = st.text_input("Process name",
        placeholder="e.g. CNC Milling — Aluminium Housing")
    machine = st.text_input("Machine / Equipment",
        placeholder="e.g. DMG MORI DMU 50")
    material = st.text_input("Material",
        placeholder="e.g. Aluminium 6061-T6")

with col2:
    steps_input = st.text_area("Process steps (one per line)",
        placeholder="Material loading\nClamping\nRough milling\nInspection",
        height=120)
    quality_req = st.text_area("Key quality requirements (one per line)",
        placeholder="Surface finish Ra 0.8\nTolerance +/-0.05mm",
        height=80)

tolerances_input = st.text_area("Critical tolerances (one per line)",
    placeholder="Bore diameter +/-0.02mm\nFlatness 0.05mm\nPerpendicularity 0.03mm",
    height=80)

uploaded_files = st.file_uploader(
    "Upload process documents (optional — improves accuracy)",
    type="pdf", accept_multiple_files=True)

if st.button("Generate FMEA", type="primary"):
    if not process_name or not steps_input:
        st.error("Please enter at least a process name and steps.")
    else:
        with st.spinner("Generating FMEA — this takes 20-30 seconds..."):
            steps = [s.strip() for s in steps_input.split("\n") if s.strip()]
            reqs = [r.strip() for r in quality_req.split("\n") if r.strip()]
            tolerances = [t.strip() for t in tolerances_input.split("\n") if t.strip()]
            pdf_paths = []
            for f in uploaded_files:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                tmp.write(f.read())
                tmp.close()
                pdf_paths.append(tmp.name)

            fmea_data = generate_fmea_with_context(
                process_name, steps, machine, material,
                reqs, tolerances, pdf_paths)

            for p in pdf_paths:
                try:
                    os.unlink(p)
                except:
                    pass

            st.session_state["fmea_data"] = fmea_data

if "fmea_data" in st.session_state:
    fmea = st.session_state["fmea_data"]
    rows = fmea["fmea_rows"]

    high = len([r for r in rows if r["priority"] == "HIGH"])
    med  = len([r for r in rows if r["priority"] == "MEDIUM"])
    low  = len([r for r in rows if r["priority"] == "LOW"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total items", len(rows))
    c2.metric("High priority", high)
    c3.metric("Medium priority", med)
    c4.metric("Low priority", low)

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    excel_file = export_fmea_to_excel(fmea, "fmea_output.xlsx")
    with open(excel_file, "rb") as f:
        st.download_button(
            "Download FMEA Excel",
            f,
            file_name=f"FMEA_{process_name}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )