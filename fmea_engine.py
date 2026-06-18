from openai import OpenAI
import json
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

SYSTEM_PROMPT = """You are an expert manufacturing quality engineer
specialising in FMEA (Failure Mode and Effects Analysis) per AIAG standards.

Given a manufacturing process description, generate a complete PFMEA table.

For each process step, identify ALL realistic failure modes. Score each:
- Severity (1-10): impact of failure on customer/process
- Occurrence (1-10): likelihood of failure cause occurring
- Detection (1-10): likelihood current controls will NOT detect failure
- RPN = Severity x Occurrence x Detection

Consider the following when generating failure modes:
- Specific tolerances provided
- Material properties
- Machine capabilities
- Production line specific checks mentioned in documents

Return ONLY valid JSON in this exact structure, no other text:
{
  "process_name": "string",
  "fmea_rows": [
    {
      "process_step": "string",
      "failure_mode": "string",
      "effect": "string",
      "severity": integer,
      "cause": "string",
      "occurrence": integer,
      "current_controls": "string",
      "detection": integer,
      "rpn": integer,
      "recommended_actions": "string",
      "priority": "HIGH/MEDIUM/LOW"
    }
  ]
}

Priority rules: HIGH if RPN > 150, MEDIUM if 75-150, LOW if < 75.
Generate minimum 2 failure modes per process step."""


def generate_fmea(process_name, process_steps, machine,
                  material, quality_requirements, tolerances, context=""):

    user_prompt = f"""Generate a complete PFMEA for:

Process: {process_name}
Machine: {machine}
Material: {material}
Process Steps: {', '.join(process_steps)}
Key Quality Requirements: {', '.join(quality_requirements)}
Critical Tolerances: {', '.join(tolerances)}
Additional Context from Documents: {context}

Generate minimum 2 failure modes per process step.
Be specific to this exact process, material, and tolerances."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3
    )

    raw = response.choices[0].message.content
    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
    fmea_data = json.loads(clean.strip())

    for row in fmea_data["fmea_rows"]:
        correct_rpn = row["severity"] * row["occurrence"] * row["detection"]
        row["rpn"] = correct_rpn
        if correct_rpn > 150:
            row["priority"] = "HIGH"
        elif correct_rpn > 75:
            row["priority"] = "MEDIUM"
        else:
            row["priority"] = "LOW"

    return fmea_data


if __name__ == "__main__":
    result = generate_fmea(
        process_name="CNC Milling — Aluminium Housing",
        process_steps=["Material loading", "Clamping",
                       "Rough milling", "Finish milling", "Inspection"],
        machine="DMG MORI DMU 50",
        material="Aluminium 6061-T6",
        quality_requirements=["Surface finish Ra 0.8", "Tolerance +/-0.05mm"],
        tolerances=["Bore diameter +/-0.02mm", "Flatness 0.05mm",
                    "Perpendicularity 0.03mm"]
    )
    print(json.dumps(result, indent=2))
