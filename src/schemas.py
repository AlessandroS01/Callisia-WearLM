"""
Data Schemas Module.
Contains all Pydantic models used to enforce structured data throughout the app.
"""
from typing import List

from pydantic import BaseModel, Field


class ClinicalReportOutput(BaseModel):
    """
    The strict JSON schema that the LLM must follow when generating clinical insights.
    """

    # AI Chain Of Thought
    internal_reasoning: str = Field(
        description="Step-by-step logical deduction of the data before formulating the "
                    "observation. This part will be hidden from the UI. Debug purposes"
    )

    # Clinical Outputs
    primary_observation: str = Field(
        description="A 1-2 sentence executive summary of the patient's state over the 120-second "
                    "window. Do not include technical telemetry notes here."
    )
    cardiovascular_state: str = Field(
        description="Summary of the HR percentiles, trends, and clinical zones."
    )
    autonomic_tone: str = Field(
        description="Assessment of heart rate volatility (rigidity vs. dynamism) as a proxy for "
                    "sympathetic/parasympathetic activity."
    )
    movement_context: str = Field(
        description="Summary of physical activity and its correlation to the HR "
                    "(e.g., 'Elevated HR is justified by heavy movement')."
    )

    # Flags
    anomalies_detected: List[str] = Field(
        description="List of specific anomalies like sudden jolts or extreme variance. "
                    "Empty list if none."
    )
    requires_attention: bool = Field(
        description="True ONLY if dangerous anomalies exist "
                    "(e.g., falls, sustained unexplained tachycardia). "
                    "False for normal baseline activity."
    )

    # Action and Tech Notes
    recommended_system_action: str = Field(
        description="Operational or monitoring recommendations (e.g., 'Verify sensor contact',"
                    " 'Monitor for sustained trend'). Do NOT prescribe medical treatments."
    )
    technical_notes: str = Field(
        description="Use this field to acknowledge data alignment, receptive field artifacts, "
                    "or missing arrays. Keep it brief."
    )
