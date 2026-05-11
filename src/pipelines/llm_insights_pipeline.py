"""
LLM Insights Pipeline Module.

This module provides the :class:`LLMInsightsPipeline` class, which serves as
the cognitive engine of the clinical insights architecture (Stages 3 & 4).
It is responsible for taking raw, continuous physiological predictions (like
heart rate arrays), aggregating them into meaningful clinical contexts, and
interfacing with Large Language Models (LLMs) to generate human-readable
clinical text.
"""
import json
import os

from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from src.prompts import CLINICAL_SYSTEM_PROMPT


class LLMInsightsPipeline:
    """
    Executes the data aggregation and LLM explanation workflow.

    This pipeline handles the transformation of structured numerical arrays
    into actionable clinical insights. It manages the prompt engineering,
    context aggregation, and API interactions required to produce the final
    clinical text.

    :ivar config: The global configuration dictionary containing LLM parameters,
                      prompt templates, and API settings.
    :vartype config: dict
    """

    def __init__(self, config: dict):
        """
        Initializes the insights pipeline with the necessary configuration.

        :param config: A dictionary containing the settings required for
                       aggregation and LLM interaction.
        """
        self.config = config
        self.model = self._model_creation()

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", CLINICAL_SYSTEM_PROMPT),
            ("user", "Here is the patient's 120-second telemetry payload:\n{clinical_data}")
        ])

        self.chain = self.prompt | self.model | StrOutputParser()

        self.chain.get_graph().to_json()

    def _model_creation(self):

        load_dotenv()

        llm_param_list = self.config.get("llm", {})
        model_name = llm_param_list.get("model_name", "gemini-3.1-flash-lite")
        temperature = llm_param_list.get("temperature", 0.1)

        model = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature
        )

        return model


    def run(self, payload: dict):
        """
        Executes the LLM insights generation process.

        :param payload: A dictionary containing the aggregated data

        Note:
            This method is currently a placeholder. It will eventually be
            designed to accept the predicted physiological arrays (e.g., HR,
            time indices) from the SignalProcessingPipeline, aggregate the
            data, and trigger the LLM inference.
        """

        print("Generating LLM clinical insights...")

        # 1. Format the dictionary into clean, indented JSON for the LLM
        formatted_payload = json.dumps(payload, indent=2)

        # 2. Invoke the chain, passing the formatted JSON into the prompt variable
        final_report = self.chain.invoke({
            "clinical_data": formatted_payload
        })

        return final_report
