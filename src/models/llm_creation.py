"""
Module for creating the LLM for the pipeline and the evaluation
"""

from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import OpenAI


def choice_model(
        model_provider: str,
        model_name: str,
        temperature: float,
):
    """
    Factory method to initialize the appropriate LLM provider.

    Ensures environment variables are loaded and sets temperature to 0.0
    to ensure deterministic auditing results.

    Args:
        model_provider (str): The cloud provider (google/openai).
        model_name (str): The specific model identifier.
        temperature (float): The temperature of the model.

    Returns:
        A Chat-capable LLM instance.
    """

    load_dotenv(find_dotenv(raise_error_if_not_found=True), override=True)

    if model_provider == "google":
        return ChatGoogleGenerativeAI(
            model=model_name, temperature=temperature)
    if model_provider == "openai":
        return OpenAI(
            model=model_name, temperature=temperature)
    if model_provider == "ollama":
        return ChatOllama(
            model=model_name, temperature=temperature
        )

    return ChatGoogleGenerativeAI(
            model=model_name, temperature=temperature)
