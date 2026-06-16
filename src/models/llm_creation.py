"""
Module for creating the LLM for the pipeline and the evaluation
"""
import os

from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


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
        model_provider (str): The cloud provider
        model_name (str): The specific model identifier.
        temperature (float): The temperature of the model.

    Returns:
        A Chat-capable LLM instance.
    """

    load_dotenv(find_dotenv(raise_error_if_not_found=True), override=True)

    if model_provider == "openai":
        return ChatOpenAI(
            model=model_name, temperature=temperature)
    if model_provider == "ollama-2":
        return ChatOllama(
            model=model_name, temperature=temperature
        )
    if model_provider == 'passau':
        passau_key = os.environ.get("PASSAU_KEY")
        return ChatOpenAI(
            model=model_name,
            api_key=passau_key,
            base_url="https://llms.innkube.fim.uni-passau.de",
            temperature=temperature
        )
    if model_provider == "mistral":
        return ChatMistralAI(
            model=model_name, temperature=temperature
        )
    if model_provider == "groq":
        return ChatGroq(
            model=model_name, temperature=temperature
        )
    return ChatGoogleGenerativeAI(
            model=model_name, temperature=temperature)
