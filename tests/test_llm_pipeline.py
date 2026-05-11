from src.pipelines import LLMInsightsPipeline


def test(base_config):
    pipe = LLMInsightsPipeline(config=base_config)
