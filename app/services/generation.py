from app.services.mistral_client import MistralClient
from app.services.prompting import OutputFormat, build_system_prompt, build_user_prompt
from app.services.retrieval import RetrievedChunk


class GenerationService:
    def __init__(self, mistral_client: MistralClient) -> None:
        self.mistral_client = mistral_client

    def generate(
        self,
        query: str,
        intent: str,
        output_format: OutputFormat,
        evidence: list[RetrievedChunk],
    ) -> str:
        system_prompt = build_system_prompt(intent, output_format)
        user_prompt = build_user_prompt(query, evidence, output_format)
        return self.mistral_client.generate_completion(system_prompt=system_prompt, user_prompt=user_prompt)
