from utils import GenAIEmbeddingClient, GenAIChatClient
from rag_inference import RAGInferencer
from config_param import Param

class RAGInferenceRunner:
    def __init__(
        self,
        embedding_provider: str,
        embedding_model: str,
        chat_provider: str,
        chat_model: str,
        api_key: str,
        api_version: str,
        azure_endpoint: str = None,
        base_url: str = None,
    ):
        # Only initialize the clients here without threading or locks
        self.embedding_client = GenAIEmbeddingClient(
            provider=embedding_provider,
            model=embedding_model,
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            base_url=base_url
        )

        self.chat_client = GenAIChatClient(
            provider=chat_provider,
            model=chat_model,
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            base_url=base_url
        )

        self.rag_inferencer = RAGInferencer(self.embedding_client, self.chat_client)

    def run_inference(
        self,
        question: str,
        index_path: str,
        system_prompt: str = None,
        top_k: int = 5,
        mode: str = "standard"
    ) -> pd.DataFrame:
        if system_prompt is None:
            system_prompt = Param.get_rag_prompt()

        return self.rag_inferencer.infer(
            system_prompt=system_prompt,
            index_path=index_path,
            question=question,
            mode=mode,
            top_k=top_k
        )