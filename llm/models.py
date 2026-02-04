import os
from openai import OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI
from azure.identity import DefaultAzureCredential
from azure.core.credentials import TokenCredential
from azure.core.pipeline.policies import BearerTokenCredentialPolicy
from azure.core.pipeline import PipelineRequest, PipelineContext
from azure.core.rest import HttpRequest
from typing import Callable


def _norm_strish(v):
    # Defensive: avoid accidental tuple/list from trailing commas.
    if isinstance(v, (tuple, list)):
        v = v[0] if v else None
    return str(v).strip() if v is not None else None


class ModelSelector:

    def __init__(self, mode: str, model_name: str | None):
        self.mode = mode
        self.model_name = model_name
        self.selected_model = self.select_model(model_name)


    def select_model(self, model_name: str | None) -> str:
        if self.mode == "azure":

            if model_name == "gpt-5":
                self.azure_gpt5()
                return "gpt-5"
            elif model_name == "gpt-5.2":
                self.azure_gpt5_2()
                return "gpt-5.2"

            self.azure_default()
            return model_name

        if self.mode == "openai":
            openai_key = os.getenv("OPENAI_APIKEY")
            self.client = OpenAI(api_key=openai_key)
            self.async_client = AsyncOpenAI(api_key=openai_key)
            return model_name or os.getenv("OPENAI_MODEL")

        return model_name

    def azure_gpt5(self):

        api_version = _norm_strish(os.getenv("AZURE_OPENAIAPI_VERSION") or "2025-04-01-preview")
        azure_endpoint = _norm_strish(
            os.getenv("AZURE_OPENAI_ENDPOINT")
            or "https://csnf-singularity-aoai-eastus2.openai.azure.com/"
        )

        auth_kwargs = self._get_azure_auth_kwargs()
        self.client = AzureOpenAI(api_version=api_version, azure_endpoint=azure_endpoint, **auth_kwargs)
        self.async_client = AsyncAzureOpenAI(api_version=api_version, azure_endpoint=azure_endpoint, **auth_kwargs)

    def azure_gpt5_2(self):
        api_version = _norm_strish(
            os.getenv("AZURE_OPENAIAPI_VERSION_GPT5_2")
            or os.getenv("AZURE_OPENAIAPI_VERSION")
            or "2024-12-01-preview"
        )
        azure_endpoint = _norm_strish(
            os.getenv("AZURE_OPENAI_ENDPOINT_GPT5_2")
            or os.getenv("AZURE_OPENAI_ENDPOINT")
            or "https://gpt-gem.openai.azure.com/"
        )
        api_key = _norm_strish(os.getenv("AZURE_OPENAI_APIKEY_GPT5_2") or os.getenv("AZURE_OPENAI_APIKEY"))

        if api_key:
            self.client = AzureOpenAI(api_version=api_version, azure_endpoint=azure_endpoint, api_key=api_key)
            self.async_client = AsyncAzureOpenAI(api_version=api_version, azure_endpoint=azure_endpoint, api_key=api_key)
        else:
            self.client = AzureOpenAI(api_version=api_version, azure_endpoint=azure_endpoint, azure_ad_token_provider=self.token_provider)
            self.async_client = AsyncAzureOpenAI(api_version=api_version, azure_endpoint=azure_endpoint, azure_ad_token_provider=self.token_provider)


    def azure_default(self):
        api_version = _norm_strish(os.getenv("AZURE_OPENAIAPI_VERSION") or "2025-04-01-preview")
        azure_endpoint = _norm_strish(
            os.getenv("AZURE_OPENAI_ENDPOINT")
            or "https://csnf-singularity-aoai-eastus2.openai.azure.com/"
        )

        self.client = AzureOpenAI(api_version=api_version, azure_endpoint=azure_endpoint, azure_ad_token_provider=self.token_provider)
        self.async_client = AsyncAzureOpenAI(api_version=api_version, azure_endpoint=azure_endpoint, azure_ad_token_provider=self.token_provider)

    def _get_azure_auth_kwargs(self) -> dict:
        azure_api_key = _norm_strish(os.getenv("AZURE_OPENAI_APIKEY"))
        azure_api_endpoint = _norm_strish(os.getenv("AZURE_OPENAI_ENDPOINT"))
        if azure_api_key and azure_api_endpoint:
            return {"api_key": azure_api_key}
        return {"azure_ad_token_provider": self.token_provider}



    def _make_request(self) -> PipelineRequest[HttpRequest]:
        return PipelineRequest(
            HttpRequest("CredentialWrapper", "https://fakeurl"), PipelineContext(None)
        )

    def get_bearer_token_provider(
        credential: TokenCredential, *scopes: str
    ) -> Callable[[], str]:
        policy = BearerTokenCredentialPolicy(credential, *scopes)
        def wrapper(self) -> str:
            request = self._make_request()
            policy.on_request(request)
            return request.http_request.headers["Authorization"][len("Bearer ") :]
        return wrapper

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(
            managed_identity_client_id="e6162a0d-e540-4454-995f-30bcb97f35b4"
        ),
        "https://cognitiveservices.azure.com/.default",
    )

if __name__ == "__main__":
    selector = ModelSelector(mode="azure", model_name="gpt-5")
    print(f"Selected model: {selector.selected_model}")


def build_chat_request_kwargs(
    *,
    messages: list,
    model: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    timeout: int | None = None,
) -> dict:

    request_kwargs: dict = {
        "messages": messages,
        "model": model,
        "timeout": timeout,
        "top_p": top_p,
        "max_completion_tokens": max_tokens,
    }


    if model not in {"gpt-5", "gpt-5.2"}:
        request_kwargs["temperature"] = 0 if temperature is None else temperature

    return request_kwargs
