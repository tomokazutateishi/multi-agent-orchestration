"""
役割ごとのLLM割り当て。Documents/.env の API キーを参照。
USE_OPENAI_ONLY=1 で全役を OpenAI に。環境変数 ROLE_XXX_MODEL で上書き可能。
形式: ROLE_ORCHESTRATOR_MODEL=google:gemini-2.0-flash など（プロバイダー:モデルID）

4役構成（社長は削除済み）:
  1. オーケストレーター/プランナー（PM寄り）: orchestrator + marketer + planner
  2. エグゼキューター＋ドメインエキスパート（技術）: engineer
  3. ドメインエキスパート（UI/UX）: ワークフロー用（run.py には未ノード化）
  4. クリティック/レビュアー＋エグゼキューター（検証・再現）: reviewer
"""
import os


OPTIMAL_ROLE_MODELS = {
    "orchestrator": "google:gemini-2.0-flash",
    "marketer": "openai:gpt-4o-mini",
    "planner": "openai:gpt-4o-mini",
    "engineer": "openai:gpt-4o",
    "reviewer": "google:gemini-2.0-flash",
}

FALLBACK_ROLE_MODELS = {
    "orchestrator": "openai:gpt-4o-mini",
    "marketer": "openai:gpt-4o-mini",
    "planner": "openai:gpt-4o-mini",
    "engineer": "openai:gpt-4o",
    "reviewer": "openai:gpt-4o-mini",
}

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "google": "gemini-2.0-flash",
    "anthropic": "claude-3-5-sonnet-20241022",
}


def _parse_provider_model(spec: str):
    """ROLE_XXX_MODEL の 'provider:model_id' をパース"""
    if not spec or ":" not in spec:
        return None, None
    provider, model = spec.strip().split(":", 1)
    return provider.strip().lower(), model.strip()


def _build_llm(provider: str, model_id: str):
    """プロバイダーとモデルIDから LangChain LLM を生成"""
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_id, api_key=os.environ.get("OPENAI_API_KEY"))
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model_id, google_api_key=os.environ.get("GOOGLE_API_KEY"))
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model_id, api_key=os.environ.get("ANTHROPIC_API_KEY"))
    return None


def build_llm_registry():
    """役割ごとの LLM を構築。USE_OPENAI_ONLY=1 のときは全役 OpenAI。4役構成（社長削除済み）。"""
    use_openai_only = os.environ.get("USE_OPENAI_ONLY") == "1"
    by_role = {}
    for role in ("orchestrator", "marketer", "planner", "engineer", "reviewer"):
        env_key = f"ROLE_{role.upper()}_MODEL"
        spec = os.environ.get(env_key)
        if spec:
            provider, model_id = _parse_provider_model(spec)
        else:
            provider, model_id = None, None
        if use_openai_only:
            provider, model_id = "openai", (OPTIMAL_ROLE_MODELS.get(role) or "gpt-4o-mini").split(":")[-1]
        if not provider or not model_id:
            default = OPTIMAL_ROLE_MODELS.get(role, "openai:gpt-4o-mini")
            provider, model_id = _parse_provider_model(default)
        llm = _build_llm(provider, model_id)
        if llm is None:
            fallback = FALLBACK_ROLE_MODELS.get(role, "openai:gpt-4o-mini")
            provider, model_id = _parse_provider_model(fallback)
            llm = _build_llm(provider, model_id)
        by_role[role] = llm
    return {"by_role": by_role}
