import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field


MODEL = None
TOKENIZER = None
MODEL_NAME = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.2
    max_tokens: int = 512
    stream: bool = False
    seed: Optional[int] = None
    top_p: Optional[float] = None


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = "hf-server-completion"
    object: str = "chat.completion"
    model: str
    choices: list[ChatCompletionChoice]


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def dtype_from_name(name: str | None):
    if name in {None, "auto"}:
        return "auto"
    import torch

    if not hasattr(torch, name):
        raise ValueError(f"Unknown torch dtype: {name}")
    return getattr(torch, name)


def quantization_config():
    if not env_bool("HF_LOAD_IN_4BIT", False):
        return None

    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=os.environ.get("HF_BNB_4BIT_QUANT_TYPE", "nf4"),
        bnb_4bit_use_double_quant=env_bool("HF_BNB_4BIT_USE_DOUBLE_QUANT", True),
        bnb_4bit_compute_dtype=dtype_from_name(
            os.environ.get("HF_BNB_4BIT_COMPUTE_DTYPE", "bfloat16")
        ),
    )


def load_model():
    global MODEL, TOKENIZER, MODEL_NAME

    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model = os.environ["HF_BASE_MODEL"]
    adapter_path = os.environ.get("HF_ADAPTER_PATH")
    tokenizer_id = os.environ.get("HF_TOKENIZER") or base_model
    trust_remote_code = env_bool("HF_TRUST_REMOTE_CODE", False)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        trust_remote_code=trust_remote_code,
        use_fast=env_bool("HF_USE_FAST_TOKENIZER", True),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "device_map": os.environ.get("HF_DEVICE_MAP", "auto"),
        "dtype": dtype_from_name(os.environ.get("HF_DTYPE", "auto")),
        "trust_remote_code": trust_remote_code,
    }
    qconfig = quantization_config()
    if qconfig is not None:
        model_kwargs["quantization_config"] = qconfig

    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    MODEL = model
    TOKENIZER = tokenizer
    MODEL_NAME = os.environ.get(
        "HF_SERVED_MODEL_NAME",
        Path(adapter_path or base_model).name,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    if req.stream:
        raise ValueError("Streaming responses are not implemented by this example server.")

    import torch

    messages = [message.model_dump() for message in req.messages]
    if hasattr(TOKENIZER, "apply_chat_template") and TOKENIZER.chat_template:
        prompt = TOKENIZER.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)

    if req.seed is not None:
        torch.manual_seed(req.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(req.seed)

    inputs = TOKENIZER(prompt, return_tensors="pt").to(MODEL.device)
    generation_kwargs = {
        "max_new_tokens": req.max_tokens,
        "do_sample": req.temperature > 0,
        "temperature": req.temperature,
        "pad_token_id": TOKENIZER.pad_token_id,
        "eos_token_id": TOKENIZER.eos_token_id,
    }
    if req.top_p is not None:
        generation_kwargs["top_p"] = req.top_p

    with torch.no_grad():
        output = MODEL.generate(**inputs, **generation_kwargs)

    generated = output[0][inputs["input_ids"].shape[-1] :]
    text = TOKENIZER.decode(generated, skip_special_tokens=True).strip()

    return ChatCompletionResponse(
        model=req.model or MODEL_NAME,
        choices=[
            ChatCompletionChoice(
                message=ChatMessage(role="assistant", content=text),
            )
        ],
    )
