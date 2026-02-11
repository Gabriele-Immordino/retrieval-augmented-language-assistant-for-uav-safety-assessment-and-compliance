from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from flask import Flask, jsonify, request, Response
from werkzeug.exceptions import HTTPException

from LLM.LLM_openAI_QueryCall import QueryGenerator
from LLM.LLM_openAI_Chatbot import LLMChatbot
from LLM.LLM_openAI_Classification import IndicatorName, InitialOperationInput, LLMIndicatorAssistant
from PreProcessing.embeddingToolsFAISSv2 import EmbeddingToolFAISS
from RAG.ragv2 import RAG

from db import DatabaseService, Message, User
import os

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:////app/chat.db")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://host.docker.internal:11434/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "ollama")

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / "PreProcessing" / "ProcessedFiles"

_chatbot: LLMChatbot | None = None
_rag: RAG | None = None
_embedder: EmbeddingToolFAISS | None = None
_preprocessor: QueryGenerator | None = None
_indicator_engine: LLMIndicatorAssistant | None = None
_indicator_rag: RAG | None = None
_config_cache: Dict[str, Any] | None = None


_ALLOWED_INDICATORS: Tuple[IndicatorName, ...] = (
    "likely_regulatory_pathway",
    "initial_ground_risk_orientation",
    "initial_air_risk_orientation",
    "expected_assessment_depth",
)

_ALLOWED_VLOS = ("VLOS", "BVLOS")
_ALLOWED_GROUND_ENV = ("controlled_area", "sparsely_populated", "populated")
_ALLOWED_AIRSPACE = ("controlled", "uncontrolled")
_ALLOWED_MASS = ("lt_25kg", "gte_25kg")
_ALLOWED_ALTITUDE = ("le_50m", "gt_50m_le_120m", "gt_120m_le_150m", "gt_150m")
_ALLOWED_REASONING = ("none", "low", "medium", "high")


def _error_response(status: int, code: str, message: str, details: Any = None):
    payload: Dict[str, Any] = {"error": {"code": code, "message": message}}
    if details is not None:
        payload["error"]["details"] = details
    return jsonify(payload), status


def _ensure_chatbot() -> LLMChatbot:
    global _chatbot, _rag, _embedder

    if _chatbot is not None:
        return _chatbot

    if not INDEX_DIR.exists():
        raise FileNotFoundError(f"Index directory not found: {INDEX_DIR}")

    _embedder = EmbeddingToolFAISS(output_dir=INDEX_DIR)
    _embedder.load_index()
    _rag = RAG(embedding_tool=_embedder, chunks=_embedder.metadata)
    _chatbot = LLMChatbot(rag_system=_rag, base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

    return _chatbot


def _ensure_preprocessor() -> QueryGenerator:
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = QueryGenerator(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    return _preprocessor


def _load_config() -> Dict[str, Any]:
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    cfg_path = BASE_DIR / "config.yml"
    if not cfg_path.exists():
        _config_cache = {}
        return _config_cache

    try:
        import yaml  # type: ignore

        with cfg_path.open("r", encoding="utf-8") as f:
            _config_cache = yaml.safe_load(f) or {}
    except Exception:
        _config_cache = {}

    return _config_cache


def _ensure_indicator_engine() -> LLMIndicatorAssistant:
    global _indicator_engine, _indicator_rag, _embedder

    if _indicator_engine is not None:
        return _indicator_engine

    cfg = _load_config()
    output_dir = Path(cfg.get("output_dir"))
    rag_mode = str(cfg.get("rag_mode"))
    reranker_mode = str(cfg.get("reranker_mode"))

    if _embedder is None:
        _embedder = EmbeddingToolFAISS(output_dir=BASE_DIR / output_dir)
        _embedder.load_index()

    chunks = _embedder.metadata
    if not chunks:
        chunk_dir = Path(cfg.get("path_fname"))
        chunk_file = Path(BASE_DIR / chunk_dir / str(cfg.get("fname")))
        if not chunk_file.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunk_file}")
        with chunk_file.open("r", encoding="utf-8") as f:
            chunks = json.load(f)

    _indicator_rag = RAG(
        embedding_tool=_embedder,
        chunks=chunks,
        default_mode=rag_mode,
        reranker=reranker_mode,
    )

    _indicator_engine = LLMIndicatorAssistant(rag_system=_indicator_rag, base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    return _indicator_engine


def _normalize_chat_history(raw: Any) -> List[Dict[str, str]]:
    if not isinstance(raw, list):
        raise ValueError("Field 'chat_history' must be a list of role/content objects")

    cleaned: List[Dict[str, str]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"chat_history[{idx}] must be an object with role and content")

        role = item.get("role")
        content = item.get("content")

        if role not in ("user", "assistant"):
            raise ValueError(f"chat_history[{idx}].role must be 'user' or 'assistant'")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"chat_history[{idx}].content must be a non-empty string")

        cleaned.append({"role": role, "content": content.strip()})

    if not cleaned:
        raise ValueError("chat_history cannot be empty")

    return cleaned


def _parse_chatbot_payload(payload: Any) -> Tuple[str, str, bool, str, List[Dict[str, str]]]:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")

    user_id = payload.get("user_id")
    user_name = payload.get("user_name")
    preprocess_query = payload.get("preprocess_query")
    chat_history_raw = payload.get("chat_history")
    reasoning_effort = _normalize_reasoning_effort(payload.get("reasoning_effort", "medium"), "medium")

    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("Field 'user_id' is required and must be a non-empty string")
    if not isinstance(user_name, str) or not user_name.strip():
        raise ValueError("Field 'user_name' is required and must be a non-empty string")
    if not isinstance(preprocess_query, bool):
        raise ValueError("Field 'preprocess_query' is required and must be a boolean")
    
    chat_history = _normalize_chat_history(chat_history_raw)

    return user_id.strip(), user_name.strip(), preprocess_query, reasoning_effort, chat_history


def _normalize_reasoning_effort(value: Any, fallback: str = "medium") -> str:
    candidate = value
    if isinstance(candidate, str):
        lowered = candidate.strip().lower()
        if lowered in _ALLOWED_REASONING:
            return lowered
    return fallback


def _extract_latest_user_question(chat_history: List[Dict[str, str]]) -> str:
    for msg in reversed(chat_history):
        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
            question = msg["content"].strip()
            if question:
                return question
    raise ValueError("chat_history must contain at least one user message with content")


def _format_subquery_display(subquery_index: int, total: int, question: str, answer: str) -> str:
    heading = f"**Sub-query {subquery_index}/{total}"
    question_clean = (question or "").strip()
    if question_clean:
        heading += f": {question_clean}"
    heading += "**"
    body = (answer or "").strip()
    if body:
        return f"{heading}\n\n{body}"
    return heading


def _parse_indicator_classif_payload(payload: Any) -> Tuple[InitialOperationInput, List[IndicatorName]]:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")

    def _require_choice(field: str, allowed: Tuple[str, ...]) -> str:
        val = payload.get(field)
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"Field '{field}' is required and must be a non-empty string")
        cleaned = val.strip()
        if cleaned not in allowed:
            raise ValueError(f"Field '{field}' must be one of {', '.join(allowed)}")
        return cleaned

    indicators_raw = payload.get("indicators", list(_ALLOWED_INDICATORS))
    if not isinstance(indicators_raw, list) or len(indicators_raw) == 0:
        raise ValueError("Field 'indicators' must be a non-empty array of indicator names")

    indicators: List[IndicatorName] = []
    for idx, name in enumerate(indicators_raw):
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"indicators[{idx}] must be a non-empty string")
        cleaned = name.strip()
        if cleaned not in _ALLOWED_INDICATORS:
            raise ValueError(f"indicators[{idx}] must be one of {', '.join(_ALLOWED_INDICATORS)}")
        if cleaned not in indicators:
            indicators.append(cleaned)  # de-dup while preserving order

    op = InitialOperationInput(
        maximum_takeoff_mass_category=_require_choice("maximum_takeoff_mass_category", _ALLOWED_MASS),
        vlos_or_bvlos=_require_choice("vlos_or_bvlos", _ALLOWED_VLOS),
        ground_environment=_require_choice("ground_environment", _ALLOWED_GROUND_ENV),
        airspace_type=_require_choice("airspace_type", _ALLOWED_AIRSPACE),
        maximum_altitude_category=_require_choice("maximum_altitude_category", _ALLOWED_ALTITUDE),
    )

    return op, indicators


def _get_user_history(external_user_id: str):
    db = dbs.SessionLocal()
    try:
        from db import User  # local import to avoid circular issues if you extend db.py later

        user = db.query(User).filter(User.user_id == external_user_id).one_or_none()
        if user is None:
            return None, []

        msgs = (
            db.query(Message)
            .filter(Message.user_id == user.id)
            .order_by(Message.created_at.asc())
            .limit(500)
            .all()
        )

        return user, msgs
    finally:
        db.close()


def _build_prompt_preview(chatbot: LLMChatbot, question: str, chat_history: List[Dict[str, str]]):
    hits = chatbot.rag_system.search(question)
    messages, included_indices = chatbot.build_messages_client(question, chat_history, hits)
    return {
        "messages": messages,
        "sources": chatbot.extract_sources(hits, included_indices),
        "chunks": hits,
    }


def _store_message(external_user_id: str, username: str, role: str, content: str) -> None:
    if role not in ("user", "assistant", "system"):
        return
    if not isinstance(content, str) or not content.strip():
        return

    db = dbs.SessionLocal()
    try:
        user = dbs.get_or_create_user(db, external_user_id, username)
        db.add(Message(user_id=user.id, role=role, content=content.strip()))
        db.commit()
    finally:
        db.close()


@app.get("/preload_models")
def preload_models():
    try:
        _ = _ensure_chatbot()
        _ = _ensure_preprocessor()
        return jsonify({"status": "models preloaded"}), 200
    except Exception as exc:
        return _error_response(500, "preload_failed", f"Failed to preload models: {exc}")


@app.get("/health")
def health():
    try:
        _ = _ensure_chatbot()
        return jsonify({"status": "ok"}), 200
    except Exception as exc:
        return jsonify({"status": "error", "details": str(exc)}), 500


@app.post("/api/v1/chat")
def chat():

    if not request.is_json:
        print("Request is not JSON", flush=True)
        return _error_response(415, "unsupported_media_type", "Content-Type must be application/json")

    payload = request.get_json(silent=True)
    if payload is None:
        print("Failed to parse JSON payload", flush=True)
        return _error_response(400, "invalid_json", "Request body is not valid JSON")


    try:
        user_id, user_name, preprocess_query, reasoning_effort, chat_history = _parse_chatbot_payload(payload)
        user_question = _extract_latest_user_question(chat_history)
    except ValueError as exc:
        print(f"Payload validation error: {exc}", flush=True)
        return _error_response(400, "invalid_request", str(exc))


    try:
        print("Initializing chatbot...", flush=True)
        chatbot = _ensure_chatbot()
        print("Chatbot initialized successfully", flush=True)
    except Exception as exc:
        print(f"Failed to initialize chatbot: {exc}", flush=True)
        return _error_response(500, "initialization_failed", f"Failed to initialize LLM: {exc}")

    _store_message(user_id, user_name, "user", user_question)

    if preprocess_query:
        try:
            preprocessor = _ensure_preprocessor()
        except Exception as exc:
            return _error_response(500, "preprocessor_init_failed", f"Failed to load preprocessor: {exc}")

        try:
            generated = preprocessor.generate_queries(user_question)
            generated_queries = [
                q.get("query", "").strip()
                for q in generated.get("queries", [])
                if isinstance(q, dict) and isinstance(q.get("query"), str) and q.get("query").strip()
            ]
            if len(generated_queries) == 0:
                raise ValueError("Preprocessing error: no valid queries generated")
        except Exception as exc:
            return _error_response(500, "preprocess_failed", f"Failed to preprocess query: {exc}")

        combined_parts: List[str] = []
        prompt_previews: List[Dict[str, Any]] = []

        for idx, q in enumerate(generated_queries, start=1):
            prompt_preview = _build_prompt_preview(chatbot, q, chat_history)
            prompt_previews.append(prompt_preview)

            try:
                result = chatbot.answer(q, chat_history, reasoning_effort=reasoning_effort)
            except Exception as exc:
                return _error_response(
                    500,
                    "answer_failed",
                    f"Failed while answering generated query {idx}: {exc}",
                )

            combined_parts.append(f"{result.get('answer', '')}".strip())

        final_answer = "\n\n".join([p for p in combined_parts if p])

        aggregated_sources = [src for p in prompt_previews for src in p.get("sources", [])]

        _store_message(user_id, user_name, "assistant", final_answer)

        response_body = {
            "user_id": user_id,
            "user_name": user_name,
            "original_question": user_question,
            "generated_queries": generated_queries,
            "answer": final_answer,
            "sources": aggregated_sources,
            "reasoning": " | ".join((p.get("reasoning", "") for p in prompt_previews if p.get("reasoning"))),
        }

        return jsonify(response_body), 200

    _ = _build_prompt_preview(chatbot, user_question, chat_history)
    try:
        result = chatbot.answer(user_question, chat_history, reasoning_effort=reasoning_effort)
    except Exception as exc:
        print(f"Failed to generate answer: {exc}", flush=True)
        return _error_response(500, "answer_failed", f"Failed to generate answer: {exc}")

    assistant_answer = result.get("answer", "")
    _store_message(user_id, user_name, "assistant", assistant_answer)

    response_body = {
        "user_id": user_id,
        "user_name": user_name,
        "original_question": user_question,
        "answer": assistant_answer,
        "sources": result.get("sources", []),
        "reasoning": result.get("reasoning", ""),
    }

    return jsonify(response_body), 200


@app.post("/api/v1/chat/stream")
def chat_stream():
    """Streaming chat endpoint using Server-Sent Events."""
    if not request.is_json:
        return _error_response(415, "unsupported_media_type", "Content-Type must be application/json")

    payload = request.get_json(silent=True)
    if payload is None:
        return _error_response(400, "invalid_json", "Request body is not valid JSON")

    try:
        user_id, user_name, preprocess_query, reasoning_effort, chat_history = _parse_chatbot_payload(payload)
        user_question = _extract_latest_user_question(chat_history)
    except ValueError as exc:
        return _error_response(400, "invalid_request", str(exc))

    try:
        chatbot = _ensure_chatbot()
    except Exception as exc:
        return _error_response(500, "initialization_failed", f"Failed to initialize LLM: {exc}")

    _store_message(user_id, user_name, "user", user_question)

    query_plan: List[Dict[str, str]] = [{"query": user_question, "reasoning_effort": reasoning_effort}]
    generated_queries: List[str] = []

    if preprocess_query:
        try:
            preprocessor = _ensure_preprocessor()
        except Exception as exc:
            return _error_response(500, "preprocessor_init_failed", f"Failed to load preprocessor: {exc}")

        try:
            generated = preprocessor.generate_queries(user_question)
            plan: List[Dict[str, str]] = []
            for item in generated.get("queries", []):
                if not isinstance(item, dict):
                    continue
                q_text = item.get("query")
                if not isinstance(q_text, str):
                    continue
                cleaned = q_text.strip()
                if not cleaned:
                    continue
                plan.append(
                    {
                        "query": cleaned,
                        "reasoning_effort": _normalize_reasoning_effort(
                            item.get("reasoning_level"),
                            reasoning_effort,
                        ),
                    }
                )

            if len(plan) == 0:
                raise ValueError("Preprocessing error: no valid queries generated")

            query_plan = plan
            generated_queries = [entry["query"] for entry in plan]
        except Exception as exc:
            return _error_response(500, "preprocess_failed", f"Failed to preprocess query: {exc}")

    multi_mode = preprocess_query and len(query_plan) > 1
    total_segments = len(query_plan)

    def _sse(payload: Dict[str, Any]) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    def generate():
        try:
            if chatbot.rag_system is None:
                raise RuntimeError("rag_system is not set")

            for idx, segment in enumerate(query_plan, start=1):
                query_text = segment["query"]
                segment_effort = segment.get("reasoning_effort", reasoning_effort)

                meta: Dict[str, Any] = {}
                if multi_mode:
                    meta = {
                        "subquery_index": idx,
                        "total_subqueries": total_segments,
                        "subquery_text": query_text,
                    }
                    yield _sse({"type": "subquery_start", **meta})

                hits = chatbot.rag_system.search(query_text)
                messages, included_indices = chatbot.build_messages_client(query_text, chat_history, hits)
                sources = chatbot.extract_sources(hits, included_indices)

                yield _sse({"type": "sources", "sources": sources, **meta})

                common_args = dict(
                    model=chatbot.model_name,
                    messages=messages,
                    max_tokens=56000,
                    temperature=0.2,
                    top_p=0.9,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    n=1,
                    reasoning_effort=segment_effort,
                )

                response = chatbot.client.chat.completions.create(
                    stream=True,
                    **common_args,
                )

                collected_answer: List[str] = []
                collected_reasoning: List[str] = []

                for chunk in response:
                    choice = chunk.choices[0]
                    delta = choice.delta

                    token = delta.content or ""
                    if token:
                        collected_answer.append(token)
                        payload = {"type": "token", "token": token}
                        payload.update(meta)
                        yield _sse(payload)

                    reasoning_piece = getattr(delta, "reasoning", None)
                    if reasoning_piece:
                        collected_reasoning.append(reasoning_piece)
                        payload = {"type": "reasoning", "token": reasoning_piece}
                        payload.update(meta)
                        yield _sse(payload)

                final_answer = "".join(collected_answer).strip()
                final_reasoning = "".join(collected_reasoning).strip()
                display_answer = final_answer
                if multi_mode:
                    display_answer = _format_subquery_display(idx, total_segments, query_text, final_answer)

                _store_message(user_id, user_name, "assistant", display_answer)

                final_payload = {
                    "type": "subquery_done" if multi_mode else "done",
                    "answer": final_answer,
                    "display_answer": display_answer,
                    "reasoning": final_reasoning,
                }
                final_payload.update(meta)
                yield _sse(final_payload)

            if multi_mode:
                yield _sse(
                    {
                        "type": "done",
                        "subquery_count": total_segments,
                        "generated_queries": generated_queries,
                    }
                )

        except Exception as exc:
            yield _sse({"type": "error", "message": str(exc)})

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',
        }
    )


@app.get("/api/v1/history/<external_user_id>")
def history(external_user_id: str):
    user, msgs = _get_user_history(external_user_id)

    if user is None:
        return jsonify({"user_id": external_user_id, "user_name": None, "messages": []}), 200

    return (
        jsonify(
            {
                "user_id": external_user_id,
                "user_name": user.username,
                "messages": [
                    {"role": m.role, "content": m.content, "ts": m.created_at.isoformat()}
                    for m in msgs
                ],
            }
        ),
        200,
    )


@app.post("/api/v1/history")
def history_post():
    if not request.is_json:
        return _error_response(415, "unsupported_media_type", "Content-Type must be application/json")

    payload = request.get_json(silent=True)
    if payload is None:
        return _error_response(400, "invalid_json", "Request body is not valid JSON")

    user_id = payload.get("user_id") if isinstance(payload, dict) else None
    if not isinstance(user_id, str) or not user_id.strip():
        return _error_response(400, "invalid_request", "Field 'user_id' is required and must be a non-empty string")

    user, msgs = _get_user_history(user_id.strip())
    if user is None:
        return jsonify({"user_id": user_id.strip(), "user_name": None, "messages": []}), 200

    return (
        jsonify(
            {
                "user_id": user_id.strip(),
                "user_name": user.username,
                "messages": [
                    {"role": m.role, "content": m.content, "ts": m.created_at.isoformat()}
                    for m in msgs
                ],
            }
        ),
        200,
    )


@app.delete("/api/v1/history/<external_user_id>")
def clear_history(external_user_id: str):
    """Delete all chat messages for a given user."""
    db = dbs.SessionLocal()
    try:

        user = db.query(User).filter(User.user_id == external_user_id).one_or_none()
        if user is None:
            return jsonify({"user_id": external_user_id, "deleted_count": 0}), 200

        deleted = db.query(Message).filter(Message.user_id == user.id).delete()
        db.commit()

        return jsonify({"user_id": external_user_id, "deleted_count": deleted}), 200
    except Exception as exc:
        db.rollback()
        return _error_response(500, "delete_failed", f"Failed to clear history: {exc}")
    finally:
        db.close()


@app.post("/api/v1/classification")
def classification():
    if not request.is_json:
        return _error_response(415, "unsupported_media_type", "Content-Type must be application/json")

    payload = request.get_json(silent=True)
    if payload is None:
        return _error_response(400, "invalid_json", "Request body is not valid JSON")

    try:
        op, indicators = _parse_indicator_classif_payload(payload)
    except ValueError as exc:
        return _error_response(400, "invalid_request", str(exc))

    try:
        engine = _ensure_indicator_engine()
    except Exception as exc:
        return _error_response(500, "initialization_failed", f"Failed to initialize classification engine: {exc}")

    cfg = _load_config()

    def _cfg_num(key: str, default: Any):
        try:
            return type(default)(cfg.get(key, default))
        except Exception:
            return default

    top_k = cfg.get("top_k")
    ce_keep_k = cfg.get("ce_keep_k")
    try:
        top_k = int(top_k) if top_k is not None else None
    except Exception:
        top_k = None
    try:
        ce_keep_k = int(ce_keep_k) if ce_keep_k is not None else None
    except Exception:
        ce_keep_k = None

    temperature = float(_cfg_num("temperature", 0.2))
    top_p = float(_cfg_num("top_p", 0.9))
    max_new_tokens = int(_cfg_num("max_new_tokens", 56000))
    frequency_penalty = float(_cfg_num("frequency_penalty", 0.0))
    presence_penalty = float(_cfg_num("presence_penalty", 0.0))
    reasoning_effort = str(cfg.get("reasoning_effort", "medium"))

    engine.clear_chat_history()

    results: Dict[str, Dict[str, Any]] = {}
    sources: Dict[str, List[str]] = {}

    for name in indicators:
        try:
            out = engine.answer_indicator(
                indicator=name,
                op=op,
                top_k=top_k,
                ce_keep_k=ce_keep_k,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stream=False,
                print_sources=False,
                reasoning_effort=reasoning_effort,
            )
        except Exception as exc:
            return _error_response(500, "classification_failed", f"Failed to compute indicator '{name}': {exc}")

        results[name] = out.get("result", {})
        sources[name] = out.get("sources", [])

    return (
        jsonify(
            {
                "operation": asdict(op),
                "indicators": results,
                "sources": sources,
            }
        ),
        200,
    )


@app.errorhandler(Exception)
def handle_errors(err: Exception):
    if isinstance(err, HTTPException):
        return _error_response(
            err.code,
            err.name.lower().replace(" ", "_"),
            err.description,
        )

    return _error_response(
        500,
        "internal_server_error",
        f"An unexpected error occurred: {err}",
    )

dbs = DatabaseService(DATABASE_URL)

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8080"))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    app.run(host=host, port=port, debug=debug)
