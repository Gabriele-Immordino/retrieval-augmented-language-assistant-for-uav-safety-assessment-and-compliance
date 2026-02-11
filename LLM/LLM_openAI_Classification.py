import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Literal, Any, List as _List

from openai import OpenAI
from RAG.ragv2 import RAG
from LLM.query_structure import BASE_QUERIES, VLOS_MODE_TERMS, GROUND_ENV_TERMS, AIRSPACE_TYPE_TERMS, MASS_TERMS, ALTITUDE_TERMS#, PATHWAY_HINTS


VlosMode = Literal["VLOS", "BVLOS"]
GroundEnv = Literal["controlled_area", "sparsely_populated", "populated"]
AirspaceType = Literal["controlled", "uncontrolled"]
MassCategory = Literal["lt_25kg", "gte_25kg"] # less than 25 kg, greater than or equal to 25 kg
AltitudeCategory = Literal[
    "le_50m",      # up to 50 m above ground level (AGL)
    "gt_50m_le_120m",  # above 50 m and up to 120 m AGL
    "gt_120m_le_150m", # above 120 m and up to 150 m AGL
    "gt_150m",     # above 150 m AGL
]

IndicatorName = Literal[
    "likely_regulatory_pathway",
    "initial_ground_risk_orientation",
    "initial_air_risk_orientation",
    "expected_assessment_depth",
]


@dataclass
class InitialOperationInput:
    maximum_takeoff_mass_category: MassCategory
    vlos_or_bvlos: VlosMode
    ground_environment: GroundEnv
    airspace_type: AirspaceType
    maximum_altitude_category: AltitudeCategory



class LLMIndicatorAssistant:
    """
    SKYSAFE single indicator engine.

    You query one indicator at a time.
    The model must return a single JSON object:
    { "name": "...", "value": "...", "explanation": "..." }

    The model is constrained to use only the provided context.
    Context is built from RAG retrieved chunks plus a minimal input block
    containing only the fields required for the requested indicator.
    """

    def __init__(
        self,
        model_name: str = "gpt-oss:20b",
        rag_system: Optional[RAG] = None,
        system_rules_path: str = "LLM/system_rules_classificationTask.txt",
        developer_content_path: str = "LLM/developer_content.txt",
        base_url:str="http://localhost:11434/v1",
        api_key:str="ollama",
    ) -> None:
        if rag_system is None:
            raise ValueError("rag_system must be provided")

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.rag_system = rag_system

        self.system_rules = self._load_text_file(system_rules_path)
        self.developer_content = self._load_text_file(developer_content_path)

        self.chat_history: _List[Dict[str, str]] = []

        self.no_context_token = "[NO CONTEXT]"

        self._indicator_specs: Dict[str, Dict[str, Any]] = {
            "likely_regulatory_pathway": {
                "required_fields": [
                    "maximum_takeoff_mass_category",
                    "vlos_or_bvlos",
                    "ground_environment",
                    "airspace_type",
                    "maximum_altitude_category",
                ],
                "prompt": (
                    "Requested indicator: likely_regulatory_pathway\n"
                    "Return value guidance: a short string such as Open, Specific PDRA, or Specific SORA.\n"
                    "Return only the requested indicator with its explanation.\n"
                ),
            },
            "initial_ground_risk_orientation": {
                "required_fields": [
                    "maximum_takeoff_mass_category",
                    "ground_environment",
                ],
                "prompt": (
                    "Requested indicator: initial_ground_risk_orientation\n"
                    "Return value guidance: one of very_low, low, medium, high.\n"
                    "Explain using only the provided inputs and the context.\n"
                ),
            },
            "initial_air_risk_orientation": {
                "required_fields": [
                    "airspace_type",
                    "maximum_altitude_category",
                    "ground_environment",
                ],
                "prompt": (
                    "Requested indicator: initial_air_risk_orientation\n"
                    "Return value guidance: one of very_low, low, medium, high.\n"
                    "Explain using only the provided inputs and the context.\n"
                ),
            },
            "expected_assessment_depth": {
                "required_fields": [
                    "maximum_takeoff_mass_category",
                    "vlos_or_bvlos",
                    "ground_environment",
                    "airspace_type",
                ],
                "prompt": (
                    "Requested indicator: expected_assessment_depth\n"
                    "Return value guidance: a short string such as simple_declaration, "
                    "structured_assessment, or full_sora.\n"
                    "Explain which provided inputs increase assessment depth and why.\n"
                ),
            },
        }

    def _load_text_file(self, file_path: str) -> str:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    def clear_chat_history(self) -> None:
        self.chat_history = []

    def _select_inputs_for_indicator(
        self,
        op: InitialOperationInput,
        indicator: str,
    ) -> Dict[str, Any]:
        d = asdict(op)
        required = self._indicator_specs[indicator]["required_fields"]
        return {k: d[k] for k in required}

    def _format_op_inputs_block(self, selected_inputs: Dict[str, Any]) -> str:
        lines = [f"{k}: {selected_inputs[k]}" for k in selected_inputs]
        return "Operation inputs:\n" + "\n".join(lines)

    def _format_context(
        self,
        chunks: List[Dict],
        max_chars: int = 12000,
    ) -> Tuple[str, _List[int]]:
        parts: _List[str] = []
        used = 0
        included_indices: _List[int] = []

        for rec in chunks:
            idx = rec.get("chunk_index")
            title = rec.get("chunk_title", "")
            page = rec.get("page", "")
            head = f"[{idx}] {title}, page {page}"
            body = rec.get("chunk_text", "") or ""
            block = f"{head} > {body.strip()}\n"

            if used + len(block) > max_chars:
                break

            parts.append(block)
            if idx is not None:
                included_indices.append(idx)
            used += len(block)

        ctx = "\n\n".join(parts).strip()
        if not ctx:
            ctx = self.no_context_token

        return ctx, included_indices

    def _extract_sources(self, retrieved: List[Dict], included_indices: List[int]) -> List[str]:
        out: _List[str] = []
        by_chunk_index: Dict[int, Dict] = {}

        for rec in retrieved:
            idx = rec.get("chunk_index")
            if idx is not None:
                by_chunk_index[idx] = rec

        for idx in included_indices:
            rec = by_chunk_index.get(idx)
            if rec is None:
                continue
            filename = rec.get("source_file", "Unknown file source")
            title = rec.get("chunk_title", "Unknown section")
            page = rec.get("page", "(?)")
            out.append(f"[{idx}] {filename}: {title}, page {page}")

        return out

    def _build_messages(
        self,
        indicator: str,
        op: InitialOperationInput,
        retrieved_sources: List[Dict],
    ) -> Tuple[List[Dict[str, str]], List[int]]:
        ctx, included_indices = self._format_context(retrieved_sources)

        selected_inputs = self._select_inputs_for_indicator(op, indicator)
        # print("Selected inputs for indicator:", selected_inputs, "indicator:", indicator)
        op_block = self._format_op_inputs_block(selected_inputs)

        indicator_prompt = self._indicator_specs[indicator]["prompt"]

        user_request = (
            "You will receive a context section and a small set of operation inputs.\n"
            "Use only the context and the provided inputs.\n"
            "Return exactly one JSON object with keys name, value, explanation.\n"
            "Do not output any additional text.\n\n"
            f"{indicator_prompt}\n\n"
            f"{op_block}"
        )

        context_message = f"Context:\n{ctx}"

        messages: _List[Dict[str, str]] = [
            {"role": "system", "content": self.system_rules},
            {"role": "developer", "content": self.developer_content},
        ]

        for msg in self.chat_history:
            role = msg.get("role", "")
            content = str(msg.get("content", "")).strip()
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": user_request})
        messages.append({"role": "user", "content": context_message})

        return messages, included_indices

    def answer_indicator(
        self,
        indicator: IndicatorName,
        op: InitialOperationInput,
        top_k: Optional[int] = None,
        ce_keep_k: Optional[int] = None,
        max_new_tokens: int = 56000,
        temperature: float = 0.2,
        top_p: float = 0.9,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stream: bool = False,
        print_sources: bool = True,
        print_prompt: bool = False,
        reasoning_effort: str = "medium",
    ) -> Dict[str, Any]:
        """
        Retrieve context with RAG and ask the model for one indicator.

        Returns:
          {
            "result": {"name": "...", "value": "...", "explanation": "..."},
            "sources": [...],
            "chunks": [...],
            "raw_answer": "..."
          }
        """
        query_terms = self.build_query_terms(indicator, op)
        question_for_retrieval = " ".join(query_terms)

        if top_k is None and ce_keep_k is None:
            hits = self.rag_system.search(question_for_retrieval)
        else:
            hits = self.rag_system.search(question_for_retrieval, top_k=top_k, ce_keep_k=ce_keep_k)

        messages, included_indices = self._build_messages(indicator, op, hits)

        if print_prompt:
            print("----- MESSAGES START -----")
            for m in messages:
                print(f"{m['role'].upper()}: {m['content']}\n")
            print("----- MESSAGES END -----\n")

        common_args = dict(
            model=self.model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            n=1,
            reasoning_effort=reasoning_effort,
        )

        if stream:
            collected: _List[str] = []
            response = self.client.chat.completions.create(stream=True, **common_args)
            for chunk in response:
                choice = chunk.choices[0]
                delta = choice.delta
                token = delta.content or ""
                if token:
                    collected.append(token)
                    print(token, end="", flush=True)
            raw_answer = "".join(collected).strip()
            print()
        else:
            response = self.client.chat.completions.create(stream=False, **common_args)
            raw_answer = (response.choices[0].message.content or "").strip()

        sources = self._extract_sources(hits, included_indices)

        parsed = None
        parse_error = None
        try:
            parsed = json.loads(raw_answer)
        except Exception as e:
            parse_error = str(e)

        if parsed is None:
            result = {
                "name": str(indicator),
                "value": "",
                "explanation": f"Model output was not valid JSON. Parsing error: {parse_error}",
            }
        else:
            result = parsed

        self.chat_history.append({"role": "user", "content": f"Indicator requested: {indicator}"})
        self.chat_history.append({"role": "assistant", "content": raw_answer})

        if print_sources:
            print("Sources:")
            for s in sources:
                print(f" • {s}")
            print()

        return {"result": result, "sources": sources, "chunks": hits, "raw_answer": raw_answer}
    
    @staticmethod
    def build_query_terms(indicator: str, op: InitialOperationInput) -> List[str]:
        terms: List[str] = []
        terms.extend(BASE_QUERIES[indicator])

        # Always add the user-selected discriminators (but only the *specific* ones)
        terms.extend(VLOS_MODE_TERMS[op.vlos_or_bvlos])
        terms.extend(GROUND_ENV_TERMS[op.ground_environment])
        terms.extend(AIRSPACE_TYPE_TERMS[op.airspace_type])
        terms.extend(MASS_TERMS[op.maximum_takeoff_mass_category])
        terms.extend(ALTITUDE_TERMS[op.maximum_altitude_category])

        # # Pathway-only extra hinting to hit the right STS/PDRA row fast
        # if indicator == "likely_regulatory_pathway":
        #     key = (
        #         op.vlos_or_bvlos,
        #         op.ground_environment,
        #         op.maximum_takeoff_mass_category,
        #         op.maximum_altitude_category,
        #     )
        #     terms.extend(PATHWAY_HINTS.get(key, []))

        # De-dup while preserving order
        seen = set()
        out = []
        for t in terms:
            t_norm = t.strip().lower()
            if not t_norm or t_norm in seen:
                continue
            seen.add(t_norm)
            out.append(t)
        return out


