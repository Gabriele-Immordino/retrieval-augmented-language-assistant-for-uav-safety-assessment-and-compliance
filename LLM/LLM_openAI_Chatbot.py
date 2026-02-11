import os
from typing import List, Dict, List as _List, Optional, Tuple

from openai import OpenAI
from RAG.ragv2 import RAG


class LLMChatbot:
    def __init__(
        self,
        model_name: str = "gpt-oss:20b",
        rag_system: Optional[RAG] = None,
        system_rules_path: str = "LLM/system_rules_chatbot.txt",
        developer_content_path: str = "LLM/developer_content.txt",
        base_url:str="http://localhost:11434/v1",
        api_key:str="ollama",
    ) -> None:
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.model_name = model_name
        self.rag_system = rag_system
        self.system_rules = self.load_system_rules(system_rules_path)
        self.developer_content = self.load_system_rules(developer_content_path)

        # Simple tokens that replace the previous LLMWrapper fields
        self.no_context_token = "NO_CONTEXT_AVAILABLE"

    # -------- System rules --------

    def load_system_rules(self, file_path: str) -> str:
        """Load system rules from a text file."""
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"System rules file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            rules = f.read()
        return rules

    # -------- Context handling --------

    def format_context(
        self,
        chunks: List[Dict],
        max_chars: int = 12000
    ) -> Tuple[str, _List[int]]:
        """Combine retrieved chunks into a single context string, limited by max_chars.

        Returns a tuple: (context_string, included_indices) where included_indices
        are chunk_index values of chunks included in the context.
        """

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

        ctx = "\n\n".join(parts)
        if not ctx or not ctx.strip():
            ctx = self.no_context_token

        return ctx.strip(), included_indices

    def extract_sources(
        self,
        retrieved: List[Dict],
        included_indices: List[int]
    ) -> List[str]:
        """Create source references aligned with context numbering."""

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

    # -------- Chat message building --------

    def build_messages_client(
        self,
        question: str,
        chat_history: List[Dict[str, str]],
        retrieved_sources: List[Dict[str, str]],
    ) -> Tuple[List[Dict[str, str]], List[int]]:
        """Build messages for OpenAI client from chat history and retrieved sources."""

        context_block, included_indices = self.format_context(retrieved_sources)
        retrieved_context = (
            "Use the context to answer the question. "
            f"Context:\n{context_block}\n\n"
            f"If the context is '{self.no_context_token}', answer from your general knowledge "
            "but say that no specific document context was available."
        )
        messages: _List[Dict[str, str]] = []
        # System message with rules and context
        messages.append({"role": "system", "content": self.system_rules})
        messages.append({"role": "developer", "content": self.developer_content})
        # Previous turns
        for msg in chat_history:
            role = msg["role"]
            content = str(msg.get("content", "")).strip()
            if not content:
                print("Empty content in chat history, skipping.")
                continue
            if role not in ("user", "assistant"):
                print(f"Unknown role '{role}' in chat history, skipping.")
                continue
            messages.append({"role": role, "content": content.strip()})
        # Current question and context
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": "I will read the context shared next, then answer based on that content."})
        messages.append({"role": "user", "content": retrieved_context})
        
        return messages, included_indices
        
    # -------- Main answer method --------

    def answer(
        self,
        question: str,
        chat_history: List[Dict[str, str]],
        max_new_tokens: int = 56000,
        top_k: int = None,
        ce_keep_k: int = None,
        temperature: float = 0.2,
        top_p: float = 0.9,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stream: bool = False,
        print_sources: bool = True,
        print_prompt: bool = False,
        reasoning_effort: str = "medium",
    ) -> Dict:
        """
        Unified generation entry point using Ollama gpt-oss through OpenAI client.

        Returns a dict with keys: {"answer", "sources", "chunks"}.
        """


        if self.rag_system is None:
            raise RuntimeError("rag_system is not set for LLMChatbot")

        if top_k is None and ce_keep_k is None:
            # Default retrieval for graph RAG
            hits = self.rag_system.search(
                question,
            )
        else:
            # Retrieve context for baseline RAG
            hits = self.rag_system.search(
                question,
                top_k=top_k,
                ce_keep_k=ce_keep_k,
            )

        # Build messages
        messages, included_indices = self.build_messages_client(question, chat_history, hits)

        if print_prompt:
            print("----- MESSAGES START -----")
            for m in messages:
                print(f"{m['role'].upper()}: {m['content']}\n")
            print("----- MESSAGES END -----\n")

        # Shared parameters for chat completion
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

        # Extract sources
        sources = self.extract_sources(hits, included_indices)
        # Optionally print sources
        if print_sources:
            print("Sources:")
            for src in sources:
                print(f" • {src}")
            print()

        # Call Ollama with or without streaming
        if stream:
            collected: _List[str] = []
            reasoning_collected: _List[str] = []

            print("Answer:", end=" ", flush=True)
            print("\n")

            response = self.client.chat.completions.create(
                stream=True,
                **common_args,
            )

            for chunk in response:
                choice = chunk.choices[0]
                delta = choice.delta

                # Normal text tokens
                token = delta.content or ""
                if token:
                    collected.append(token)
                    print(token, end="", flush=True)

                # Reasoning tokens inside the stream
                reasoning_piece = getattr(delta, "reasoning", None)
                if reasoning_piece:
                    reasoning_collected.append(reasoning_piece)

            print("\n")

            if reasoning_collected:
                print("Reasoning:", end=" ", flush=True)
                print("".join(reasoning_collected), end="", flush=True)
                print("\n")
            

            final_answer = "".join(collected).strip()

        else:
            response = self.client.chat.completions.create(
                stream=False,
                **common_args,
            )

            full_choice = response.choices[0]
            msg = full_choice.message

            reasoning_piece = getattr(msg, "reasoning", None)

            final_answer = (msg.content or "").strip()

            print("Answer:", final_answer, flush=True)

            if reasoning_piece:
                print("Reasoning:", reasoning_piece, flush=True)

        return {"answer": final_answer, "sources": sources, "chunks": hits, "reasoning": reasoning_piece if not stream else "".join(reasoning_collected) }
