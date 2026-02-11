import json
from typing import Any, Dict, List

from openai import OpenAI


class QueryGenerator:
    """
    Wrapper around the llm model via OpenAI-compatible API
    that turns text chunks into concise queries paired with reasoning levels.
    """

    def __init__(
        self,
        model_name: str = "qwen3:8b",
        base_url:str="http://localhost:11434/v1",
        api_key: str = "ollama",
        max_queries: int = 3,
    ) -> None:
        """
        Initialize the QueryGenerator.
        """
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.model_name = model_name
        self.max_queries = max_queries

    def _system_message(self) -> str:
        """Build the system prompt for query generation."""
        return (
            "You generate queries from a provided query.\n\n"
            "First decide if the query can be answered as a single focused question.\n"
            "If yes, return exactly one query.\n"
            "Split into multiple queries only if the query clearly contains independent questions.\n"
            "Avoid semantically similar queries and use the minimum number needed to cover the intent.\n\n"
            f"Return at most {self.max_queries} queries, ordered by usefulness.\n"
            "Each query object must include 'query' as a string and 'reasoning_level' as one of 'none', 'low', 'medium', 'high'.\n\n"
            "Output a single valid JSON object with one key named 'queries' containing the list of query objects.\n"
            "Do not include any text before or after the JSON."
        )

    def _build_user_message(self, query: str) -> str:
        """Build the user prompt for query generation."""
        if not query:
            raise ValueError(f"Query is empty. Query: {query}")

        return (
            "Generate targeted queries that help understand, verify, or expand on the following query.\n\n"
            "Query text:\n"
            f"\"\"\"{query}\"\"\"\n\n"
            "Return only the JSON object described above."
        )

    def _generate(self, query: str) -> str:
        """
        Call the gpt-oss-20b model via OpenAI-compatible API.

        Args:
            query: The user query to preprocess.

        Returns:
            The raw text output from the model.

        Raises:
            Exception: If the API call fails after retries.
        """
        messages = [
            {"role": "system", "content": self._system_message()},
            {"role": "user", "content": self._build_user_message(query)},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=56000,
                temperature=0.2,
                top_p=0.9,
            )
            text = response.choices[0].message.content or ""
            return text.strip()

        except Exception as exc:
            print(f"Tried to connect to model API at {self.client.base_url}, {self.model_name} but failed: {exc}")
            raise RuntimeError(f"Error calling model API: {exc}") from exc

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        """
        Extract JSON from model output, handling various formats.

        Args:
            text: The raw text output from the model.

        Returns:
            A dictionary with a 'queries' key containing the list of query objects.

        Raises:
            ValueError: If no valid JSON can be extracted.
        """
        s = text.strip()

        # Try direct parsing first
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            parsed = None

        if isinstance(parsed, dict):
            return parsed

        if isinstance(parsed, list):
            return {"queries": parsed}

        # Try to find JSON array
        first_arr = s.find("[")
        last_arr = s.rfind("]")
        first_obj = s.find("{")
        last_obj = s.rfind("}")

        if first_arr != -1 and last_arr != -1 and last_arr > first_arr:
            arr_str = s[first_arr : last_arr + 1]
            try:
                return {"queries": json.loads(arr_str)}
            except json.JSONDecodeError:
                pass

        # Try to find JSON object
        if first_obj != -1 and last_obj != -1 and last_obj > first_obj:
            obj_str = s[first_obj : last_obj + 1]
            try:
                return json.loads(obj_str)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON object from model output: {obj_str!r}") from e

        raise ValueError(f"Could not find JSON in model output: {text!r}")

    def generate_queries(self, query: str) -> Dict[str, Any]:
        """
        Generate preprocessed queries from the input query.

        Args:
            query: The user query to preprocess.

        Returns:
            A dictionary with a 'queries' key containing a list of query objects.
            Each query object has 'query' (str) and 'reasoning_level' (str) keys.

        Raises:
            ValueError: If the query is empty or the model output is invalid.
        """
        raw_output = self._generate(query)

        data = self._extract_json(raw_output)

        queries = data.get("queries", [])
        if not isinstance(queries, list):
            raise ValueError(f"'queries' is not a list in model output: {data!r}")

        normalized_queries: List[Dict[str, str]] = []
        for entry in queries:
            if not isinstance(entry, dict):
                continue

            query_text = entry.get("query")
            level = entry.get("reasoning_level", "")

            if not isinstance(query_text, str):
                continue

            query_text = query_text.strip()
            if not query_text:
                continue

            level = level.strip() if isinstance(level, str) else ""
            normalized_queries.append(
                {
                    "query": query_text,
                    "reasoning_level": level,
                }
            )

            if len(normalized_queries) >= self.max_queries:
                break

        return {
            "queries": normalized_queries,
        }
    
