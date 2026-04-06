"""
Perplexity API — research с LLM синтезом (sonar model) + параллельные запросы.
Использование: для market research, сбора данных по проектам.
"""
import requests
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import PERPLEXITY_API_KEY, PERPLEXITY_MAX_TOKENS_PAGE


def research(query: str, model: str = "sonar", max_tokens: int = None) -> dict:
    """
    Research через Perplexity Chat Completions API (sonar model).
    Возвращает: { query, answer, citations, cost }
    """
    r = requests.post(
        "https://api.perplexity.ai/chat/completions",
        headers={
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": query}],
            "max_tokens": max_tokens or PERPLEXITY_MAX_TOKENS_PAGE,
        },
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    return {
        "query": query,
        "answer": data["choices"][0]["message"]["content"],
        "citations": data.get("citations", []),
        "cost": data.get("usage", {}).get("cost", {}).get("total_cost", 0),
    }


def research_parallel(queries: list[str], model: str = "sonar", max_workers: int = 5) -> list[dict]:
    """
    Параллельный research по списку запросов.
    Возвращает список результатов в том же порядке что queries.
    """
    results = [None] * len(queries)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(research, q, model): i for i, q in enumerate(queries)}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
                print(f"[pplx] ✓ '{queries[idx][:60]}'")
            except Exception as e:
                print(f"[pplx] ✗ '{queries[idx][:60]}': {e}")
                results[idx] = {"query": queries[idx], "answer": f"ERROR: {e}", "citations": [], "cost": 0}
    return results


def save_results(data: list | dict, path: str):
    """Сохраняет результаты в JSON файл."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[pplx] Saved → {path}")


if __name__ == "__main__":
    # CLI: python perplexity_search.py "query1" "query2" ...
    if len(sys.argv) > 1:
        queries = sys.argv[1:]
        print(f"[pplx] Running {len(queries)} parallel queries...\n")
        results = research_parallel(queries)
        for res in results:
            print(f"\n{'='*60}")
            print(f"Q: {res['query']}")
            print(f"A: {res['answer'][:500]}...")
            if res['citations']:
                print(f"Sources: {res['citations'][0]}")
        total_cost = sum(r['cost'] for r in results)
        print(f"\n[pplx] Total cost: ${total_cost:.4f}")
    else:
        # Быстрый тест
        r = research("NVIDIA A100 80GB cloud GPU price per hour 2026")
        print(r["answer"][:300])
        print(f"Cost: ${r['cost']:.4f}")
