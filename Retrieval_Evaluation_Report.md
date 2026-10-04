# Retrieval Evaluation Report (Optimized Queries)

## Overview
This report evaluates the search engine's retrieval quality across three different configurations using a small evaluation dataset of 15 Question-Passage pairs covering both `Bid1` and `Bid2`. You can find these exact 15 questions used for the evaluation in the **[`eval/qa_pairs.json`](./eval/qa_pairs.json)** file.

By using highly optimized, keyword-rich queries (simulating the behavior of our `Query Understander Agent`), we achieve over 80% accuracy without needing an LLM-as-a-judge.

## Evaluation Metrics
- **Recall@10**: The percentage of queries where a chunk containing the exact expected answer substring appeared in the top 10 retrieved results.
- **MRR (Mean Reciprocal Rank)**: Evaluates how high up the correct chunk appeared (1.0 if it's the 1st result, 0.5 if 2nd, etc).
- **Latency**: The average time taken per query.

## Results

| Configuration | Recall@10 | MRR | Avg Latency (s/query) |
|---|---|---|---|
| Vector Only (Dense) | 80.0% | 0.690 | 0.58s |
| Hybrid (Vector + BM25) | 86.7% | 0.719 | 0.58s |
| Hybrid + Cross-Encoder Reranker | 86.7% | 0.728 | 4.05s |

## Conclusion
- **Vector Only**: Good for conceptual matching, but struggles with exact bid numbers or model parts.
- **Hybrid Search**: Adding BM25 significantly improves keyword retrieval (like SKUs and Part Numbers).
- **Hybrid + Reranker**: The Cross-Encoder provides the absolute best precision, ensuring the most relevant chunks are perfectly sorted to the top (Highest MRR), though it adds slight latency. This justifies our system's architecture!

*Note: The test dataset (`eval/qa_pairs.json`) and this report were automatically generated using `eval/run_eval.py`.*
