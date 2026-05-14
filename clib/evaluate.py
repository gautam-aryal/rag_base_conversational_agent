from rouge_score import rouge_scorer
from bert_score import score as bert_score
import numpy as np
import pandas as pd
class InformationRetrievalEvaluator:
  # -------------------------
  # 3. Evaluation Metrics
  # -------------------------
  def precision_at_k(retrieved, relevant, k):
      retrieved_k = retrieved[:k]
      print("---------")
      print(retrieved_k)
      print(relevant)
      print("---------------")
      print(set(retrieved_k))
      print(set(set(relevant)))
      return len(set(retrieved_k) & set(relevant)) / k

  def recall_at_k(retrieved, relevant, k):
      retrieved_k = retrieved[:k]
      return len(set(retrieved_k) & set(relevant)) / len(relevant) if relevant else 0

  def f1_at_k(p, r):
      return (2 * p * r / (p + r)) if (p + r) > 0 else 0

  def reciprocal_rank(retrieved, relevant):
      for rank, doc_id in enumerate(retrieved, start=1):
          if doc_id in relevant:
              return 1.0 / rank
      return 0

  def average_precision(retrieved, relevant):
      score, hit = 0.0, 0
      for rank, doc_id in enumerate(retrieved, start=1):
          if doc_id in relevant:
              hit += 1
              score += hit / rank
      return score / len(relevant) if relevant else 0

  def ndcg_at_k(retrieved, relevant, k):
      dcg, idcg = 0.0, 0.0
      for i, doc_id in enumerate(retrieved[:k]):
          rel = 1 if doc_id in relevant else 0
          dcg += (2**rel - 1) / np.log2(i + 2)
      for i in range(min(len(relevant), k)):
          idcg += (2**1 - 1) / np.log2(i + 2)
      return dcg / idcg if idcg > 0 else 0



class GeneratorEvaluator:
  # -------------------------
  # Evaluation Metrics
  # -------------------------
  def __init__(self):
      try:
          self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
          self.bert_score = bert_score
      except ImportError:
          raise ImportError("Please install required packages: `pip install rouge_score bert_score`")

  def calculate_rouge(self, generated_texts, reference_texts):
      if len(generated_texts) != len(reference_texts):
          raise ValueError("Number of generated and reference texts must match")

      rouge_scores = {'rouge1': {'precision': 0, 'recall': 0, 'f1': 0},
                    'rouge2': {'precision': 0, 'recall': 0, 'f1': 0},
                    'rougeL': {'precision': 0, 'recall': 0, 'f1': 0}}

      n = len(generated_texts)
      for gen, ref in zip(generated_texts, reference_texts):
          scores = self.rouge_scorer.score(ref, gen)
          for metric in rouge_scores:
              rouge_scores[metric]['precision'] += scores[metric].precision
              rouge_scores[metric]['recall'] += scores[metric].recall
              rouge_scores[metric]['f1'] += scores[metric].fmeasure

      # Average the scores
      for metric in rouge_scores:
          rouge_scores[metric]['precision'] /= n
          rouge_scores[metric]['recall'] /= n
          rouge_scores[metric]['f1'] /= n

      return rouge_scores

  def calculate_bert_score(self, generated_texts, reference_texts, lang="en", model_type="roberta-large"):
      if len(generated_texts) != len(reference_texts):
          raise ValueError("Number of generated and reference texts must match")

      P, R, F1 = self.bert_score(generated_texts, reference_texts, lang=lang, model_type=model_type)

      return {
          'precision': P.mean().item(),
          'recall': R.mean().item(),
          'f1': F1.mean().item()
      }

  def calculate_bert_score_nepali(self, generated_texts, reference_texts, lang="ne", model_type="facebook/mbart-large-50"):
    if len(generated_texts) != len(reference_texts):
      raise ValueError("Number of generated and reference texts must match")

    P, R, F1 = self.bert_score(generated_texts, reference_texts, lang=lang, model_type=model_type)

    return {
    'precision': P.mean().item(),
    'recall': R.mean().item(),
    'f1': F1.mean().item()
    }


from tabulate import tabulate

class Evaluation:
  def __init__(self, df, k = 3, chunk_type = 'fix'):
    self.df = df
    self.k = k
    self.chunk_type = chunk_type

  def actual_ids(self):
    if self.chunk_type == 'fix':
      return self.df['actual_fix_chunks_ids'].tolist()
    elif self.chunk_type == 'overlapping':
      print(self.df['actual_overlapping_chunks_ids'].tolist())
      return self.df['actual_overlapping_chunks_ids'].tolist()
    
    elif self.chunk_type == 'semantic':
      return self.df['actual_semantic_chunks_ids'].tolist()
    elif self.chunk_type == 'recursive':
      return self.df['actual_recursive_chunk_ids'].tolist()

  def eval_retrieval(self):

    precision_at_k = InformationRetrievalEvaluator.precision_at_k(
      retrieved = self.df['retrived_ids'].tolist(),
      relevant = self.actual_ids(),
      k = self.k
    )

    recall_at_k = InformationRetrievalEvaluator.recall_at_k(
      retrieved = self.df['retrived_ids'].tolist(),
      relevant = self.actual_ids(),
      k = self.k
    )

    f1_at_k = InformationRetrievalEvaluator.f1_at_k(
      precision_at_k,
      recall_at_k
    )

    reciprocal_rank = InformationRetrievalEvaluator.reciprocal_rank(
      retrieved = self.df['retrived_ids'].tolist(),
      relevant = self.actual_ids()
    )

    average_precision = InformationRetrievalEvaluator.average_precision(
      retrieved = self.df['retrived_ids'].tolist(),
      relevant = self.actual_ids(),
      )

    ndcg_at_k = InformationRetrievalEvaluator.ndcg_at_k(
      retrieved = self.df['retrived_ids'].tolist(),
      relevant = self.actual_ids(),
      k = self.k
    )

    data = [
        [1, f"Precision@{self.k}", precision_at_k],
        [2, f"Recall@{self.k}", recall_at_k],
        [3, f"F1@{self.k}", f1_at_k],
        [4, "Reciprocal Rank", reciprocal_rank],
        [5, "Average Precsision",  average_precision],
        [6, f"NDCG@{self.k}",  ndcg_at_k]
    ]

    headers = ["SN", "Name", "Score"]

    print(tabulate(data, headers=headers, tablefmt="grid"))

  def eval_generator(self):
    generator_evaluator = GeneratorEvaluator()
    rouge_scores = generator_evaluator.calculate_rouge(
      generated_texts = self.df['generated_texts'].tolist(),
      reference_texts = self.df['reference_texts'].tolist()
    )
    bert_score = generator_evaluator.calculate_bert_score(
      generated_texts = self.df['generated_texts'].tolist(),
      reference_texts = self.df['reference_texts'].tolist()
    )

    # bert_score_nepali = generator_evaluator.calculate_bert_score_nepali(
    #   generated_texts = self.df['generated_texts'].tolist(),
    #   reference_texts = self.df['reference_texts'].tolist()
    # )

    data = [
        [1, "Rouge1", rouge_scores['rouge1']['precision'], rouge_scores['rouge1']['recall'], rouge_scores['rouge1']['f1']],
        [2, "Rouge2", rouge_scores['rouge2']['precision'], rouge_scores['rouge2']['recall'], rouge_scores['rouge2']['f1']],
        [3, "RougeL", rouge_scores['rougeL']['precision'], rouge_scores['rougeL']['recall'], rouge_scores['rougeL']['f1']],
        [4, "Bert Score", bert_score['precision'], bert_score['recall'], bert_score['f1']],
    ]

    headers = ["SN", "Name", "Precision", "Recall", "F1"]

    print(tabulate(data, headers=headers, tablefmt="grid"))


from typing import List, Union

class GtmInformationRetrievalEvaluator:
    """
    Evaluate retrieval results (Precision@K, Recall@K, F1@K) for RAG systems.

    Works with:
      - retrieved: list[str]  (each like "0,4,1")
      - reference: list[Union[str, int]]  (each may be "1,2,3" or 3)
    """

    def __init__(self, k: int = 3):
        self.k = k

    def _parse_retrieved(self, retrieved: List[str]) -> List[List[str]]:
        """
        Convert retrieved list of 'comma-separated strings' into list of chunk ID lists.
        Example: ['0,4,1', '343,338,344'] → [['0','4','1'], ['343','338','344']]
        """
        parsed = []
        for r in retrieved:
            ids = [x.strip() for x in str(r).split(",") if x.strip()]
            parsed.append(ids)
        return parsed

    def _parse_references(self, references: List[Union[str, int]]) -> List[List[str]]:
        """
        Convert mixed-type references (int or comma-separated string) into uniform list of lists.
        Example: [0, '0,1', '1,2,3'] → [['0'], ['0','1'], ['1','2','3']]
        """
        parsed = []
        for ref in references:
            if isinstance(ref, int):
                parsed.append([str(ref)])
            elif isinstance(ref, float):
              parsed.append([str(ref)])
            elif isinstance(ref, str):
                ids = [x.strip() for x in ref.split(",") if x.strip()]
                parsed.append(ids)
            else:
                raise ValueError(f"Unsupported reference type: {type(ref)}")
        return parsed

    def precision_at_k(self, retrieved_k: List[str], relevant: List[str]) -> float:
        """Compute Precision@K = (# relevant retrieved) / K"""
        retrieved_set = set(retrieved_k[:self.k])
        relevant_set = set(relevant)
        if self.k == 0:
            return 0.0
        return len(retrieved_set & relevant_set) / self.k

    def recall_at_k(self, retrieved_k: List[str], relevant: List[str]) -> float:
        """Compute Recall@K = (# relevant retrieved) / (# relevant total)"""
        retrieved_set = set(retrieved_k[:self.k])
        relevant_set = set(relevant)
        if len(relevant_set) == 0:
            return 0.0
        return len(retrieved_set & relevant_set) / len(relevant_set)

    def f1_at_k(self, precision: float, recall: float) -> float:
        """Compute F1@K"""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def evaluate(self, retrieved: List[str], references: List[Union[str, int]]) -> dict:
        """Compute mean Precision@K, Recall@K, and F1@K across all samples"""
        retrieved_list = self._parse_retrieved(retrieved)
        reference_list = self._parse_references(references)

        precisions, recalls, f1s = [], [], []

        for ret, ref in zip(retrieved_list, reference_list):
            p = self.precision_at_k(ret, ref)
            r = self.recall_at_k(ret, ref)
            f1 = self.f1_at_k(p, r)
            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)

        # return {
        #     f"Precision@{self.k}": sum(precisions) / len(precisions),
        #     f"Recall@{self.k}": sum(recalls) / len(recalls),
        #     f"F1@{self.k}": sum(f1s) / len(f1s)
        # }

        return {
            f"K": self.k,
            f"Precision": sum(precisions) / len(precisions),
            f"Recall": sum(recalls) / len(recalls),
            f"F1": sum(f1s) / len(f1s)
        }

    def actual_ids(self, df, chunk_type):
      if chunk_type == 'fix':
        return df['actual_fix_chunks_ids'].tolist()
      elif chunk_type == 'overlapping':
        return df['actual_overlapping_chunks_ids'].tolist()
      elif chunk_type == 'semantic':
        return df['actual_semantic_chunks_ids'].tolist()
      elif chunk_type == 'recursive':
        return df['actual_recursive_chunk_ids'].tolist()

    def eval_retrieval(self, df, chunk_type = 'fix', k = 3):
      self.k = k
      return self.evaluate(
        df['retrived_ids'].tolist(),
        self.actual_ids(df, chunk_type)
      )

