import argparse
import json
import os
from pathlib import Path
from typing import Any
from pydantic import BaseModel, TypeAdapter
from dataclasses import dataclass, asdict
from tqdm import tqdm
from datetime import datetime
from retrieval.retriever import Retriever, RetrieverConfig
from indexing.embeddings import embedder
from generation.formatters import get_ranks

class TestCase(BaseModel):
    id: int
    question: str
    related_article: int

@dataclass
class AnswerStats:
    question_id: int
    desired_article: int
    retrieval_answer: list[int]
    is_answered: bool
    rank: int | None
    precision_at_1: float
    precision_at_3: float
    precision_at_5: float
    reciprocal_rank: float 
    # dcg: float | None
    # idcg: float | None
    # ndcg: float | None

@dataclass
class OverallReport:
    number_of_questions: int
    number_of_answered_questions: int
    overall_answer_accuracy: float
    mean_precision_at_1: float
    mean_precision_at_3: float
    mean_precision_at_5: float
    mean_reciprocal_rank: float 
    

def load_dataset(dataset_path: Path) -> list[TestCase]:
    """For loading and validating dataset"""
    raw_dataset = [] 
    with open(dataset_path, 'r', encoding="utf-8") as f:
        raw_dataset = json.load(f)
    print(f"Dataset with {len(raw_dataset)} questions loaded.")

    try:
        test_cases = TypeAdapter(list[TestCase]).validate_python(raw_dataset)
        print("Test cases validated")
    except:
        print("Failed at validating test cases.")
        raise ValueError
     
    return test_cases

def evaluate(test_cases: list[TestCase], retriever) -> list[AnswerStats]:
    """Evaluation per each question"""
    all_answer_stats = []
    with tqdm(total=len(test_cases), desc="Evaluating") as pbar:
        for test_case in test_cases:
            ranks = get_ranks(retriever.retrieve(test_case.question))
            answer = test_case.related_article
            all_answer_stats.append(
                AnswerStats(
                    question_id = test_case.id,
                    desired_article = test_case.related_article,
                    retrieval_answer = ranks,
                    is_answered = True if answer in ranks else False,
                    rank = ranks.index(answer) + 1 if answer in ranks else None,
                    precision_at_1 = 1 if ranks[0] == answer else 0.0,
                    precision_at_3 = ranks[:3].count(answer) / 3.0,
                    precision_at_5 = ranks[:5].count(answer) / 5.0,
                    reciprocal_rank = 1 / (ranks.index(answer) + 1) if answer in ranks else 0.0
                    
                )
            )
            pbar.update(1)
    return all_answer_stats

def generate_overall_report(answer_stats: list[dict[str, Any]]) -> OverallReport:
    aggregated_answer_stats_dict = {}
    for answer_stat in answer_stats:
        for key, value in answer_stat.items():
            if key not in aggregated_answer_stats_dict:
                aggregated_answer_stats_dict[key] = []
            aggregated_answer_stats_dict[key].append(value)
    q_num = len(answer_stats)
    a_num = sum(aggregated_answer_stats_dict["is_answered"])
    return OverallReport(
        number_of_questions = q_num,
        number_of_answered_questions = a_num,
        overall_answer_accuracy = a_num / q_num,
        mean_precision_at_1 = sum(aggregated_answer_stats_dict["precision_at_1"]) / q_num,
        mean_precision_at_3 = sum(aggregated_answer_stats_dict["precision_at_3"]) / q_num,
        mean_precision_at_5 = sum(aggregated_answer_stats_dict["precision_at_5"]) / q_num,
        mean_reciprocal_rank =  sum(aggregated_answer_stats_dict["reciprocal_rank"]) / q_num
    )

def generate_all_reports(all_answer_stats: list[AnswerStats], output_path: Path): #, categories: list[tuple[int,int]] | None
    """ Generate all overall reports and save"""
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_path, f"RetrievalEval_{TIMESTAMP}")
    os.makedirs(output_path, exist_ok=True)

    all_answer_stats_asdict = []
    for answer_stat in all_answer_stats:
        all_answer_stats_asdict.append(asdict(answer_stat))

    with open(os.path.join(output_path, f'all_answer_stats.json'), 'w') as fp:
        json.dump(all_answer_stats_asdict, fp, indent=4)

    main_report = generate_overall_report(all_answer_stats_asdict)
    with open(os.path.join(output_path, f'main_report.json'), 'w') as fp:
        json.dump(asdict(main_report), fp, indent=4)
    
    # if categories:
    #     for idx, category in enumerate(categories):
    #         q_idx = 0
    #         category_report = generate_overall_report(all_answer_stats[])
    #         with open(os.path.join(output_path, f'category_{idx+1}_report.json'), 'w') as fp:
    #             json.dump(category, fp, indent=4)

        
def main():
    # parse args
    parser = argparse.ArgumentParser(description="Evaluating retriever accuracy")
    parser.add_argument("--dataset-path", required=True, type=Path, help="Path to the retrieval evaluation dataset")
    parser.add_argument("--output-directory", default="../../data/eval/results/" ,type=Path, help="Path to the retrieval evaluation dataset")
    args = parser.parse_args()
    test_cases = load_dataset(args.dataset_path)
    print("Booting up retriever...")
    retriever = Retriever.from_existing_collection(
        embedding=embedder,
        config=RetrieverConfig(
            collection_name="direct-taxes-law",
            qdrant_url="http://localhost:6333",
            # qdrant_api_key=None,
            top_k=5,
        )
    )
    print("Retriever initialized.")
    all_answer_stats = evaluate(test_cases, retriever)
    generate_all_reports(all_answer_stats, output_path=args.output_directory)


if __name__ == "__main__":
    """Evaluating retriever accuracy"""
    main()