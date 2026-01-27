#!/usr/bin/env python3
import paper_dataset
import llm_filter
import json

BACKEND = "hf"

if __name__ == "__main__":
    real_tokenization = paper_dataset.load("data/real_tokenization.jsonl")

    # METRICS + TASKS
    print("=== METRIC + TASK EXTRACTION ===")
    all_answers_metrics_tasks = list(
        llm_filter.get_metrics_tasks(
            real_tokenization,
            backend=BACKEND,
            file="data/raw_answers/metrics_tasks_answer.txt",
        )
    )

    with open("data/raw_answers/metrics_tasks_answer.json", "wt") as f:
        json.dump(all_answers_metrics_tasks, f, indent=2)


    # UNITS
    print("=== UNIT EXTRACTION ===")
    all_answers_unit = list(
        llm_filter.get_units(
            real_tokenization, backend=BACKEND, file="data/raw_answers/unit_answer.txt"
        )
    )
    with open("data/raw_answers/unit_answer.json", "wt") as f:
        json.dump(all_answers_unit, f, indent=2)

    # METRICS
    print("=== METRIC EXTRACTION ===")
    all_answers_metrics = list(
        llm_filter.get_metrics(
            real_tokenization,
            backend=BACKEND,
            file="data/raw_answers/metric_answer.txt",
        )
    )

    with open("data/raw_answers/metric_answer.json", "wt") as f:
        json.dump(all_answers_metrics, f, indent=2)

    # MOTIVATIONS
    print("=== MOTIVATION EXTRACTION ===")
    all_answers_motivation = list(
        llm_filter.get_motivation(
            real_tokenization,
            backend=BACKEND,
            file="data/raw_answers/motivation_answer.txt",
        )
    )
    with open("data/raw_answers/motivation_answer.json", "wt") as f:
        json.dump(all_answers_motivation, f, indent=2)

    # LANGUAGES
    print("=== LANGUAGE EXTRACTION ===")
    all_answers_language = list(
        llm_filter.get_languages(
            real_tokenization,
            backend=BACKEND,
            file="data/raw_answers/language_answer.txt",
        )
    )
    with open("data/raw_answers/language_answer.json", "wt") as f:
        json.dump(all_answers_language, f, indent=2)
