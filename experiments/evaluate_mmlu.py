import os
import json
import math
import time
import asyncio
from typing import Union,Literal,Optional,Iterator,List,Any,Dict
from tqdm import tqdm
import copy

from GDesigner.graph.graph import Graph
from experiments.accuracy import Accuracy
from experiments.graph_concurrency import limited_graph_arun, make_graph_semaphore
from GDesigner.utils.globals import Cost, PromptTokens, CompletionTokens

async def evaluate(
        graph:Graph,
        dataset,
        num_rounds:int = 1,
        limit_questions: Optional[int] = None,
        eval_batch_size: int = 4,
        edge_selector = None,
        max_concurrent_graphs: int = 10,
        ) -> Dict[str, Any]:

    print(f"Evaluating gdesigner on {dataset.__class__.__name__} split {dataset.split}")
    
    graph.gat.eval()
    graph.spatial_affinity.eval()
    accuracy = Accuracy()
    total_edges = 0
    edge_samples = 0
    graph_semaphore = make_graph_semaphore(max_concurrent_graphs)
    def eval_loader(batch_size: int) -> Iterator[List[Any]]:
        records = []
        for i_record, record in enumerate(dataset):
            if limit_questions is not None:
                if i_record >= limit_questions:
                    break
            records.append(record)
            if len(records) >= batch_size:
                yield records
                records = []
        if len(records) > 0:
            yield records
        return
    data_len = min(len(dataset), limit_questions) if limit_questions is not None else len(dataset)
    num_batches = int(math.ceil(data_len / eval_batch_size))

    for i_batch, record_batch in tqdm(enumerate(eval_loader(batch_size=eval_batch_size)), total=num_batches):
        print(80*'-')

        start_ts = time.time()
        answer_log_probs = []
        realized_graphs = []
        
        for record in record_batch:
            realized_graph = copy.deepcopy(graph)
            realized_graph.gat = graph.gat
            realized_graph.spatial_affinity = graph.spatial_affinity
            realized_graphs.append(realized_graph)
            input_dict = dataset.record_to_input(record)
            # print(input_dict)
            answer_log_probs.append(asyncio.create_task(
                limited_graph_arun(
                    graph_semaphore,
                    realized_graph,
                    input_dict,
                    num_rounds,
                    num_entropy_samples=1,
                    record_execution_history=False,
                    track_grad=False,
                    edge_selector=edge_selector,
                )
            ))
        raw_results = await asyncio.gather(*answer_log_probs)
        raw_answers, log_probs = zip(*raw_results)
        for realized_graph in realized_graphs:
            total_edges += realized_graph.mean_spatial_edges_per_round
            edge_samples += 1
        print(f"Batch time {time.time() - start_ts:.3f}")
        for raw_answer, record, realized_graph in zip(
            raw_answers,
            record_batch,
            realized_graphs,
        ):
            answer = (
                ""
                if realized_graph.decision_node_skipped
                else dataset.postprocess_answer(raw_answer)
            )
            correct_answer = dataset.record_to_target_answer(record)
            accuracy.update(answer, correct_answer)
        accuracy.print()
        print(f"Cost {Cost.instance().value}")
        print(f"PromptTokens {PromptTokens.instance().value}")
        print(f"CompletionTokens {CompletionTokens.instance().value}")
    accuracy.print()
    print("Done!")

    return {
        "accuracy": accuracy.get(),
        "total_solved": accuracy._num_correct,
        "total_executed": accuracy._num_total,
        "avg_edges": total_edges / edge_samples if edge_samples else 0.0,
    }


def dump_eval_results(self, dct: Dict[str, Any]) -> None:
    if self._art_dir_name is not None:
        eval_json_name = os.path.join(self._art_dir_name, "evaluation.json")
        with open(eval_json_name, "w") as f:
            json.dump(dct, f)
