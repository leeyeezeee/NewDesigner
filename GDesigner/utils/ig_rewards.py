import asyncio
from typing import Any, Dict, Iterable, List, Tuple

from GDesigner.llm.gpt_chat import EmptyChatCompletionError
from GDesigner.utils.ig_scorer import FinalAnswerScorer, TargetSpec, edge_key


def _flatten_outputs(results: Iterable[Any]) -> List[Any]:
    outputs: List[Any] = []
    for result in results:
        if isinstance(result, list):
            outputs.extend(result)
        else:
            outputs.append(result)
    return outputs


async def _sample_node_outputs(
    node,
    input_data: Any,
    spatial_info: Dict[str, Any],
    temporal_info: Dict[str, Any],
) -> List[Any]:
    result = await node._async_execute(input_data, spatial_info, temporal_info)
    return _flatten_outputs([result])


def _require_nonblank_outputs(
    outputs: Iterable[Any],
    *,
    key: str,
    round_idx: int,
    source_id: str,
    target_id: str,
    phase: str,
) -> List[Any]:
    output_list = list(outputs)
    if output_list and all(str(output).strip() for output in output_list):
        return output_list
    blank_positions = [
        index
        for index, output in enumerate(output_list)
        if not str(output).strip()
    ]
    raise RuntimeError(
        "Edge IG received missing or blank candidate output. "
        f"edge_key={key!r}, round={round_idx}, source={source_id!r}, "
        f"target={target_id!r}, phase={phase!r}, "
        f"output_count={len(output_list)}, blank_positions={blank_positions}"
    )


def _realized_spatial_edge_occurrences(
    graph,
    histories: Dict[Tuple[str, int], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    occurrences: List[Dict[str, Any]] = []
    for edge_info in getattr(graph, "edge_log_probs", []):
        if edge_info.get("type") != "spatial":
            continue
        history_item = histories.get(
            (edge_info.get("target"), edge_info.get("round"))
        )
        if history_item is None:
            continue
        if edge_info.get("source") not in history_item.get("spatial_info", {}):
            continue
        occurrences.append(edge_info)
    return occurrences


async def compute_edge_information_gain(
    graph,
    input_data: Any,
    *,
    target_spec: TargetSpec,
    scorer: FinalAnswerScorer,
) -> Dict[str, Dict[str, Any]]:
    """Measure final-answer information gain for each realized spatial edge."""
    if not graph.edge_log_probs:
        return {}

    histories: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for node_id, node in graph.nodes.items():
        for history_item in node.execution_history:
            histories[(node_id, history_item["round"])] = history_item

    details: Dict[str, Dict[str, Any]] = {}
    after_outputs_cache: Dict[Tuple[str, int], List[Any]] = {}
    after_score_cache: Dict[Tuple[str, int], Any] = {}

    for edge_info in _realized_spatial_edge_occurrences(graph, histories):
        target_id = edge_info["target"]
        source_id = edge_info["source"]
        round_idx = edge_info["round"]
        key = edge_key(edge_info)
        history_item = histories.get((target_id, round_idx))
        target_node = graph.nodes.get(target_id)
        if history_item is None or target_node is None:
            continue

        spatial_info = {
            node_id: dict(info)
            for node_id, info in history_item.get("spatial_info", {}).items()
        }
        temporal_info = {
            node_id: dict(info)
            for node_id, info in history_item.get("temporal_info", {}).items()
        }
        if source_id not in spatial_info:
            continue
        before_spatial_info = dict(spatial_info)
        before_spatial_info.pop(source_id, None)

        before_response_empty = False
        try:
            before_outputs = await _sample_node_outputs(
                target_node,
                input_data,
                before_spatial_info,
                temporal_info,
            )
        except EmptyChatCompletionError:
            before_outputs = []
            before_response_empty = True
        except Exception as exc:
            raise RuntimeError(
                "Edge IG counterfactual generation failed. "
                f"edge_key={key!r}, round={round_idx}, source={source_id!r}, "
                f"target={target_id!r}, phase='before'"
            ) from exc
        before_outputs = [
            output for output in before_outputs if str(output).strip()
        ]
        before_response_empty = before_response_empty or not before_outputs

        after_cache_key = (target_id, round_idx)
        if after_cache_key in after_outputs_cache:
            after_outputs = after_outputs_cache[after_cache_key]
        else:
            after_outputs = _require_nonblank_outputs(
                history_item.get("outputs", []),
                key=key,
                round_idx=round_idx,
                source_id=source_id,
                target_id=target_id,
                phase="after",
            )
            after_outputs_cache[after_cache_key] = list(after_outputs)

        target_is_final = target_node is graph.decision_node
        if target_is_final and target_spec.mode != "execution":
            before_score_task = scorer.teacher_answer_logprob(
                target_node,
                input_data,
                before_spatial_info,
                temporal_info,
                target_spec,
            )
        elif target_is_final:
            before_score_task = scorer.final_agent_execution_score(
                target_node,
                input_data,
                [] if before_response_empty else before_outputs,
                target_spec,
            )
        elif target_spec.mode != "execution":
            if before_response_empty:
                before_score_task = scorer.teacher_answer_logprob(
                    graph.decision_node,
                    input_data,
                    {},
                    {},
                    target_spec,
                )
            else:
                before_score_task = scorer.final_agent_teacher_answer_logprob(
                    graph.decision_node,
                    input_data,
                    before_outputs,
                    target_spec,
                    candidate_id=target_id,
                    candidate_role=getattr(target_node, "role", "Candidate"),
                )
        else:
            before_score_task = scorer.final_agent_execution_score(
                graph.decision_node,
                input_data,
                before_outputs,
                target_spec,
                candidate_id=target_id,
                candidate_role=getattr(target_node, "role", "Candidate"),
            )

        if after_cache_key in after_score_cache:
            before_score = await before_score_task
            after_score = after_score_cache[after_cache_key]
        else:
            if target_is_final and target_spec.mode != "execution":
                after_score_task = scorer.teacher_answer_logprob(
                    target_node,
                    input_data,
                    spatial_info,
                    temporal_info,
                    target_spec,
                )
            elif target_is_final:
                after_score_task = scorer.score_outputs(
                    target_node,
                    input_data,
                    after_outputs,
                    target_spec,
                )
            elif target_spec.mode != "execution":
                after_score_task = scorer.final_agent_teacher_answer_logprob(
                    graph.decision_node,
                    input_data,
                    after_outputs,
                    target_spec,
                    candidate_id=target_id,
                    candidate_role=getattr(target_node, "role", "Candidate"),
                )
            else:
                after_score_task = scorer.final_agent_execution_score(
                    graph.decision_node,
                    input_data,
                    after_outputs,
                    target_spec,
                    candidate_id=target_id,
                    candidate_role=getattr(target_node, "role", "Candidate"),
                )
            before_score, after_score = await asyncio.gather(
                before_score_task,
                after_score_task,
            )
            after_score_cache[after_cache_key] = after_score

        before_answer_score = float(before_score.score)
        after_answer_score = float(after_score.score)
        ig_gain = after_answer_score - before_answer_score
        details[key] = {
            "type": "spatial",
            "round": round_idx,
            "source": source_id,
            "target": target_id,
            "raw_ig_gain": ig_gain,
            "round_ig_gain": ig_gain,
            "ig_gain": ig_gain,
            "before_answer_score": before_answer_score,
            "after_answer_score": after_answer_score,
            "before_target_message_present": not before_response_empty,
            "before_counterfactual_silence": before_response_empty,
            "before_answer_details": before_score.details,
            "after_answer_details": after_score.details,
        }
        if target_spec.mode != "execution":
            details[key].update({
                "ig_mode": "final_agent_teacher_logprob_diff",
                "before_teacher_logprob": before_answer_score,
                "after_teacher_logprob": after_answer_score,
                "teacher_forcing_agent": "final_agent",
                "teacher_forcing_context": (
                    "counterfactual_no_candidate_vs_local_candidate"
                    if before_response_empty
                    else "local_candidate_output"
                ),
                "teacher_forcing_candidate_role": getattr(target_node, "role", ""),
            })
        else:
            details[key].update({
                "ig_mode": "final_agent_execution_score_diff",
                "scoring_agent": "final_agent",
                "execution_candidate_role": getattr(target_node, "role", ""),
            })

    return details
