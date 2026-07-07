import shortuuid
from typing import Any, List, Optional, Dict, Tuple
from abc import ABC
import numpy as np
import torch
import asyncio

import GDesigner.agents
import GDesigner.prompt
from GDesigner.graph.node import Node
from GDesigner.agents.agent_registry import AgentRegistry
from GDesigner.prompt.prompt_set_registry import PromptSetRegistry
from GDesigner.llm.profile_embedding import get_sentence_embedding
from GDesigner.gnn.gcn import GCN,MLP
from torch_geometric.utils import dense_to_sparse

_DECISION_NODE_MAX_TRIES = 5
_DECISION_NODE_TIMEOUT_SECONDS = 1200


class Graph(ABC):
    """
    A framework for managing and executing a network of nodes using a language model.

    This class enables the creation of a graph structure for processing and analyzing data. Each node
    in the graph can perform specific operations, allowing for complex data processing workflows.
    The graph supports integration with language models, making it suitable for tasks that require
    natural language processing capabilities.

    The communication of the node depends on the node.spatial_predecessors and node.spatial_successors.
    
    Attributes:
        domain (str): The domain for which this graph is used.
        llm_name (str): The name of the llm that used for processing within the nodes.
        nodes (dict): A collection of nodes, each identified by a unique UUID.

    Methods:
        build_graph(): Method to be implemented for constructing the graph structure.
        add_node(node): Adds a new node to the graph with a unique identifier.
        run(inputs, num_steps=10, single_agent=False): Executes the graph for a specified number of steps, processing provided inputs.
    """

    def __init__(self, 
                domain: str,
                llm_name: Optional[str],
                agent_names: List[str],
                decision_method: str,
                optimized_spatial:bool = False,
                initial_spatial_probability: float = 0.5,
                fixed_spatial_masks:List[List[int]] = None,
                optimized_temporal:bool = False,
                initial_temporal_probability: float = 0.5,
                fixed_temporal_masks:List[List[int]] = None,
                node_kwargs:List[Dict] = None,
                refine_rank:int = 4,
                edge_bias_scale: float = 0.5,
                ):
        
        if fixed_spatial_masks is None:
            fixed_spatial_masks = [[1 if i!=j else 0 for j in range(len(agent_names))] for i in range(len(agent_names))]
        if fixed_temporal_masks is None:
            fixed_temporal_masks = [[1 for j in range(len(agent_names))] for i in range(len(agent_names))]
        fixed_spatial_masks = torch.tensor(fixed_spatial_masks).view(-1)
        fixed_temporal_masks = torch.tensor(fixed_temporal_masks).view(-1)
        assert len(fixed_spatial_masks)==len(agent_names)*len(agent_names),"The fixed_spatial_masks doesn't match the number of agents"
        assert len(fixed_temporal_masks)==len(agent_names)*len(agent_names),"The fixed_temporal_masks doesn't match the number of agents"
        
        self.id:str = shortuuid.ShortUUID().random(length=4)
        self.domain:str = domain
        self.llm_name:str = llm_name
        self.agent_names:List[str] = agent_names
        self.optimized_spatial = optimized_spatial
        self.optimized_temporal = optimized_temporal
        self.decision_node:Node = AgentRegistry.get(decision_method, **{"domain":self.domain,"llm_name":self.llm_name})
        self.nodes:Dict[str,Node] = {}
        self.potential_spatial_edges:List[List[str, str]] = []
        self.potential_temporal_edges:List[List[str,str]] = []
        self.edge_log_probs:List[Dict[str, Any]] = []
        self.realized_edge_counts:List[int] = []
        self.node_kwargs = node_kwargs if node_kwargs is not None else [{} for _ in agent_names]
        self.anchor_spatial_matrix = fixed_spatial_masks.view(len(agent_names), len(agent_names)).float()
        self.refine_rank = min(max(1, int(refine_rank)), len(agent_names))
        self.edge_bias_scale = float(edge_bias_scale)
        self.refinement_weight = torch.nn.Parameter(
            torch.eye(self.refine_rank),
            requires_grad=optimized_spatial,
        )
        self.refinement_anchor_loss = torch.tensor(0.0)
        self.refinement_sparse_loss = torch.tensor(0.0)
        self.edge_bias_l2_loss = torch.tensor(0.0)
        self.spatial_edge_probabilities = None
        
        self.init_nodes() # add nodes to the self.nodes
        self.node_id_to_index = {node_id: idx for idx, node_id in enumerate(self.nodes)}
        self.init_potential_edges() # add potential edges to the self.potential_spatial/temporal_edges
        self.spatial_edge_bias = torch.nn.Parameter(
            torch.zeros(len(self.potential_spatial_edges)),
            requires_grad=optimized_spatial,
        )
        
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role_adj_matrix = self.construct_adj_matrix()
        self.features = self.construct_features()
        self.gcn = GCN(self.features.size(1)*2,16,self.features.size(1))
        self.mlp = MLP(384,16,16)

        init_spatial_logit = torch.log(torch.tensor(initial_spatial_probability / (1 - initial_spatial_probability))) if optimized_spatial else 10.0
        # self.spatial_logits = torch.nn.Parameter(torch.ones(len(self.potential_spatial_edges), requires_grad=optimized_spatial) * init_spatial_logit,
        #                                          requires_grad=optimized_spatial) # trainable edge logits
        self.spatial_masks = torch.nn.Parameter(fixed_spatial_masks,requires_grad=False)  # fixed edge masks

        init_temporal_logit = torch.log(torch.tensor(initial_temporal_probability / (1 - initial_temporal_probability))) if optimized_temporal else 10.0
        self.temporal_logits = torch.nn.Parameter(torch.ones(len(self.potential_temporal_edges), requires_grad=optimized_temporal) * init_temporal_logit,
                                                 requires_grad=optimized_temporal) # trainable edge logits
        self.temporal_masks = torch.nn.Parameter(fixed_temporal_masks,requires_grad=False)  # fixed edge masks

    def refinement_parameters(self) -> List[torch.nn.Parameter]:
        parameters = []
        if self.refinement_weight.requires_grad:
            parameters.append(self.refinement_weight)
        if self.spatial_edge_bias.requires_grad:
            parameters.append(self.spatial_edge_bias)
        return parameters
    
    def construct_adj_matrix(self):
        role_connect:List[Tuple[str,str]] = self.prompt_set.get_role_connection()
        num_nodes = self.num_nodes
        role_adj = torch.zeros((num_nodes,num_nodes))
        role_2_id = {}
        
        for edge in role_connect:
            in_role, out_role = edge
            role_2_id[in_role] = []
            role_2_id[out_role] = []
        for i, node_id in enumerate(self.nodes):
            role = self.nodes[node_id].role
            role_2_id[role].append(i)
            
        for edge in role_connect:
            in_role,out_role = edge
            in_ids = role_2_id[in_role]
            out_ids = role_2_id[out_role]
            for in_id in in_ids:
                for out_id in out_ids:
                    role_adj[in_id][out_id] = 1
        
        edge_index, edge_weight = dense_to_sparse(role_adj)
        return edge_index
    
    def construct_features(self):
        features = []
        for node_id in self.nodes:
            role = self.nodes[node_id].role
            profile = self.prompt_set.get_description(role)
            feature = get_sentence_embedding(profile)
            features.append(feature)
        features = torch.tensor(np.array(features))
        return features
    
    def construct_new_features(self, query):
        query_embedding = torch.tensor(get_sentence_embedding(query))
        query_embedding = query_embedding.unsqueeze(0).repeat((self.num_nodes,1))
        new_features = torch.cat((self.features,query_embedding),dim=1)
        return new_features

    def _reset_refinement_losses(self, reference: Optional[torch.Tensor] = None) -> None:
        if reference is None:
            reference = self.refinement_weight
        self.refinement_anchor_loss = reference.new_tensor(0.0)
        self.refinement_sparse_loss = reference.new_tensor(0.0)
        self.edge_bias_l2_loss = reference.new_tensor(0.0)

    def _refine_spatial_logits(self, raw_spatial_logits: torch.Tensor) -> torch.Tensor:
        if not self.optimized_spatial:
            self.spatial_edge_probabilities = None
            self._reset_refinement_losses(raw_spatial_logits)
            return raw_spatial_logits.view(-1)

        raw_spatial_logits = raw_spatial_logits.view(self.num_nodes, self.num_nodes)
        mask = self.spatial_masks.view(self.num_nodes, self.num_nodes).to(
            device=raw_spatial_logits.device,
            dtype=raw_spatial_logits.dtype,
        )
        sketched_adj = torch.sigmoid(raw_spatial_logits) * mask
        anchor_adj = self.anchor_spatial_matrix.to(
            device=raw_spatial_logits.device,
            dtype=raw_spatial_logits.dtype,
        )
        rank = min(self.refine_rank, self.num_nodes)
        left_singular_vectors, _, _ = torch.linalg.svd(
            sketched_adj,
            full_matrices=False,
        )
        left_singular_vectors = left_singular_vectors[:, :rank]
        refinement_weight = self.refinement_weight[:rank, :rank].to(
            device=raw_spatial_logits.device,
            dtype=raw_spatial_logits.dtype,
        )
        refined_adj = left_singular_vectors @ refinement_weight @ left_singular_vectors.t()

        self.refinement_anchor_loss = (
            0.5 * torch.linalg.matrix_norm(sketched_adj - refined_adj, ord="fro").pow(2)
            + 0.5 * torch.linalg.matrix_norm(anchor_adj - refined_adj, ord="fro").pow(2)
        )
        self.refinement_sparse_loss = torch.linalg.svdvals(refinement_weight).sum()
        edge_bias = self.spatial_edge_bias.to(
            device=raw_spatial_logits.device,
            dtype=raw_spatial_logits.dtype,
        ).view(self.num_nodes, self.num_nodes) * mask
        active_edges = mask.sum().clamp_min(1.0)
        self.edge_bias_l2_loss = edge_bias.pow(2).sum() / active_edges
        refined_prob = refined_adj.clamp(1e-6, 1.0 - 1e-6)
        refined_logits = torch.logit(refined_prob)
        biased_logits = refined_logits + self.edge_bias_scale * edge_bias
        self.spatial_edge_probabilities = torch.sigmoid(biased_logits).clamp(1e-6, 1.0 - 1e-6).view(-1)
        return biased_logits.view(-1)

    def prepare_spatial_logits(
            self,
            task: str,
            track_grad: bool = True,
            ) -> None:
        def _compute_spatial_logits() -> None:
            new_features = self.construct_new_features(task)
            logits = self.gcn(new_features,self.role_adj_matrix)
            logits = self.mlp(logits)
            raw_spatial_logits = min_max_norm(torch.flatten(logits @ logits.t()))
            self.spatial_logits = self._refine_spatial_logits(raw_spatial_logits)

        if track_grad:
            _compute_spatial_logits()
        else:
            with torch.no_grad():
                _compute_spatial_logits()

    def edge_selector_task_embedding(self, task: str) -> torch.Tensor:
        return torch.tensor(np.array(get_sentence_embedding(task)), dtype=torch.float32)

    def edge_selector_feature(
            self,
            task: str,
            source_id: str,
            target_id: str,
            task_embedding: Optional[torch.Tensor] = None,
            ) -> torch.Tensor:
        task_embedding = task_embedding if task_embedding is not None else self.edge_selector_task_embedding(task)
        source_role_embedding = self.features[self.node_id_to_index[source_id]]
        target_role_embedding = self.features[self.node_id_to_index[target_id]]
        return torch.cat([
            task_embedding.float(),
            source_role_embedding.detach().float(),
            target_role_embedding.detach().float(),
        ])

    def apply_edge_selector(self, task: str, edge_selector, round_idx: int) -> None:
        if edge_selector is None:
            return

        selected_edges = [
            edge_info
            for edge_info in self.edge_log_probs
            if edge_info.get("round") == round_idx
        ]
        if not selected_edges:
            return

        was_training = edge_selector.training
        edge_selector.eval()
        device = next(edge_selector.parameters()).device
        task_embedding = self.edge_selector_task_embedding(task)
        with torch.no_grad():
            valid_edges = []
            feature_rows = []
            for edge_info in selected_edges:
                source_id = edge_info["source"]
                target_id = edge_info["target"]
                if source_id not in self.nodes or target_id not in self.nodes:
                    continue
                valid_edges.append(edge_info)
                feature_rows.append(self.edge_selector_feature(
                    task,
                    source_id,
                    target_id,
                    task_embedding=task_embedding,
                ))
            if not valid_edges:
                if was_training:
                    edge_selector.train()
                return

            features = torch.stack(feature_rows).to(device)
            keep_probabilities = torch.sigmoid(edge_selector(features)).view(-1)
            keep_samples = torch.bernoulli(keep_probabilities).bool()
            for edge_info, keep_probability, keep_sample in zip(
                    valid_edges,
                    keep_probabilities.cpu(),
                    keep_samples.cpu(),
                    ):
                keep = bool(keep_sample.item())
                edge_info["selector_probability"] = float(keep_probability.item())
                edge_info["selector_keep"] = keep
                if not keep:
                    source_id = edge_info["source"]
                    target_id = edge_info["target"]
                    self.find_node(source_id).remove_successor(
                        self.find_node(target_id),
                        edge_info["type"],
                    )
        if was_training:
            edge_selector.train()
        
    @property
    def spatial_adj_matrix(self):
        matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for i, node1_id in enumerate(self.nodes):
            for j, node2_id in enumerate(self.nodes):
                if self.nodes[node2_id] in self.nodes[node1_id].spatial_successors: 
                    matrix[i, j] = 1
        return matrix

    @property
    def temporal_adj_matrix(self):
        matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for i, node1_id in enumerate(self.nodes):
            for j, node2_id in enumerate(self.nodes):
                if self.nodes[node2_id] in self.nodes[node1_id].temporal_successors: 
                    matrix[i, j] = 1
        return matrix

    @property
    def num_edges(self):
        num_edges = 0
        for node in self.nodes.values():
            num_edges += len(node.spatial_successors)
        return num_edges

    @property
    def communication_edge_count(self):
        node_ids = set(self.nodes.keys())
        num_edges = 0
        for node in self.nodes.values():
            num_edges += sum(
                1 for successor in node.spatial_successors
                if successor.id in node_ids
            )
            num_edges += sum(
                1 for successor in node.temporal_successors
                if successor.id in node_ids
            )
        return num_edges
    
    @property
    def num_nodes(self):
        return len(self.nodes)

    def find_node(self, id: str):
        if id in self.nodes.keys():
            return self.nodes[id]
        raise Exception(f"Node not found: {id} among "
                        f"{[node.id for node in self.nodes.values()]}")
        
    def add_node(self, node: Node):
        node_id = node.id if node.id is not None else shortuuid.ShortUUID().random(length=4)
        while node_id in self.nodes:
            node_id = shortuuid.ShortUUID().random(length=4)
        node.id = node_id
        self.nodes[node_id] = node
        if hasattr(self, "node_id_to_index"):
            self.node_id_to_index[node_id] = len(self.node_id_to_index)
        return node
    
    def init_nodes(self):
        """
        Creates and adds new nodes to the graph.
        """
        for agent_name,kwargs in zip(self.agent_names,self.node_kwargs):
            if agent_name in AgentRegistry.registry:
                kwargs["domain"] = self.domain
                kwargs["llm_name"] = self.llm_name
                agent_instance = AgentRegistry.get(agent_name, **kwargs)
                self.add_node(agent_instance)
    
    def init_potential_edges(self):
        """
        Creates and potential edges to the graph.
        """
        for node1_id in self.nodes.keys():
            for node2_id in self.nodes.keys():
                self.potential_spatial_edges.append([node1_id,node2_id])
                self.potential_temporal_edges.append([node1_id,node2_id])

    def clear_spatial_connection(self):
        """
        Clear all the spatial connection of the nodes in the graph.
        """
        for node_id in self.nodes.keys():
            self.nodes[node_id].spatial_predecessors = []
            self.nodes[node_id].spatial_successors = []
        self.decision_node.spatial_predecessors = []
        self.decision_node.spatial_successors = []
    
    def clear_temporal_connection(self):
        """
        Clear all the temporal connection of the nodes in the graph.
        """
        for node_id in self.nodes.keys():
            self.nodes[node_id].temporal_predecessors = []
            self.nodes[node_id].temporal_successors = []

    def connect_decision_node(self):
        for node_id in self.nodes.keys():
            self.nodes[node_id].add_successor(self.decision_node)

    def construct_spatial_connection(
            self,
            round:int = 0,
            temperature: float = 1.0,
            threshold: float = None,
            track_grad: bool = True,
            ): # temperature must >= 1.0
        self.clear_spatial_connection()
        log_probs = [torch.tensor(0.0, requires_grad=self.optimized_spatial and track_grad)]
        
        for edge_idx, (potential_connection, edge_logit, edge_mask) in enumerate(zip(self.potential_spatial_edges, self.spatial_logits, self.spatial_masks)):
            out_node:Node = self.find_node(potential_connection[0])
            in_node:Node = self.find_node(potential_connection[1])
            if edge_mask == 0.0:
                continue
            elif edge_mask == 1.0 and self.optimized_spatial==False:
                if not self.check_cycle(in_node, {out_node}):
                    out_node.add_successor(in_node,'spatial')
                continue
            if not self.check_cycle(in_node, {out_node}):
                if self.spatial_edge_probabilities is not None:
                    edge_prob = self.spatial_edge_probabilities[edge_idx]
                else:
                    edge_prob = torch.sigmoid(edge_logit / temperature)
                edge_prob = edge_prob.clamp(1e-6, 1.0 - 1e-6)
                if threshold:
                    edge_prob = torch.tensor(1 if edge_prob > threshold else 0)
                if torch.rand(1) < edge_prob:
                    out_node.add_successor(in_node,'spatial')
                    edge_info = {
                        "type": "spatial",
                        "round": round,
                        "source": out_node.id,
                        "target": in_node.id,
                        "edge_key": f"spatial:{round}:{out_node.id}->{in_node.id}",
                    }
                    if track_grad:
                        edge_log_prob = torch.log(edge_prob)
                        log_probs.append(edge_log_prob)
                        edge_info["log_prob"] = edge_log_prob
                    self.edge_log_probs.append(edge_info)
                else:
                    if track_grad:
                        log_probs.append(torch.log(1 - edge_prob))
                    
        return torch.sum(torch.stack(log_probs))
    
    def construct_temporal_connection(
            self,
            round:int = 0,
            temperature: float = 1.0,
            threshold: float = None,
            track_grad: bool = True,
            ):  # temperature must >= 1.0
        self.clear_temporal_connection()
        log_probs = [torch.tensor(0.0, requires_grad=self.optimized_temporal and track_grad)]
        if round == 0:
            return torch.sum(torch.stack(log_probs))  
        for potential_connection, edge_logit, edge_mask in zip(self.potential_temporal_edges, self.temporal_logits, self.temporal_masks):
            out_node:Node = self.find_node(potential_connection[0])
            in_node:Node = self.find_node(potential_connection[1])
            if edge_mask == 0.0:
                continue
            elif edge_mask == 1.0 and self.optimized_temporal==False:
                if not self.check_cycle(in_node, {out_node}):
                    out_node.add_successor(in_node,'temporal')
                continue
            
            edge_prob = torch.sigmoid(edge_logit / temperature)
            if threshold:
                edge_prob = torch.tensor(1 if edge_prob > threshold else 0)
            if torch.rand(1) < edge_prob:
                out_node.add_successor(in_node,'temporal')
                edge_info = {
                    "type": "temporal",
                    "round": round,
                    "source": out_node.id,
                    "target": in_node.id,
                    "edge_key": f"temporal:{round}:{out_node.id}->{in_node.id}",
                }
                if track_grad:
                    edge_log_prob = torch.log(edge_prob)
                    log_probs.append(edge_log_prob)
                    edge_info["log_prob"] = edge_log_prob
                self.edge_log_probs.append(edge_info)
            else:
                if track_grad:
                    log_probs.append(torch.log(1 - edge_prob))
                    
        return torch.sum(torch.stack(log_probs))


    def run(self, inputs: Any, 
                  num_rounds:int = 3, 
                  max_tries: int = 3, 
                  max_time: int = 600,
                  num_entropy_samples: int = 1,
                  record_execution_history: bool = True,
                  track_grad: bool = True,
                  edge_selector = None,
                  record_decision_logprobs: bool = False,) -> List[Any]:
        # inputs:{'task':"xxx"}
        log_probs = 0
        self.edge_log_probs = []
        self.realized_edge_counts = []
        task = inputs.get("task", str(inputs)) if isinstance(inputs, dict) else str(inputs)
        self.prepare_spatial_logits(task, track_grad=track_grad)
        for round in range(num_rounds):
            log_probs += self.construct_spatial_connection(round, track_grad=track_grad)
            log_probs += self.construct_temporal_connection(round, track_grad=track_grad)
            self.apply_edge_selector(task, edge_selector, round)
            self.realized_edge_counts.append(self.communication_edge_count)
            
            in_degree = {node_id: len(node.spatial_predecessors) for node_id, node in self.nodes.items()}
            zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0]

            while zero_in_degree_queue:
                current_node_id = zero_in_degree_queue.pop(0)
                tries = 0
                while tries < max_tries:
                    try:
                        self.nodes[current_node_id].execute(
                            inputs,
                            round_idx=round,
                            num_entropy_samples=num_entropy_samples,
                            record_execution_history=record_execution_history,
                        ) # output is saved in the node.outputs
                        break
                    except Exception as e:
                        print(f"Error during execution of node {current_node_id}: {e}")
                    tries += 1
                for successor in self.nodes[current_node_id].spatial_successors:
                    if successor.id not in self.nodes.keys():
                        continue
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)
            
            self.update_memory()
            
        self.connect_decision_node()
        tries = 0
        while tries < _DECISION_NODE_MAX_TRIES:
            try:
                self.decision_node.execute(
                    inputs,
                    return_logprobs=record_decision_logprobs,
                )
                break
            except Exception as e:
                print(f"Error during execution of decision node: {e}")
                tries += 1
        final_answers = self.decision_node.outputs
        if len(final_answers) == 0:
            final_answers.append("No answer of the decision node")
            
        return final_answers, log_probs

    async def arun(self, input: Dict[str,str], 
                  num_rounds:int = 3, 
                  max_tries: int = 3, 
                  max_time: int = 600,
                  num_entropy_samples: int = 1,
                  record_execution_history: bool = True,
                  track_grad: bool = True,
                  edge_selector = None,
                  record_decision_logprobs: bool = False,) -> List[Any]:
        # inputs:{'task':"xxx"}
        log_probs = 0
        self.edge_log_probs = []
        self.realized_edge_counts = []
        self.prepare_spatial_logits(input['task'], track_grad=track_grad)

        for round in range(num_rounds):
            log_probs += self.construct_spatial_connection(round, track_grad=track_grad)
            log_probs += self.construct_temporal_connection(round, track_grad=track_grad)
            self.apply_edge_selector(input.get("task", str(input)), edge_selector, round)
            self.realized_edge_counts.append(self.communication_edge_count)
            
            in_degree = {node_id: len(node.spatial_predecessors) for node_id, node in self.nodes.items()}
            zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0]

            while zero_in_degree_queue:
                current_node_id = zero_in_degree_queue.pop(0)
                tries = 0
                while tries < max_tries:
                    try:
                        await asyncio.wait_for(
                            self.nodes[current_node_id].async_execute(
                                input,
                                round_idx=round,
                                num_entropy_samples=num_entropy_samples,
                                record_execution_history=record_execution_history,
                            ),
                            timeout=max_time,
                        ) # output is saved in the node.outputs
                        break
                    except Exception as e:
                        print(f"Error during execution of node {current_node_id}: {e}")
                    tries += 1
                for successor in self.nodes[current_node_id].spatial_successors:
                    if successor.id not in self.nodes.keys():
                        continue
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)
            
            self.update_memory()
            
        self.connect_decision_node()
        tries = 0
        while tries < _DECISION_NODE_MAX_TRIES:
            try:
                await asyncio.wait_for(
                    self.decision_node.async_execute(
                        input,
                        return_logprobs=record_decision_logprobs,
                    ),
                    timeout=_DECISION_NODE_TIMEOUT_SECONDS,
                )
                break
            except Exception as e:
                print(f"Error during execution of decision node: {e}")
                tries += 1
        final_answers = self.decision_node.outputs
        if len(final_answers) == 0:
            final_answers.append("No answer of the decision node")
        return final_answers, log_probs
    
    def update_memory(self):
        for id,node in self.nodes.items():
            node.update_memory()

    def clear_execution_history(self):
        for node in self.nodes.values():
            node.execution_history = []
    
    def check_cycle(self, new_node, target_nodes):
        if new_node in target_nodes:
            return True
        for successor in new_node.spatial_successors:
            if self.check_cycle(successor, target_nodes):
                return True
        return False

    def prune_temporal_edges(self, pruning_rate: float) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.optimized_temporal or pruning_rate <= 0:
            return self.temporal_masks, torch.empty(0, dtype=torch.long)

        active_idx = torch.nonzero(self.temporal_masks > 0, as_tuple=False).view(-1)
        if active_idx.numel() == 0:
            return self.temporal_masks, active_idx

        prune_num_edges = int(torch.round(active_idx.numel() * torch.tensor(pruning_rate)).item())
        prune_num_edges = min(max(1, prune_num_edges), active_idx.numel())
        temporal_logits = self.temporal_logits.detach().view(-1)
        prune_idx = active_idx[torch.argsort(temporal_logits[active_idx])[:prune_num_edges]]
        with torch.no_grad():
            self.temporal_masks[prune_idx] = 0
        return self.temporal_masks, prune_idx

    def update_masks(self, pruning_rate: float, spatial_logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.optimized_spatial:
            num_edges = (self.spatial_masks > 0).sum()
            num_masks = (self.spatial_masks == 0).sum()
            prune_num_edges = torch.round(num_edges*pruning_rate) if torch.round(num_edges*pruning_rate)>0 else 1
            edge_logits = spatial_logits if spatial_logits is not None else getattr(self, "spatial_logits", None)
            if edge_logits is None:
                raise RuntimeError(
                    "spatial_logits are required for spatial pruning. Pass batch-level "
                    "spatial_logits to update_masks because spatial logits are computed "
                    "per query in arun()."
                )
            _edge_logits = edge_logits.detach().clone().view(-1)
            min_edge_logit = _edge_logits.min()
            _edge_logits[self.spatial_masks == 0] = min_edge_logit - 1.0
            sorted_edges_idx = torch.argsort(_edge_logits)
            prune_idx = sorted_edges_idx[:int(prune_num_edges + num_masks)]
            self.spatial_masks[prune_idx] = 0
        
        if self.optimized_temporal:
            num_edges = (self.temporal_masks > 0).sum()
            num_masks = (self.temporal_masks == 0).sum()
            prune_num_edges = torch.round(num_edges*pruning_rate) if torch.round(num_edges*pruning_rate)>0 else 1
            _edge_logits = self.temporal_logits.clone()
            min_edge_logit = _edge_logits.min()
            _edge_logits[self.temporal_masks == 0] = min_edge_logit - 1.0
            sorted_edges_idx = torch.argsort(_edge_logits)
            prune_idx = sorted_edges_idx[:int(prune_num_edges + num_masks)]
            self.temporal_masks[prune_idx] = 0
        return self.spatial_masks, self.temporal_masks

def min_max_norm(tensor:torch.Tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    if torch.isclose(max_val, min_val):
        return torch.zeros_like(tensor)
    normalized_0_to_1 = (tensor - min_val) / (max_val - min_val)
    normalized_minus1_to_1 = normalized_0_to_1 * 2 - 1
    return normalized_minus1_to_1
