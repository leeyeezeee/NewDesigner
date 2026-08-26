import shortuuid
import math
from typing import Any, List, Optional, Dict, Tuple
from abc import ABC
import numpy as np
import torch
import torch.nn.functional as F
import asyncio
import traceback

import GDesigner.agents
import GDesigner.prompt
from GDesigner.graph.node import Node
from GDesigner.agents.agent_registry import AgentRegistry
from GDesigner.prompt.prompt_set_registry import PromptSetRegistry
from GDesigner.llm.profile_embedding import (
    get_sentence_embedding,
)
from GDesigner.llm.gpt_chat import EmptyChatCompletionError
from GDesigner.llm.price import MissingRemoteTokenUsageError
from GDesigner.gnn.gcn import MLP
from GDesigner.gnn.gat import InitialResidualGATv2Encoder
from torch_geometric.utils import dense_to_sparse
from torch.nn.utils.parametrizations import spectral_norm

_DECISION_NODE_MAX_TRIES = 5
_DECISION_NODE_TIMEOUT_SECONDS = 1200


def _format_exception(exc: Exception) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


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
                refine_rank: int = 4,
                 ):
        self.id:str = shortuuid.ShortUUID().random(length=4)
        self.domain:str = domain
        self.llm_name:str = llm_name
        self.agent_names:List[str] = list(agent_names)
        self.optimized_spatial = optimized_spatial
        self.optimized_temporal = optimized_temporal
        self.prompt_set = PromptSetRegistry.get(domain)
        self.decision_node:Node = AgentRegistry.get(decision_method, **{"domain":self.domain,"llm_name":self.llm_name})
        agent_node_count = len(self.agent_names)
        fixed_spatial_masks = self._prepare_fixed_mask(
            fixed_spatial_masks,
            agent_node_count,
            spatial=True,
        )
        fixed_temporal_masks = self._prepare_fixed_mask(
            fixed_temporal_masks,
            agent_node_count,
            spatial=False,
        )

        self.nodes:Dict[str,Node] = {}
        self.potential_spatial_edges:List[List[str, str]] = []
        self.potential_temporal_edges:List[List[str,str]] = []
        self.edge_log_probs:List[Dict[str, Any]] = []
        self.spatial_sampling_temperature = 1.0
        self.realized_edge_counts:List[int] = []
        self.realized_spatial_edge_counts:List[int] = []
        # Preserve the realized agent-only graph from every communication
        # round.  The optional graph critic consumes these detached snapshots;
        # the execution and edge-ablation paths continue to use live edges.
        self.realized_spatial_adjacencies:List[torch.Tensor] = []
        # Raw role + question features produced immediately before the actor
        # GAT.  The optional critic reuses this detached tensor and never calls
        # the frozen sentence encoder a second time.
        self.task_conditioned_feature_task:Optional[str] = None
        self.task_conditioned_features:Optional[torch.Tensor] = None
        self.node_kwargs = node_kwargs if node_kwargs is not None else [{} for _ in agent_names]
        self.edge_embedding_dim = 16
        self.init_nodes() # add nodes to the self.nodes
        self.node_id_to_index = {node_id: idx for idx, node_id in enumerate(self.nodes)}
        self.init_potential_edges() # add potential edges to the self.potential_spatial/temporal_edges
        self.role_adj_matrix = self.construct_adj_matrix()
        self.features = self.construct_features()
        node_feature_dim = self.features.size(1)
        topology_input_dim = node_feature_dim * 2
        self.spatial_policy_architecture = (
            "initial_residual_gatv2_affinity_low_rank_dynamic_cycle_external_decision_v7"
        )
        self.gat = InitialResidualGATv2Encoder(
            topology_input_dim,
            bottleneck_channels=16,
            out_channels=node_feature_dim,
            heads=4,
            dropout=0.1,
        )
        self.edge_mlp = MLP(node_feature_dim, 16, self.edge_embedding_dim)
        affinity = torch.nn.Linear(
            self.edge_embedding_dim,
            self.edge_embedding_dim,
            bias=False,
        )
        torch.nn.init.xavier_uniform_(affinity.weight)
        self.spatial_affinity = spectral_norm(affinity)
        self.spatial_affinity.requires_grad_(optimized_spatial)

        # Keep the task-conditioned affinity decoder and refine its sketched
        # adjacency with the low-rank decoder from G-Designer Eq. (14).
        self.anchor_spatial_matrix = fixed_spatial_masks.reshape(
            agent_node_count,
            agent_node_count,
        ).float()
        # Keep the decoder genuinely low-rank. Once the external decision node
        # is removed, allowing rank == agent_node_count would initialize
        # Z W Z^T as the identity and collapse every off-diagonal agent edge.
        max_refine_rank = max(1, agent_node_count - 1)
        self.refine_rank = min(max(1, int(refine_rank)), max_refine_rank)
        self.refinement_weight = torch.nn.Parameter(
            torch.eye(self.refine_rank),
            requires_grad=optimized_spatial,
        )
        self.refinement_anchor_loss = torch.tensor(0.0)
        self.refinement_sparse_loss = torch.tensor(0.0)
        self.spatial_edge_probabilities: Optional[torch.Tensor] = None

        # self.spatial_logits = torch.nn.Parameter(torch.ones(len(self.potential_spatial_edges), requires_grad=optimized_spatial) * init_spatial_logit,
        #                                          requires_grad=optimized_spatial) # trainable edge logits
        self.spatial_masks = torch.nn.Parameter(fixed_spatial_masks,requires_grad=False)  # fixed edge masks

        init_temporal_logit = torch.log(torch.tensor(initial_temporal_probability / (1 - initial_temporal_probability))) if optimized_temporal else 10.0
        self.temporal_logits = torch.nn.Parameter(torch.ones(len(self.potential_temporal_edges), requires_grad=optimized_temporal) * init_temporal_logit,
                                                 requires_grad=optimized_temporal) # trainable edge logits
        self.temporal_masks = torch.nn.Parameter(fixed_temporal_masks,requires_grad=False)  # fixed edge masks

    def spatial_parameters(self) -> List[torch.nn.Parameter]:
        parameters = list(self.gat.parameters())
        parameters.extend(self.edge_mlp.parameters())
        parameters.extend(self.spatial_affinity.parameters())
        if self.refinement_weight.requires_grad:
            parameters.append(self.refinement_weight)
        return [parameter for parameter in parameters if parameter.requires_grad]

    @staticmethod
    def _prepare_fixed_mask(
            mask,
            agent_node_count: int,
            *,
            spatial: bool,
            ) -> torch.Tensor:
        if mask is None:
            if spatial:
                base = torch.ones((agent_node_count, agent_node_count))
                base.fill_diagonal_(0)
            else:
                base = torch.ones((agent_node_count, agent_node_count))
        else:
            base = torch.as_tensor(mask).reshape(-1)
            expected = agent_node_count * agent_node_count
            if base.numel() != expected:
                raise ValueError(
                    "The fixed edge mask must describe only the agent policy "
                    f"nodes ({expected} values); received {base.numel()}. The "
                    "external decision node is not part of either adjacency matrix."
                )
            base = base.view(agent_node_count, agent_node_count).clone()

        return base.reshape(-1)
    
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
            role_2_id.setdefault(role, []).append(i)
            
        for edge in role_connect:
            in_role,out_role = edge
            in_ids = role_2_id[in_role]
            out_ids = role_2_id[out_role]
            for in_id in in_ids:
                for out_id in out_ids:
                    role_adj[in_id][out_id] = 1

        edge_index, edge_weight = dense_to_sparse(role_adj)
        return edge_index
    
    def _node_profile_text(self, node: Node) -> str:
        return self.prompt_set.get_description(node.role)

    def construct_features(self):
        features = []
        for node_id in self.nodes:
            node = self.nodes[node_id]
            profile = self._node_profile_text(node)
            feature = get_sentence_embedding(profile)
            features.append(feature)
        features = torch.tensor(np.array(features))
        return features
    
    def construct_new_features(self, query):
        query_embedding = torch.tensor(
            get_sentence_embedding(query),
            dtype=self.features.dtype,
            device=self.features.device,
        )
        query_embedding = query_embedding.unsqueeze(0).repeat((self.num_nodes, 1))
        return torch.cat((self.features, query_embedding), dim=1)

    @staticmethod
    def _node_embedding_statistics(embeddings: torch.Tensor) -> Dict[str, float]:
        embeddings = embeddings.detach().float()
        node_count = int(embeddings.size(0))
        if node_count < 2:
            return {
                "mean_pairwise_cosine": 1.0,
                "std_pairwise_cosine": 0.0,
                "mean_feature_variance": 0.0,
            }
        normalized = F.normalize(embeddings, p=2.0, dim=-1, eps=1e-6)
        similarities = normalized @ normalized.t()
        off_diagonal = ~torch.eye(
            node_count,
            dtype=torch.bool,
            device=embeddings.device,
        )
        pairwise = similarities[off_diagonal]
        return {
            "mean_pairwise_cosine": float(pairwise.mean().cpu().item()),
            "std_pairwise_cosine": float(
                pairwise.std(unbiased=False).cpu().item()
            ),
            "mean_feature_variance": float(
                embeddings.var(dim=0, unbiased=False).mean().cpu().item()
            ),
        }

    @staticmethod
    def _attention_statistics(
        edge_index: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> Dict[str, float]:
        weights = attention_weights.detach().float()
        if weights.ndim == 1:
            weights = weights.unsqueeze(-1)
        if weights.numel() == 0:
            return {
                "normalized_entropy": 0.0,
                "weight_std": 0.0,
                "weight_min": 0.0,
                "weight_max": 0.0,
            }
        target_indices = edge_index[1].detach()
        entropies = []
        normalized_weight_values = []
        for target in torch.unique(target_indices):
            target_weights = weights[target_indices == target]
            degree = int(target_weights.size(0))
            if degree <= 1:
                continue
            totals = target_weights.sum(dim=0)
            valid_heads = totals > 1e-12
            if not bool(valid_heads.any()):
                continue
            target_weights = (
                target_weights[:, valid_heads]
                / totals[valid_heads].unsqueeze(0)
            )
            entropy = -(
                target_weights.clamp_min(1e-12)
                * target_weights.clamp_min(1e-12).log()
            ).sum(dim=0) / math.log(degree)
            entropies.append(entropy)
            normalized_weight_values.append(target_weights.reshape(-1))
        normalized_entropy = (
            torch.cat(entropies).mean()
            if entropies
            else weights.new_tensor(0.0)
        )
        normalized_weights = (
            torch.cat(normalized_weight_values)
            if normalized_weight_values
            else weights.new_zeros(1)
        )
        return {
            "normalized_entropy": float(normalized_entropy.cpu().item()),
            "weight_std": float(
                normalized_weights.std(unbiased=False).cpu().item()
            ),
            "weight_min": float(normalized_weights.min().cpu().item()),
            "weight_max": float(normalized_weights.max().cpu().item()),
        }

    def prepare_spatial_logits(
            self,
            task: str,
            track_grad: bool = True,
            ) -> None:
        def _compute_spatial_logits() -> None:
            new_features = self.construct_new_features(task)
            self.task_conditioned_feature_task = str(task)
            self.task_conditioned_features = new_features.detach()
            node_embeddings, encoder_diagnostics = self.gat(
                new_features,
                self.role_adj_matrix,
                return_diagnostics=True,
            )
            if not torch.isfinite(node_embeddings).all():
                nonfinite = int((~torch.isfinite(node_embeddings)).sum().item())
                raise FloatingPointError(
                    "Initial-residual GATv2 produced non-finite node embeddings: "
                    f"{nonfinite}/{node_embeddings.numel()} values."
                )
            edge_embeddings = F.normalize(
                self.edge_mlp(node_embeddings),
                p=2.0,
                dim=-1,
                eps=1e-6,
            )
            projected_embeddings = self.spatial_affinity(edge_embeddings)
            affinity_scores = math.sqrt(self.edge_embedding_dim) * (
                edge_embeddings @ projected_embeddings.t()
            )
            if not torch.isfinite(affinity_scores).all():
                nonfinite = int((~torch.isfinite(affinity_scores)).sum().item())
                raise FloatingPointError(
                    "Spatial affinity produced non-finite edge logits: "
                    f"{nonfinite}/{affinity_scores.numel()} values."
                )
            self.spatial_logits = self._refine_spatial_logits(affinity_scores)
            valid_mask = self.spatial_masks.reshape(-1) > 0
            valid_logits = self.spatial_logits[valid_mask]
            valid_probabilities = self.spatial_edge_probabilities[valid_mask]
            if valid_logits.numel() == 0:
                edge_distribution_diagnostics = {
                    "valid_edges": 0,
                    "logit_mean": 0.0,
                    "logit_std": 0.0,
                    "logit_min": 0.0,
                    "logit_max": 0.0,
                    "probability_mean": 0.0,
                    "probability_std": 0.0,
                    "probability_min": 0.0,
                    "probability_max": 0.0,
                    "uncertain_probability_fraction": 0.0,
                    "expected_edges": 0.0,
                }
            else:
                edge_distribution_diagnostics = {
                    "valid_edges": int(valid_logits.numel()),
                    "logit_mean": float(valid_logits.mean().detach().cpu().item()),
                    "logit_std": float(
                        valid_logits.std(unbiased=False).detach().cpu().item()
                    ),
                    "logit_min": float(valid_logits.min().detach().cpu().item()),
                    "logit_max": float(valid_logits.max().detach().cpu().item()),
                    "probability_mean": float(
                        valid_probabilities.mean().detach().cpu().item()
                    ),
                    "probability_std": float(
                        valid_probabilities.std(unbiased=False).detach().cpu().item()
                    ),
                    "probability_min": float(
                        valid_probabilities.min().detach().cpu().item()
                    ),
                    "probability_max": float(
                        valid_probabilities.max().detach().cpu().item()
                    ),
                    "uncertain_probability_fraction": float(
                        (
                            (valid_probabilities >= 0.4)
                            & (valid_probabilities <= 0.6)
                        ).float().mean().detach().cpu().item()
                    ),
                    "expected_edges": float(
                        valid_probabilities.sum().detach().cpu().item()
                    ),
                }
            self.topology_diagnostics = {
                "node_embeddings": {
                    "joint_input": self._node_embedding_statistics(new_features),
                    "projected_input": self._node_embedding_statistics(
                        encoder_diagnostics["initial"]
                    ),
                    "gatv2_layer1": self._node_embedding_statistics(
                        encoder_diagnostics["layer1"]
                    ),
                    "gatv2_layer2": self._node_embedding_statistics(
                        encoder_diagnostics["layer2"]
                    ),
                },
                "attention": {
                    "gatv2_layer1": self._attention_statistics(
                        encoder_diagnostics["attention1_edge_index"],
                        encoder_diagnostics["attention1_weights"],
                    ),
                    "gatv2_layer2": self._attention_statistics(
                        encoder_diagnostics["attention2_edge_index"],
                        encoder_diagnostics["attention2_weights"],
                    ),
                },
                "edge_distribution": edge_distribution_diagnostics,
                "final_node_embeddings": node_embeddings.detach().cpu(),
                "initial_residual_weight": float(
                    encoder_diagnostics["initial_residual_weight"]
                ),
                "sampling_temperature": self.spatial_sampling_temperature,
            }

        if track_grad:
            _compute_spatial_logits()
        else:
            with torch.no_grad():
                _compute_spatial_logits()

    def _reset_refinement_losses(
            self,
            reference: Optional[torch.Tensor] = None,
            ) -> None:
        if reference is None:
            reference = self.refinement_weight
        self.refinement_anchor_loss = reference.new_tensor(0.0)
        self.refinement_sparse_loss = reference.new_tensor(0.0)

    def _refine_spatial_logits(
            self,
            raw_spatial_logits: torch.Tensor,
            ) -> torch.Tensor:
        """Apply the G-Designer low-rank refinement decoder.

        ``spatial_affinity`` remains the task-conditioned affinity matrix.  A
        separate ``refinement_weight`` parameter is used as W in Z W Z^T so
        the two learned matrices cannot accidentally overwrite one another.
        """
        raw_matrix = raw_spatial_logits.reshape(self.num_nodes, self.num_nodes)
        mask = self.spatial_masks.reshape(self.num_nodes, self.num_nodes).to(
            device=raw_matrix.device,
            dtype=raw_matrix.dtype,
        )
        # G-Designer's sketch is dense; the execution mask defines which
        # refined entries may become communication edges.
        dense_sketched_adj = torch.sigmoid(
            raw_matrix / max(float(self.spatial_sampling_temperature), 1e-6)
        )

        if not self.optimized_spatial:
            self._reset_refinement_losses(raw_matrix)
            probabilities = (dense_sketched_adj * mask).clamp(
                1e-6,
                1.0 - 1e-6,
            )
            self.spatial_edge_probabilities = probabilities.reshape(-1)
            return torch.logit(probabilities).reshape(-1)

        anchor_adj = self.anchor_spatial_matrix.to(
            device=raw_matrix.device,
            dtype=raw_matrix.dtype,
        )
        rank = min(self.refine_rank, self.num_nodes)
        left_singular_vectors, _, _ = torch.linalg.svd(
            dense_sketched_adj,
            full_matrices=False,
        )
        basis = left_singular_vectors[:, :rank]
        refinement_weight = self.refinement_weight[:rank, :rank].to(
            device=raw_matrix.device,
            dtype=raw_matrix.dtype,
        )
        refined_adj = basis @ refinement_weight @ basis.t()

        self.refinement_anchor_loss = (
            0.5
            * torch.linalg.matrix_norm(
                dense_sketched_adj - refined_adj,
                ord="fro",
            ).pow(2)
            + 0.5
            * torch.linalg.matrix_norm(
                anchor_adj - refined_adj,
                ord="fro",
            ).pow(2)
        )
        self.refinement_sparse_loss = torch.linalg.svdvals(
            refinement_weight
        ).sum()

        probabilities = (refined_adj * mask).clamp(1e-6, 1.0 - 1e-6)
        self.spatial_edge_probabilities = probabilities.reshape(-1)
        return torch.logit(probabilities).reshape(-1)

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
        node_ids = set(self.nodes.keys())
        return sum(
            1
            for node in self.nodes.values()
            for successor in node.spatial_successors
            if successor.id in node_ids
        )

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
    def mean_spatial_edges_per_round(self) -> float:
        """Return one-round-equivalent spatial edges, excluding temporal edges."""
        if not self.realized_spatial_edge_counts:
            return 0.0
        return sum(self.realized_spatial_edge_counts) / len(
            self.realized_spatial_edge_counts
        )
    
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

    def connect_decision_node(self) -> None:
        """Aggregate every agent's latest answer outside the learned graph."""
        self.decision_node.clear_connections()
        for node in self.nodes.values():
            node.add_successor(self.decision_node, 'spatial')

    def check_cycle(self, new_node: Node, target_nodes) -> bool:
        """Return whether following spatial edges reaches a candidate source.

        This is the incremental cycle check used by the original G-Designer:
        before adding ``source -> target``, traversal starts at ``target`` and
        rejects the edge if ``source`` is already reachable.
        """
        if new_node in target_nodes:
            return True
        for successor in new_node.spatial_successors:
            if self.check_cycle(successor, target_nodes):
                return True
        return False

    def _spatial_topological_order(self) -> List[str]:
        """Topologically order the realized agent-only spatial graph."""
        in_degree = {
            node_id: len(node.spatial_predecessors)
            for node_id, node in self.nodes.items()
        }
        zero_in_degree_queue = [
            node_id for node_id, degree in in_degree.items() if degree == 0
        ]
        order: List[str] = []
        while zero_in_degree_queue:
            current_node_id = zero_in_degree_queue.pop(0)
            order.append(current_node_id)
            for successor in self.nodes[current_node_id].spatial_successors:
                if successor.id not in self.nodes:
                    continue
                in_degree[successor.id] -= 1
                if in_degree[successor.id] == 0:
                    zero_in_degree_queue.append(successor.id)

        if len(order) != len(self.nodes):
            raise RuntimeError(
                "The realized spatial graph contains a cycle; incremental "
                "G-Designer cycle detection should have rejected it."
            )
        return order

    def construct_spatial_connection(
            self,
            round:int = 0,
            temperature: float = 1.0,
            track_grad: bool = True,
            ): # temperature must >= 1.0
        self.clear_spatial_connection()
        log_probs = [torch.tensor(0.0, requires_grad=self.optimized_spatial and track_grad)]
        
        for potential_connection, edge_logit, edge_mask in zip(
                self.potential_spatial_edges,
                self.spatial_logits,
                self.spatial_masks,
                ):
            out_node:Node = self.find_node(potential_connection[0])
            in_node:Node = self.find_node(potential_connection[1])
            if edge_mask == 0.0:
                continue
            elif edge_mask == 1.0 and self.optimized_spatial==False:
                if not self.check_cycle(in_node, {out_node}):
                    out_node.add_successor(in_node,'spatial')
                continue
            if self.check_cycle(in_node, {out_node}):
                continue
            edge_distribution = torch.distributions.Bernoulli(
                logits=edge_logit / temperature
            )
            edge_sample = edge_distribution.sample()
            edge_selected = bool(edge_sample.item())
            if edge_selected:
                out_node.add_successor(in_node,'spatial')
                edge_info = {
                    "type": "spatial",
                    "round": round,
                    "source": out_node.id,
                    "target": in_node.id,
                    "edge_key": f"spatial:{round}:{out_node.id}->{in_node.id}",
                }
                if track_grad:
                    edge_log_prob = edge_distribution.log_prob(edge_sample)
                    log_probs.append(edge_log_prob)
                    edge_info["log_prob"] = edge_log_prob
                self.edge_log_probs.append(edge_info)
            elif track_grad:
                log_probs.append(edge_distribution.log_prob(edge_sample))
                    
        return torch.sum(torch.stack(log_probs))
    
    def construct_temporal_connection(
            self,
            round:int = 0,
            temperature: float = 1.0,
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
            
            edge_distribution = torch.distributions.Bernoulli(
                logits=edge_logit / temperature
            )
            edge_sample = edge_distribution.sample()
            edge_selected = bool(edge_sample.item())
            if edge_selected:
                out_node.add_successor(in_node,'temporal')
                edge_info = {
                    "type": "temporal",
                    "round": round,
                    "source": out_node.id,
                    "target": in_node.id,
                    "edge_key": f"temporal:{round}:{out_node.id}->{in_node.id}",
                }
                if track_grad:
                    edge_log_prob = edge_distribution.log_prob(edge_sample)
                    log_probs.append(edge_log_prob)
                    edge_info["log_prob"] = edge_log_prob
                self.edge_log_probs.append(edge_info)
            else:
                if track_grad:
                    log_probs.append(edge_distribution.log_prob(edge_sample))
                    
        return torch.sum(torch.stack(log_probs))


    def run(self, inputs: Any, 
                  num_rounds:int = 3, 
                  max_tries: int = 3, 
                  max_time: int = 600,
                  record_execution_history: bool = True,
                  track_grad: bool = True,
                  edge_selector = None,
                  record_node_logprobs: bool = False,
                  node_logprob_token_limit: Optional[int] = None,
                  record_decision_logprobs: bool = False,) -> List[Any]:
        # inputs:{'task':"xxx"}
        log_probs = 0
        self.edge_log_probs = []
        self.realized_edge_counts = []
        self.realized_spatial_edge_counts = []
        self.realized_spatial_adjacencies = []
        task = inputs.get("task", str(inputs)) if isinstance(inputs, dict) else str(inputs)
        self.prepare_spatial_logits(task, track_grad=track_grad)
        for round in range(num_rounds):
            log_probs += self.construct_spatial_connection(
                round,
                track_grad=track_grad,
            )
            log_probs += self.construct_temporal_connection(round, track_grad=track_grad)
            self.apply_edge_selector(task, edge_selector, round)
            self.realized_spatial_adjacencies.append(
                torch.as_tensor(self.spatial_adj_matrix, dtype=torch.float32)
            )
            self.realized_spatial_edge_counts.append(self.num_edges)
            self.realized_edge_counts.append(self.communication_edge_count)
            
            # Match G-Designer: execute the realized acyclic spatial graph in
            # topological order, including every isolated agent.
            for current_node_id in self._spatial_topological_order():
                current_node = self.nodes[current_node_id]
                tries = 0
                while tries < max_tries:
                    try:
                        current_node.execute(
                            inputs,
                            round_idx=round,
                            record_execution_history=record_execution_history,
                            return_logprobs=record_node_logprobs,
                            logprob_token_limit=node_logprob_token_limit,
                        ) # output is saved in the node.outputs
                        break
                    except MissingRemoteTokenUsageError:
                        raise
                    except EmptyChatCompletionError as e:
                        print(
                            "Empty response during execution of node "
                            f"{current_node_id} ({current_node.role}) "
                            f"round={round}; skipping retry.\n"
                            f"{type(e).__name__}: {e!r}\n{_format_exception(e)}"
                        )
                        break
                    except Exception as e:
                        print(
                            "Error during execution of node "
                            f"{current_node_id} ({current_node.role}) "
                            f"round={round} try={tries + 1}/{max_tries}: "
                            f"{type(e).__name__}: {e!r}\n{_format_exception(e)}"
                        )
                    tries += 1
            
            self.update_memory()

        self.connect_decision_node()
        tries = 0
        while tries < _DECISION_NODE_MAX_TRIES:
            try:
                self.decision_node.execute(
                    inputs,
                    round_idx=num_rounds,
                    record_execution_history=record_execution_history,
                    return_logprobs=record_decision_logprobs,
                    logprob_token_limit=node_logprob_token_limit,
                )
                break
            except MissingRemoteTokenUsageError:
                raise
            except Exception as e:
                print(
                    "Error during execution of external decision node "
                    f"try={tries + 1}/{_DECISION_NODE_MAX_TRIES}: "
                    f"{type(e).__name__}: {e!r}\n{_format_exception(e)}"
                )
            tries += 1

        final_answers = self.decision_node.outputs
        if len(final_answers) == 0:
            final_answers.append("No answer of the decision node")
            
        return final_answers, log_probs

    async def arun(self, input: Dict[str,str], 
                  num_rounds:int = 3, 
                  max_tries: int = 3, 
                  max_time: int = 600,
                  record_execution_history: bool = True,
                  track_grad: bool = True,
                  edge_selector = None,
                  record_node_logprobs: bool = False,
                  node_logprob_token_limit: Optional[int] = None,
                  record_decision_logprobs: bool = False,) -> List[Any]:
        # inputs:{'task':"xxx"}
        log_probs = 0
        self.edge_log_probs = []
        self.realized_edge_counts = []
        self.realized_spatial_edge_counts = []
        self.realized_spatial_adjacencies = []
        self.prepare_spatial_logits(input['task'], track_grad=track_grad)

        for round in range(num_rounds):
            log_probs += self.construct_spatial_connection(
                round,
                track_grad=track_grad,
            )
            log_probs += self.construct_temporal_connection(round, track_grad=track_grad)
            self.apply_edge_selector(
                input.get("task", str(input)), edge_selector, round
            )
            self.realized_spatial_adjacencies.append(
                torch.as_tensor(self.spatial_adj_matrix, dtype=torch.float32)
            )
            self.realized_spatial_edge_counts.append(self.num_edges)
            self.realized_edge_counts.append(self.communication_edge_count)
            
            for current_node_id in self._spatial_topological_order():
                current_node = self.nodes[current_node_id]
                tries = 0
                while tries < max_tries:
                    try:
                        await asyncio.wait_for(
                            current_node.async_execute(
                                input,
                                round_idx=round,
                                record_execution_history=record_execution_history,
                                return_logprobs=record_node_logprobs,
                                logprob_token_limit=node_logprob_token_limit,
                            ),
                            timeout=max_time,
                        ) # output is saved in the node.outputs
                        break
                    except EmptyChatCompletionError as e:
                        print(
                            "Empty response during execution of node "
                            f"{current_node_id} ({current_node.role}) "
                            f"round={round}; skipping retry.\n"
                            f"{type(e).__name__}: {e!r}\n{_format_exception(e)}"
                        )
                        break
                    except Exception as e:
                        print(
                            "Error during execution of node "
                            f"{current_node_id} ({current_node.role}) "
                            f"round={round} try={tries + 1}/{max_tries}: "
                            f"{type(e).__name__}: {e!r}\n{_format_exception(e)}"
                        )
                    tries += 1
            
            self.update_memory()

        self.connect_decision_node()
        tries = 0
        while tries < _DECISION_NODE_MAX_TRIES:
            try:
                await asyncio.wait_for(
                    self.decision_node.async_execute(
                        input,
                        round_idx=num_rounds,
                        record_execution_history=record_execution_history,
                        return_logprobs=record_decision_logprobs,
                        logprob_token_limit=node_logprob_token_limit,
                    ),
                    timeout=_DECISION_NODE_TIMEOUT_SECONDS,
                )
                break
            except MissingRemoteTokenUsageError:
                raise
            except Exception as e:
                print(
                    "Error during execution of external decision node "
                    f"try={tries + 1}/{_DECISION_NODE_MAX_TRIES}: "
                    f"{type(e).__name__}: {e!r}\n{_format_exception(e)}"
                )
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
        self.decision_node.execution_history = []
    
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
