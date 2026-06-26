import shortuuid
from typing import List, Any, Optional, Dict
from abc import ABC, abstractmethod
import warnings
import asyncio


class Node(ABC):
    """
    Represents a processing unit within a graph-based framework.

    This class encapsulates the functionality for a node in a graph, managing
    connections to other nodes, handling inputs and outputs, and executing
    assigned operations. It supports both individual and aggregated processing modes.

    Attributes:
        id (uuid.UUID): Unique identifier for the node.
        agent_type(str): Associated agent name for node-specific operations.
        spatial_predecessors (List[Node]): Nodes that precede this node in the graph.
        spatial_successors (List[Node]): Nodes that succeed this node in the graph.
        inputs (List[Any]): Inputs to be processed by the node.
        outputs (List[Any]): Communication outputs produced after node execution.
        entropy_samples (List[Any]): Same-context samples used only for KHEAT uncertainty estimation.
        raw_inputs (List[Any]): The original input contains the question or math problem.
        last_memory (Dict[str,List[Any]]): Input and output of the previous timestamp.
        
    Methods:
        add_predecessor(operation): 
            Adds a node as a predecessor of this node, establishing a directed connection.
        add_successor(operation): 
            Adds a node as a successor of this node, establishing a directed connection.
        memory_update():
            Update the last_memory.
        get_spatial_info():
            Get all of the info from spatial spatial_predecessors.
        execute(**kwargs): 
            Processes the inputs through the node's operation, handling each input individually.
        _execute(input, **kwargs): 
            An internal method that defines how a single input is processed by the node. This method should be implemented specifically for each node type.
        _process_inputs(raw_inputs, spatial_info, temporal_info, **kwargs)->List[Any]:
            An internal medthod to process the raw_input, the spatial info and temporal info to get the final inputs.
    """

    def __init__(self, 
                 id: Optional[str],
                 agent_name:str="",
                 domain:str="", 
                 llm_name:str = "",
                 ):
        """
        Initializes a new Node instance.
        """
        self.id:str = id if id is not None else shortuuid.ShortUUID().random(length=4)
        self.agent_name:str = agent_name
        self.domain:str = domain
        self.llm_name:str = llm_name
        self.spatial_predecessors: List[Node] = []
        self.spatial_successors: List[Node] = []
        self.temporal_predecessors: List[Node] = []
        self.temporal_successors: List[Node] = []
        self.inputs: List[Any] = []
        self.outputs: List[Any] = []
        self.entropy_samples: List[Any] = []
        self.raw_inputs: List[Any] = []
        self.role = ""
        self.last_memory: Dict[str,List[Any]] = {'inputs':[],'outputs':[],'raw_inputs':[],'entropy_samples':[]}
        self.execution_history: List[Dict[str, Any]] = []

    @property
    def node_name(self):
        return self.__class__.__name__
    
    def add_predecessor(self, operation: 'Node', st='spatial'):
        if st == 'spatial' and operation not in self.spatial_predecessors:
            self.spatial_predecessors.append(operation)
            operation.spatial_successors.append(self)
        elif st == 'temporal' and operation not in self.temporal_predecessors:
            self.temporal_predecessors.append(operation)
            operation.temporal_successors.append(self)

    def add_successor(self, operation: 'Node', st='spatial'):
        if st =='spatial' and operation not in self.spatial_successors:
            self.spatial_successors.append(operation)
            operation.spatial_predecessors.append(self)
        elif st == 'temporal' and operation not in self.temporal_successors:
            self.temporal_successors.append(operation)
            operation.temporal_predecessors.append(self)

    def remove_predecessor(self, operation: 'Node', st='spatial'):
        if st =='spatial' and operation in self.spatial_predecessors:
            self.spatial_predecessors.remove(operation)
            operation.spatial_successors.remove(self)
        elif st =='temporal' and operation in self.temporal_predecessors:
            self.temporal_predecessors.remove(operation)
            operation.temporal_successors.remove(self)

    def remove_successor(self, operation: 'Node', st='spatial'):
        if st =='spatial' and operation in self.spatial_successors:
            self.spatial_successors.remove(operation)
            operation.spatial_predecessors.remove(self)
        elif st =='temporal' and operation in self.temporal_successors:
            self.temporal_successors.remove(operation)
            operation.temporal_predecessors.remove(self)

    def clear_connections(self):
        self.spatial_predecessors: List[Node] = []
        self.spatial_successors: List[Node] = []
        self.temporal_predecessors: List[Node] = []
        self.temporal_successors: List[Node] = []        
    
    def update_memory(self):
        self.last_memory['inputs'] = self.inputs
        self.last_memory['outputs'] = self.outputs
        self.last_memory['raw_inputs'] = self.raw_inputs
        self.last_memory['entropy_samples'] = self.entropy_samples

    def get_spatial_info(self)->Dict[str,Dict]:
        """ Return a dict that maps id to info. """
        spatial_info = {}
        if self.spatial_predecessors is not None:
            for predecessor in self.spatial_predecessors:
                predecessor_outputs = predecessor.outputs
                if isinstance(predecessor_outputs, list) and len(predecessor_outputs):
                    predecessor_output = predecessor_outputs[-1]
                elif isinstance(predecessor_outputs, list) and len(predecessor_outputs)==0:
                    continue
                else:
                    predecessor_output = predecessor_outputs
                spatial_info[predecessor.id] = {"role":predecessor.role,"output":predecessor_output}

        return spatial_info

    def get_temporal_info(self)->Dict[str,Any]:
        temporal_info = {}
        if self.temporal_predecessors is not None:
            for predecessor in self.temporal_predecessors:
                predecessor_outputs = predecessor.last_memory['outputs']
                if isinstance(predecessor_outputs, list) and len(predecessor_outputs):
                    predecessor_output = predecessor_outputs[-1]
                elif isinstance(predecessor_outputs, list) and len(predecessor_outputs)==0:
                    continue
                else:
                    predecessor_output = predecessor_outputs
                temporal_info[predecessor.id] = {"role":predecessor.role,"output":predecessor_output}
        
        return temporal_info
    
    def _record_execution(self, round_idx: int, spatial_info: Dict[str, Any], temporal_info: Dict[str, Any]):
        if round_idx is None:
            return
        spatial_info_snapshot = {
            node_id: dict(info)
            for node_id, info in spatial_info.items()
        }
        temporal_info_snapshot = {
            node_id: dict(info)
            for node_id, info in temporal_info.items()
        }
        self.execution_history.append({
            "round": round_idx,
            "outputs": list(self.outputs),
            "communication_outputs": list(self.outputs),
            "entropy_samples": list(self.entropy_samples),
            "spatial_predecessors": list(spatial_info.keys()),
            "temporal_predecessors": list(temporal_info.keys()),
            "spatial_info": spatial_info_snapshot,
            "temporal_info": temporal_info_snapshot,
        })

    @staticmethod
    def _as_output_list(result: Any) -> List[Any]:
        if isinstance(result, list):
            return result
        return [result]

    def _set_execution_outputs(self, results: List[Any]):
        output_groups = [self._as_output_list(result) for result in results]
        self.entropy_samples = [
            output
            for output_group in output_groups
            for output in output_group
        ]
        self.outputs = output_groups[0] if output_groups else []

    def execute(self, input:Any, **kwargs):
        round_idx = kwargs.pop("round_idx", None)
        num_entropy_samples = kwargs.pop("num_entropy_samples", 1)
        record_execution_history = kwargs.pop("record_execution_history", True)
        num_entropy_samples = max(1, int(num_entropy_samples))
        self.outputs = []
        self.entropy_samples = []
        spatial_info:Dict[str,Dict] = self.get_spatial_info()
        temporal_info:Dict[str,Dict] = self.get_temporal_info()
        results = [
            self._execute(input, spatial_info, temporal_info, **kwargs)
            for _ in range(num_entropy_samples)
        ]

        self._set_execution_outputs(results)
        if record_execution_history:
            self._record_execution(round_idx, spatial_info, temporal_info)
        return self.outputs


    async def async_execute(self, input:Any, **kwargs):
        round_idx = kwargs.pop("round_idx", None)
        num_entropy_samples = kwargs.pop("num_entropy_samples", 1)
        record_execution_history = kwargs.pop("record_execution_history", True)
        num_entropy_samples = max(1, int(num_entropy_samples))

        self.outputs = []
        self.entropy_samples = []
        spatial_info:Dict[str,Any] = self.get_spatial_info()
        temporal_info:Dict[str,Any] = self.get_temporal_info()
        tasks = [
            asyncio.create_task(self._async_execute(input, spatial_info, temporal_info, **kwargs))
            for _ in range(num_entropy_samples)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        self._set_execution_outputs(results)
        if record_execution_history:
            self._record_execution(round_idx, spatial_info, temporal_info)
        return self.outputs
               
    @abstractmethod
    def _execute(self, input:List[Any], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """

    @abstractmethod
    async def _async_execute(self, input:List[Any], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """

    @abstractmethod
    def _process_inputs(self, raw_inputs:List[Any], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs)->List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """
