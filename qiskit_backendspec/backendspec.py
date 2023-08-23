"""
A more automated way of creating FakeBackendV2 objects with custom or realistic properties.
"""


import numpy as np
import math
import pandas as pd
from scipy import stats
from typing import Union
from qiskit.transpiler import CouplingMap
from qiskit.providers.ibmq import IBMQBackend
from qiskit.transpiler import Target, InstructionProperties
from qiskit.providers.fake_provider import FakeBackendV2
from qiskit.providers.backend import Backend, BackendV2
from qiskit.exceptions import QiskitError
from qiskit_ibm_provider.ibm_qubit_properties import IBMQubitProperties
from qiskit.circuit.library import XGate, RZGate, SXGate, CXGate, ECRGate, IGate, CZGate, ECRGate, YGate, iSwapGate, U1Gate, UGate, U2Gate, U3Gate, SGate, TGate, SwapGate, PhaseGate
from qiskit.circuit import Measure, Parameter, Reset
from qiskit.compiler.transpiler import target_to_backend_properties

# TODO: Implement pulse scheduling
# TODO: Implement dynamic gates
# TODO: Remove need of look up tables (lut) in favor of built in methods for creating gates
# TODO: Allow for Non-IBM backends: Look into getting service account or try using their test cases to see how to access variables
# TODO: Implement multi two qubit gate interaction
# TODO: Implement prep0_meas1_prob and prep1_meas0_prob values 
# TODO: Implement deprecation for IBMQBackends
# TODO: Improve error handling

class BackendSpec:
    """
    A description of a backend with modifiable specifications. 

    A `BackendSpec` can either inherit specifications from an existing backend (or FakeBackend). 
    Qubit properties, instruction properties, coupling map and basis gates can all be changed.
    A `BackendSpec` is able to generate realistic fake backends when given a parent backend during initialization 
    (or through the BackendSpec.from_backend(parent) function).
    You are also able to scale or modify these inherited values. 
    BackendSpec also supports creating a unique backend from the ground up.
    """
    def __init__(self, parent: Union[Backend, BackendV2, FakeBackendV2] = None):
        """Creates a `BackendSpec` from either a parent backend inheriting values, 
        or creating a default BackendSpec object to be built up.

        Args:
            parent (Backend | BackendV2 | FakeBackendV2) = None: `Optional`, Backend from which the `BackendSpec` will inherit data from
        """

        self._seed = np.random.randint(10000000,300000000)
        np.random.seed(self._seed)
        self._two_qubit_lut = ['cx', 'ecr', 'cz', 'swap', 'iswap'] #TODO: Add more cases
        
        if parent == None:          ## default None parameter for when user doesn't want to base it on an existing backend
            self._load_base()
            return 
        
        self.false = False
        self.parent = parent

        if isinstance(parent, IBMQBackend):
            self._load_IBMQ(parent)
            self._coupling_type = 'hexagonal'

        elif "fake" in parent.name.lower():
            self.false = True
            self._load_IBM(parent)
            self._coupling_type = 'hexagonal'
            self._load_fake_data(parent.target)
            self._load_edges(self._coupling_map.graph)
            self._tuple_remover()

            self._gen_frozen_props()
            del self.parent
            del self.false
            return
            
        if isinstance(parent, BackendV2):
            self._load_IBM(parent)
            self._coupling_type = 'hexagonal'
            
        self._load_data()
        self._load_edges(self._coupling_map.graph)
        self._tuple_remover()

        self._gen_frozen_props()
        del self.parent
        del self.properties
        del self.false


    def _load_base(self):
        self._basis_gates = ['x', 'sx', 'cx', 'rz', 'id', 'reset']
        self._coupling_type = 'line' 
        self._edge_list = [(0,1), (1,0)]
        self._coupling_map = CouplingMap()
        self._coupling_map.graph.extend_from_edge_list([(0,1), (1,0)])
        self._num_qubits = 2
    
        self._qubit_properties = pd.DataFrame({
                                                "T1": [0,0],
                                                "T2": [0,0],
                                                "frequency": [0,0],
                                                "anharmonicity": [0,0],
                                                "readout_error": [0,0],
                                                "readout_length": [0,0]
                                             })
        
        self._gate_properties = pd.DataFrame({"gate": ['x', 'x', 'sx', 'sx', 'cx', 'cx', 'rz', 'rz','id', 'id', 'reset', 'reset'],
                                        "qubits": [0, 1, 0, 1, 0, 1, (0,1), (1,0),0 ,1, 0, 1],
                                        "gate_error": [0] *  12,
                                        "gate_length": [0] * 12
                                        })
        
        self._max_circuits = 100
        self._dt = 0
        self._dtm = 0
        self._bidirectional = True
        self._gen_frozen_props()

        return


    def _load_IBMQ(self,parent):
        config = parent.configuration()
        self._basis_gates = config.basis_gates
        self._num_qubits = config.num_qubits
        self._dt = config.dt
        self._dtm = config.dtm

        # Since IBMQBackend does not have a CouplingMap obj we create one (based off the edges given)
        coupling_map = config.coupling_map
        self._edge_list = [tuple(pair) for pair in coupling_map]
        self._coupling_map = CouplingMap()
        self._coupling_map.graph.extend_from_edge_list(self._edge_list)
        
        self._max_circuits = config.max_experiments
        self.properties = parent.properties()


    def _load_IBM(self,parent):
        if self.false:
          self.properties =  target_to_backend_properties(parent.target)
          self._basis_gates = list(self.properties._gates.keys())
        else:
          self._basis_gates = parent.basis_gates
          if 'reset' not in parent.basis_gates:
            parent.basis_gates.append('reset')
          self.properties = parent.properties()
          self._dtm = parent.dtm


        self._num_qubits = parent.num_qubits
        self._dt = parent.dt
        self._coupling_map = parent.coupling_map
        self._edge_list = list(parent.coupling_map.graph.edge_list())
        self._max_circuits = parent.max_circuits

    def _load_data(self):
        qubit_props = self.properties._qubits #loading qubit props
        
        _qubit_properties = pd.DataFrame(data=qubit_props)
        _qubit_properties = _qubit_properties.transpose() #transposing so proper keys are on columns

        self._qubit_properties = _qubit_properties # setting attribute


        gate_props = self.properties._gates

        gate_prop_holder = pd.DataFrame(columns=["gate", "qubits","gate_error", "gate_length"])

        for props in gate_props:
            gate = [props] * len(gate_props[props])
            qubits = list(gate_props[props].keys())

            if props not in self._two_qubit_lut:
                qubits = list(map(lambda x: x[0], qubits))

            temp_df = pd.DataFrame(gate_props[props])
            gate_prop = temp_df.values.tolist()
            temp_dict = {
                "gate": gate,
                "qubits": qubits
                }
            
            if 'reset' not in gate:
                temp_dict['gate_error'] = gate_prop[0]
                temp_dict['gate_length'] = gate_prop[1]
            else:
                temp_dict['gate_error'] = [np.nan] * len(gate_prop[0])
                temp_dict['gate_length'] = gate_prop[0]
            temp_df = pd.DataFrame(temp_dict)

            gate_prop_holder = pd.concat([gate_prop_holder, temp_df], ignore_index=True, sort=False)


        self._gate_properties = gate_prop_holder

    def _load_fake_data(self, target):
    
        gate_props = pd.DataFrame(columns = ['gate', 'qubits', 'gate_error', 'gate_length'])
        for name in target:
            if name == 'measure':
                continue
            
            gate_keys = target[name].keys()
            gate_vals = target[name].values()
            temp_df = None
            if name in self._two_qubit_lut:
                temp_df = pd.DataFrame({
                    'gate': [name] * len(gate_vals),
                    'qubits': gate_keys,
                    'gate_error': map(lambda x: x.error, gate_vals),
                    'gate_length':map(lambda x: x.duration, gate_vals)
                })

            else:
                temp_df = pd.DataFrame({
                    'gate': [name] * target.num_qubits,
                    'qubits': map(lambda x: x[0], gate_keys),
                    'gate_error': map(lambda x: x.error, gate_vals),
                    'gate_length':map(lambda x: x.duration, gate_vals)
                })

            gate_props = pd.concat([gate_props, temp_df], ignore_index = True, sort = False)
        self._gate_properties = gate_props

        t1 = [None] * target.num_qubits
        t2 = [None] * target.num_qubits
        freq = [None] * target.num_qubits
        anharm = [None] * target.num_qubits
        i = 0
        for i in range(target.num_qubits):
            t1[i] = target.qubit_properties[i].t1
            t2[i] = target.qubit_properties[i].t2
            freq[i] = target.qubit_properties[i].frequency
            anharm[i] = target.qubit_properties[i].anharmonicity



        self._qubit_properties = pd.DataFrame({
            'T1': t1,
            'T2': t2,
            'frequency': freq,
            'anharmonicity': anharm,
            'readout_error': map(lambda x: x.error, target['measure'].values()),
            'readout_length': map(lambda x: x.duration, target['measure'].values())
        })

    def _load_edges(self, graph):
        in_edges =  np.empty(self._num_qubits)
        out_edges = np.empty(self._num_qubits)

        for i in range(self._num_qubits):
            in_edges[i]  = len(graph.in_edges(i))
            out_edges[i] = len(graph.out_edges(i))
        self._bidirectional = not False in (in_edges == out_edges)
        self._qubit_properties['total_edges'] = in_edges + out_edges

    def _tuple_remover(self):
        for key in self._qubit_properties:
            self._qubit_properties[key] = self._qubit_properties[key].apply(lambda x: x[0] if isinstance(x, tuple) else x)
        for key in self._gate_properties:
            if key == 'qubits':
                continue
            self._gate_properties[key] = self._gate_properties[key].apply(lambda x: x[0] if isinstance(x, tuple) else x)
    
    @property
    def qubit_properties(self) -> pd.DataFrame:
        """Return the `qubit_properties`

        Returns: 
            qubit_properties: A `pd.DataFrame` containing the qubit properties."""
        return self._qubit_properties

    @property
    def gate_properties(self) -> pd.DataFrame:
        """Return the `gate_properties`

        Returns: 
            gate_properties: A `pd.DataFrame` containing the gate properties."""
        return self._gate_properties
    
    @property
    def bidirectional(self) -> bool:
        """Return bidirectional coupling

        Returns: 
            bidirectional: A boolean true if the coupling is bidirectional"""
        return self._bidirectional

    @property
    def max_circuits(self) -> int:
        """The maximum number of circuits that can be run in a single job.

        If there is no limit this will return None."""
        return self._max_circuits

    @property
    def edge_list(self) -> "list[tuple[int, int]]":
        """Return the `edge_list`

        Returns: 
            edge_list: A list of all the connections, in general tuples are formatted (control, target)."""
        return self._edge_list

    @property
    def coupling_map(self) -> CouplingMap:
        """Return the `coupling_map`

        Returns: 
            coupling_map: A `CouplingMap` object"""
        return self._coupling_map

    @property
    def coupling_type(self) -> str:
        """Return the `coupling_type`

        Returns: 
            coupling_type: A string containing the coupling scheme""" 
        return self._coupling_type

    @property
    def dt(self) -> Union[float, None]:
        """Return the system time resolution of input signals

        This is required to be implemented if the backend supports Pulse
        scheduling.

        Returns:
            dt: The input signal timestep in seconds. If the backend doesn't
            define ``dt`` ``None`` will be returned
        """
        return self._dt

    @property
    def basis_gates(self) -> "list[str]":
        """Return the `basis_gates`

        Returns: 
            basis_gates: A list of all the gates available to our BackendSpec."""
        return self._basis_gates

    @property
    def dtm(self) -> float:
        """Return the system time resolution of output signals

        Returns:
            dtm: The output signal timestep in seconds."""
        return self._dtm

    @property
    def num_qubits(self) -> int:
        """Return the `num_qubits`

        Returns: 
            num_qubits: The number of physical qubits"""
        return self._num_qubits

    @property
    def frozen_qubit_properties(self) -> pd.DataFrame:
        """Return the DataFrame containing frozen and unfrozen properties.

        Returns: 
            frozen_qubit_properties: A `pd.DataFrame` containing the frozen and unfrozen qubit properties."""
        return self._frozen_qubit_properties

    @property
    def frozen_gate_properties(self) -> pd.DataFrame:
        """Return the DataFrame containing frozen and unfrozen properties.

        Returns: 
            frozen_gate_properties: A `pd.DataFrame` containing the frozen and unfrozen gate properties."""
        return self._frozen_gate_properties
    
    @property
    def seed(self) -> float:
        """Return the `seed`

        Returns: 
            seed: The seed used for property sampling"""
        return self._seed

    def get_qubit_property(self,qubit: int, qubit_property: str) -> float:
      """Get `qubit_property` of `qubit` from `self.qubit_properties`.
    
        Args:   
            qubit (int): The integer id of the qubit in the coupling map and `self.qubit_properties` DataFrame.
            qubit_property (str): The specific qubit property to retrieve from `self.qubit_properties`.
        
        Returns: 
            qubit_property: The specified `qubit_property`. 
        """
      return self._qubit_properties[qubit_property][qubit]


    def get_gate_property(self, gate_name : str, gate_property: str, qubits : Union[int, "tuple[int, int]"]) -> float:
        """Get `gate_property` of gate with `gate_name`   from `self.gate_properties`.
        
        Args:
            gate_name (str): The name of the gate.
            qubits (int | tuple(int, int)): The qubit or pair of qubits the gate acts on.
            gate_property (str): The specific property wanted. 

        Returns: 
            gate_property: The specified `gate_property`.
        """
        qubits = tuple(qubits) if gate_name in self._two_qubit_lut else qubits

        temp_df = pd.DataFrame(self._gate_properties)
        temp_df = temp_df[temp_df["gate"] == gate_name]

        property_index = temp_df.index[0]

        return self._gate_properties[gate_property][property_index]

    def qubit_selector(self, qubit_property: str, lower_bound : Union[int, float], upper_bound: Union[int, float]):
        """Get the index of a qubit with properties within the inclusive range [`lower_bound`, `upper_bound`].
        
        Args:
            qubit_property (str): The name of the `QubitProperty` wanted.
            lower_bound (int): The minimum the property can be for selection.
            upper_bound (int): The maximum the property can be for selection.
        
        Returns: 
            qubit_indices: The specified qubits with `qubit_property` in range.
        """
        
        qubit_indices = []
        for i in range(len(self._qubit_properties.index)):
            if upper_bound >= self._qubit_properties[qubit_property][i] >= lower_bound:
                qubit_indices.append(i)
        return qubit_indices


    def from_backend(self, parent : Union[Backend, BackendV2, FakeBackendV2]):
        """Initialize a `BackendSpec` using a parent Backend.
        
        Args:
            parent (Backend | BackendV2 | FakeBackendV2): The parent backend to inherit data from.
        """
        self.__init__(parent)
        
 # Modifiers

    def increase_qubits(self, increase_amount : int, coupling_type: str):
        """Increase the size of a backend's coupling map by extending by `increase_amount` of qubits 
        
        Args:
            increase_amount (int): The amount of qubits to add (at minimum)
            coupling_type (str): The coupling_scheme to extend the qubits in

        Returns: 
            Image: A rgb image of the CouplingMap object. (CouplingMap.draw() object)
        
        - Note: This function uses `BackendSpec.coupling_change` meaning the `frozen_qubit_properties` and `frozen_gate_properties` will be 
        reset. `num_qubits` may also be greater than expected due to the round up done in `BackendSpec.coupling_change`.
        
        """
        if increase_amount < 0:
            raise ValueError("Please provide a number greater than zero. To decrease see self.decrease_qubits(decrease_amount, coupling_type)")
        self._num_qubits += increase_amount
        
        return self.coupling_change(coupling_type)
        
    def decrease_qubits(self, decrease_amount: int, coupling_type: str):
        """Decrease the size of a backend's coupling map by extending by `decrease_amount` of qubits 
        
        Args:
            decrease_amount (int): The (negative) amount of qubits to remove (at maximum)
            coupling_type (str): The coupling scheme to extend the qubits in
       
        Returns: 
            Image: A rgb image of the CouplingMap object. (CouplingMap.draw() object)
        - Note: This function uses `BackendSpec.coupling_change` meaning the `frozen_qubit_properties` and `frozen_gate_properties` will be 
        reset. `num_qubits` may also be greater than expected due to the round up done in `BackendSpec.coupling_change`.
        """
        if decrease_amount > 0:
            raise ValueError("Please provide a number less than zero. To increase see self.increase_qubits(increase_amount, coupling_type)")
        elif self._num_qubits + decrease_amount < 2:
            raise ValueError("Please make sure to allow enough qubits for a map to be generated")
        self._num_qubits += decrease_amount
        return self.coupling_change(coupling_type)

# Methods for coupling changes 

    def coupling_change(self, coupling_type: str):
        """Changes coupling scheme of `BackendSpec` to `couping_type`.

        Args:
            coupling_type (str): The coupling scheme to create coupling map.
        
        - Note: The `num_qubits` could change after this function, the frozen properties will reset along with the properties being resampled.
        """
        self._coupling_type = coupling_type
        num_qubits = self._num_qubits
        
        updated_map = CouplingMap()
        graph = None

        if (coupling_type == 'hexagonal'):

            m = num_qubits
            col = 1

            hex_map = updated_map.from_hexagonal_lattice(1,col, bidirectional=self._bidirectional)
            

            while hex_map.size() < m:
                col += 1
                hex_map = updated_map.from_hexagonal_lattice(1,col, bidirectional=self._bidirectional) # increase cols if the thing starts to infinite loop

            graph = hex_map.graph
            node_list = np.array(graph.node_indices())
            reversed_one = node_list[::-1]
            
            num_graph_qubits = len(node_list)

            while num_graph_qubits != m:
                for i in reversed_one:
                    if num_graph_qubits != m:
                        graph.remove_node(i)
                        num_graph_qubits -= 1


        elif (coupling_type == 'square'):

            row = math.ceil(math.sqrt(num_qubits))
            square_map = updated_map.from_grid(row, row, bidirectional=self._bidirectional)

            graph = square_map.graph

        elif (coupling_type == 'grid'):

            row = math.ceil(np.sqrt(num_qubits))
            col = math.ceil(num_qubits/row)

            grid_map = updated_map.from_grid(row, col, bidirectional=self._bidirectional)
            graph = grid_map.graph


        elif (coupling_type == 'ata'):

            ata_map = updated_map.from_full(num_qubits, bidirectional=self._bidirectional)
            graph = ata_map.graph

        else:
            raise LookupError("Please use a valid coupling type such as: hexagonal, square, ata or grid")
            
        updated_map.graph.extend_from_edge_list(graph.edge_list())
        self._coupling_map = updated_map
        self._edge_list = list(updated_map.graph.edge_list())
        self._num_qubits = updated_map.size()
            
        self._qubit_properties, self._gate_properties = self._sample_props()
        self._gen_frozen_props()
    
    

    def _generate_couple(self, num_qubits: int, coupling_type: str) -> CouplingMap:
        """Creates coupling scheme of `couping_type` with `num_qubits` (at minimum).
        
        Args:
            num_qubits (int): Minimum amount of qubits to build up map.
            coupling_type (str): The coupling scheme to create coupling map.
        
        Returns: 
            coupling_map: CouplingMap object.
        """
        
        updated_map = CouplingMap()
        graph = None

        if (coupling_type == 'hexagonal'):
            m = num_qubits
            col = 1
            hex_map = updated_map.from_hexagonal_lattice(1,col, bidirectional=self._bidirectional)

            while hex_map.size() < m:
                col += 1
                hex_map = updated_map.from_hexagonal_lattice(1,col, bidirectional=self._bidirectional) # increase cols if the thing starts to infinite loop

            graph = hex_map.graph
            node_list = np.array(graph.node_indices())
            reversed_one = node_list[::-1]

            num_graph_qubits = len(node_list)

            while num_graph_qubits != m:
                for i in reversed_one:
                    if num_graph_qubits != m:
                        graph.remove_node(i)
                        num_graph_qubits -= 1

        elif (coupling_type == 'square'):

            row = math.ceil(math.sqrt(num_qubits))
            square_map = updated_map.from_grid(row, row, bidirectional=self._bidirectional)

            graph = square_map.graph

        elif (coupling_type == 'grid'):

            row = math.ceil(np.sqrt(num_qubits))
            col = math.ceil(num_qubits/row)

            grid_map = updated_map.from_grid(row, col, bidirectional=self._bidirectional)
            graph = grid_map.graph
            
        elif (coupling_type == 'ata'):
            ata_map = updated_map.from_full(num_qubits, bidirectional=self._bidirectional)
            updated_map = ata_map
        
        else:
            raise LookupError("Please use a valid coupling type such as: hexagonal, square, ata or grid")
            
        updated_map.graph.extend_from_edge_list(graph.edge_list())

        return updated_map
    
    def scale_qubit_property(self, property_key:str, scale_factor:float):
        """Scales all qubit properties by `scale_factor`

        Args:
            property_key (str): The qubit property to scale.
            scale_factor (float): The factor to scale properties by.
        """
        self._qubit_properties[property_key] *= scale_factor

    def scale_gate_property(self, gate_name:str, property_key:str, scale_factor:float):
        """Scales all gate properties by `scale_factor`
        
        Args:
            gate_name (str): Name of gate to scale property.
            property_key (str): The qubit property to scale.
            scale_factor (float): The factor to scale properties by.
        """
        scale = self._gate_properties.loc[self._gate_properties.gate==gate_name, property_key]
        self._gate_properties.update(scale * scale_factor)

    def set_frozen_gates_property(self, frozen: bool, gate_name: str, gate_property: str):
        """Sets frozen for ALL gate properties to be static (or values to not be sampled for in backend generation).
        
        Args:
            frozen (bool): True to make property static, False to make property sampled.
            gate_name (str): The name of the gate to frozen.
            gate_property (str): Gate property to frozen.
        """
        try:
            size = len(self._frozen_gate_properties[self._frozen_gate_properties.gate == gate_name])
            self._frozen_gate_properties.loc[self._frozen_gate_properties.gate == gate_name,gate_property] = [frozen] * size
        except:
            raise KeyError(f"Gate: {gate_name} with property: {gate_property} not found")

    def set_frozen_gate_property(self, frozen: bool, gate_name: str, gate_property: str, qubits: Union[int, "list[int, int]"]):
        """Sets frozen for gate property to be static (or values to not be sampled for in backend generation).
        
        Args:
            frozen (bool): True to make property static, False to make property sampled.
            gate_name (str): The name of the gate to frozen.
            gate_property (str): Gate property to frozen.
            qubits (int | list(int, int)): The qubits that the gate is applied to
        """
        
        try:
            qubits = tuple(qubits) if gate_name in self._two_qubit_lut else qubits
            temp_df = self._frozen_gate_properties.loc[self._frozen_gate_properties["gate"] == gate_name]
            
            property_index = temp_df.loc[temp_df["qubits"] == qubits, gate_property].index[0]
            
            self._frozen_gate_properties.loc[property_index, gate_property] = frozen
        except:
            raise KeyError(f"Gate: {gate_name} with qubits: {str(qubits)} and property: {gate_property} not found")
        
    def set_frozen_qubits_property(self, frozen: bool, qubit_property: str):
        """Sets frozen for ALL qubit properties to be static (or values to not be sampled for in backend generation).
        
        Args:        
            frozen (bool): True to make property static, False to make property sampled.
            qubit_property (str): Qubit property to frozen. """
        try:
            self._frozen_qubit_properties[qubit_property] = [frozen] * len(self._frozen_qubit_properties[qubit_property])
        except:
            raise KeyError(f"Qubits with property {str(qubit_property)} not found")

        
    def set_frozen_qubit_property(self, frozen: bool, prop_key: str, qubit_id: int):
        """Sets frozen for qubit property to be static (or values to not be sampled for in backend generation).
        
        Args:
            frozen (bool): True to make property static, False to make property sampled.
            qubit_property (str): Qubit property to frozen.
            qubit_id (int): Integer value that identifies qubit."""
        try:
            self._frozen_qubit_properties[prop_key][qubit_id] = frozen
        except:
            raise KeyError(f'Qubit {str(qubit_id)} not found with {str(prop_key)}')

    def set_seed(self, seed: float):
        """Function to set seed of samplers"""
        np.random.seed(seed)
        self._seed = seed

    def set_coupling_map(self, coupling_map: CouplingMap, coupling_type: str):
        """Set Coupling Map object of backend 

        Args:
            coupling_map (CouplingMap): Coupling Map object to set
            coupling_type (str): Name of the coupling scheme

        - Note: Changing the coupling map could have the affected of changing the
         number of qubits as well as resetting the frozens and resampling the property values."""
        self._coupling_map = coupling_map
        self._edge_list = list(coupling_map.graph.edge_list())
        self._coupling_type = coupling_type
        self._num_qubits = coupling_map.size()
        
        self._qubit_properties, self._gate_properties = self._sample_props()
        self._gen_frozen_props()

    def set_bidirectional(self, bidirectional: bool):
        """Sets bidirectional coupling edges to true or false"""
        self._bidirectional = bidirectional

    def set_qubit_property(self, qubit_id: int, qubit_property: str, value: float, freeze_property : bool = False ):
        """Sets qubit property value of specific qubit
        
        Args:        
            qubit_id (int): Integer value that specifies the qubit.
            qubit_property (str): Qubit property to be set
            value (float): Value to be set
            freeze_property (bool): `Optional`, when set true set value will be frozen as static value (to not be resampled upon generation)"""
        
        self._qubit_properties.loc[qubit_id, qubit_property] = value
        if freeze_property:
            self.set_frozen_qubit_property(True, qubit_property, qubit_id)

    def set_multi_qubit_property(self, qubits: "list[int]", qubit_property: str, values: "list[Union[int, float]]", freeze_property :bool = False):
       """Sets qubit property value of specified qubits
        
        Args:
            qubits (list[int]): List of integer values that specifies the qubits.
            qubit_property (str): Qubit property to be set
            values (list[float]): list of values to be set
            freeze_property (bool): `Optional`, when set true set value will be frozen as static value (to not be resampled upon generation)"""
       for i in qubits:
           self._qubit_properties[qubit_property][qubits[i]] = values[i]
       if freeze_property:
            self.set_frozen_qubits_property(True, qubit_property, qubits)

    def set_gate_properties(self, gate_name: str, gate_property: str, values: "list[Union[int, float]]", freeze_property: bool = False):
        """Sets all gate property values of specific gate type
        
        Args:
            gate_name (str): Name of gate.
            gate_property (str): Gate property to be set.
            values (list[float]): list of values to be set
            freeze_property (bool): `Optional`, when set true set value will be frozen as static value (to not be resampled upon generation)"""

        temp_df = self._gate_properties[self._gate_properties.gate == gate_name]
        temp_df[gate_property].values[:] = values
        self._gate_properties.update(temp_df)

        if freeze_property:
            self.set_frozen_gates_property(True, gate_name, gate_property)

    def set_gate_property(self, gate_name: str, gate_property: str, qubits: Union[int, "tuple[int, int]"], value: "list[Union[int, float]]", freeze_property: bool = False):
        """Sets `gate_property` of `gate_name` acting on `qubits` to `value`. 
        
        Args:
            gate_name (str): Name of gate.
            gate_property (str): Gate property to be set.
            qubits (int | list[int, int]): Qubits that the gate is applied to (specification of gates to set)
            value (int | float): value to be set
            freeze_property (bool): `Optional`, when set true set value will be frozen as static value (to not be resampled upon generation)"""
        qubits = tuple(qubits) if gate_name in self._two_qubit_lut else qubits

        temp_df = self._gate_properties.loc[self._gate_properties.gate==gate_name]
        index = temp_df.loc[temp_df.qubits == qubits, 'gate_error'].index[0]
        self._gate_properties.loc[index, 'gate_error'] = value

        if freeze_property:
            self.set_frozen_gate_property(True, gate_name, gate_property, qubits)

    def set_gate_properties_distribution(self,gate_name: str, gate_property: str, std: float, mean: float, freeze_property : bool = False):
        """Sets gate property values of specific gates using distribution
        
        Args:
            gate_name (str): Name of gate.
            gate_property (str): Gate property to be set.
            std (float): Standard deviation of distribution.
            mean (float): Mean of distribution.
            freeze_property (bool): `Optional`, when set true set value will be frozen as static value (to not be resampled upon generation)"""
        count = len(self._edge_list) if gate_name in self._two_qubit_lut else self._num_qubits
        vals = self.sample_distribution(std, mean, count)
        

        temp_df = self._gate_properties[self._gate_properties.gate == gate_name]
        i=0
        for q in temp_df.qubits:
            self.set_gate_property(gate_name, gate_property, q, float(vals[i]))
            i+=1

        if freeze_property:
            self.set_frozen_gates_property(True, gate_name, gate_property)

    def set_qubits_properties_distribution(self, qubits: "list[int]",  qubit_property: str, std: float, mean: float, freeze_property: bool = False):
        """Sets gate property values of specific gates using distribution
        
        Args:
            qubits (list[int]): List of integer values that specifies the qubits.
            qubit_property (str): Qubit property to be set
            std (float): Standard deviation of distribution.
            mean (float): Mean of distribution.
            freeze_property (bool): `Optional`, when set true set value will be frozen as static value (to not be resampled upon generation)"""
        vals = self.sample_distribution(std, mean, len(qubits))
        self.set_multi_qubit_property(qubits, qubit_property, vals)

        if freeze_property:
            for qubit in qubits:
                self.set_frozen_qubit_property(True,qubit,  qubit_property, qubits)


    def set_max_circuits(self, max_circuits: int):
        """Set the max circuits of backend
        
        Args: 
            max_circuits (int) : The maximum number of circuits that can be run in a single job."""
        self._max_circuits = max_circuits

    def set_dt(self, dt: float):
        """Set dt of backend
        
        Args:
            dt (float): The input signal timestep in seconds.
        """
        self._dt = dt

    # distribution_xxx => [std_xxx, mean_xxx]

    # Methods for basis gates
        
    def add_basis_gate_distribution(self, new_gate : str, distribution_error : "list[float, float]", distribution_length : "list[float, float]", freeze_property: bool = False):
        """Adds new basis gate with gate errors and gate lengths sampled from distribution.
        
        Args:
            new_gate (list[int]): The name of the new gate to add.
            distribution_error (list[float, float]): A list with the standard deviation and mean of the distribution for the error rate. 
            Formatted: [std, mean]
            distribution_length (list[float, float]): Standard deviation of distribution. 
            Formatted: [std, mean]
            freeze_property (bool): `Optional`, when set true set value will be frozen as static value (to not be resampled upon generation)"""

        if new_gate in self._two_qubit_lut:
            raise LookupError(f"Please use a two qubit basis gate function to modify {new_gate}")
        elif new_gate in self._basis_gates:
            raise AttributeError(f"You already have {new_gate} as a basis gate.")
        gates = [new_gate] * self._num_qubits
        error_vals = self.sample_distribution(distribution_error[0], distribution_error[1], self._num_qubits)
        length_vals = self.sample_distribution(distribution_length[0], distribution_length[1], self._num_qubits)

        temp_df = pd.DataFrame({"gate": gates,
                                "qubits": range(self._num_qubits),
                                "gate_error": error_vals,
                                "gate_length": length_vals       
        })

        self._gate_properties = pd.concat((self._gate_properties, temp_df), ignore_index= True, sort= False)

        temp_df['gate_error'].values[:] = freeze_property
        temp_df['gate_length'].values[:] = freeze_property
        self._basis_gates.append(new_gate) 
        self._frozen_gate_properties = pd.concat((self._frozen_gate_properties, temp_df), ignore_index= True, sort= False)

    
    def add_basis_gate_numeric(self, gate_name : str, error_vals : "list[float]", length_vals : "list[float]", freeze_property: bool = False):
        """Adds new basis gate with gate errors and gate lengths using provided values.
        
        Args:
            new_gate (list[int]): The name of the new gate to add.
            error_vals (list[float]): A list of error values.
            length_vals (list[float]): A list of the length values.
            freeze_property (bool): `Optional`, when set true set value will be frozen as static value (to not be resampled upon generation)"""
        if gate_name in self._two_qubit_lut:
            raise LookupError(f"Please use a two qubit basis gate function to modify {gate_name}")
        elif len(length_vals) != self._num_qubits != len(error_vals):
            raise AttributeError("error_val AND length_vals should be the same size as many qubits as your backend has. Please use self.increase_qubits(qubit) to change the size of your backend.")
        elif gate_name in self._basis_gates:
            raise AttributeError(f"You already have {gate_name} as a basis gate.")

        gates = [gate_name] * self._num_qubits
        qubits = range(self._num_qubits)

        temp_df = pd.DataFrame({"gate": gates,
                                "qubits": qubits,
                                "gate_error": error_vals,
                                "gate_length": length_vals       
        })

        self._gate_properties = pd.concat((self._gate_properties, temp_df), ignore_index= True, sort= False)

        temp_df['gate_error'].values[:] = freeze_property
        temp_df['gate_length'].values[:] = freeze_property 
        self._frozen_gate_properties = pd.concat((self._frozen_gate_properties, temp_df), ignore_index= True, sort= False)
        self._basis_gates.append(gate_name)

    def remove_basis_gate(self, gate_name : str):
        """Removes basis gate.
        
        Args:
            gate_name (str): The name of gate to replace"""
        
        if gate_name not in self._basis_gates:
            raise LookupError(f"{gate_name} is not in the basis gates.")
        remove = self._gate_properties.loc[self._gate_properties.gate == gate_name].index
        self._gate_properties = self._gate_properties.drop(remove)
        self._gate_properties.index = range(len(self._gate_properties.index))

        self._frozen_gate_properties = self._frozen_gate_properties.drop(remove)
        self._frozen_gate_properties.index = range(len(self._frozen_gate_properties.index))
        index = self._basis_gates.index(gate_name)
        self._basis_gates.pop(index)

    def swap_basis_gate(self, old_gate : str, new_gate : str, freeze_property: bool = False):
        """Change the basis gates from `old_gate` to `new_gate`.
        
        Args:
            old_gate (str): the gate that requires replacing
            new_gate (str): the gate that replaces `old_gate`
            freeze_property (bool): `Optional`, when set true set value will be frozen as static value (to not be resampled upon generation)"""

        if new_gate in self._two_qubit_lut:
            raise LookupError(f"Please use a two qubit basis gate function to modify {new_gate}")
        elif old_gate in self._two_qubit_lut:
            raise LookupError(f"Please use a two qubit basis gate function to modify {old_gate}")
        elif old_gate not in self._basis_gates:
            raise LookupError(f"{old_gate} is not in the basis gates.")
        elif new_gate in self._basis_gates:
            raise AttributeError(f"You already have {new_gate} as a basis gate.")
        
        replace = self._gate_properties.loc[self._gate_properties.gate==old_gate]
        replace['gate'].values[:] = [new_gate] * len(replace)
        self._gate_properties.update(replace)
        
        self._frozen_gate_properties.loc[self._frozen_gate_properties.gate ==old_gate,'gate_error'][:] = freeze_property
        self._frozen_gate_properties.loc[self._frozen_gate_properties.gate ==old_gate,'gate_length'][:] = freeze_property
        self._frozen_gate_properties.loc[self._frozen_gate_properties.gate==old_gate,'gate'][:] = new_gate

        index = self._basis_gates.index(old_gate)
        self._basis_gates.pop(index)
        self._basis_gates.append(new_gate)

    def swap_basis_gate_distribution(self, old_gate : str, new_gate : str, distribution_error : "list[float, float]", distribution_length : "list[float, float]", freeze_property :bool=False):
        """Change the basis gates from `old_gate` to `new_gate` based on distributions for `error_vals` and `length_vals`.
        
        Args:
            old_gate (str): the gate that requires replacing
            new_gate (str): the gate that replaces `old_gate`
            distribution_error (list[float, float]): A list with the standard deviation and mean of the distribution for the error rate. 
            Formatted: [std, mean]
            distribution_length (list[float, float]): Standard deviation of distribution. 
            Formatted: [std, mean]
            freeze_property (bool): `Optional`, when set true set value will be frozen as static value (to not be resampled upon generation)"""

        if new_gate in self._two_qubit_lut:
            raise LookupError(f"Please use a two qubit basis gate function to modify {new_gate}")
        elif old_gate in self._two_qubit_lut:
            raise LookupError(f"Please use a two qubit basis gate function to modify {old_gate}")
        elif old_gate not in self._basis_gates:
            raise LookupError(f"{old_gate} is not in the basis gates.")
        elif new_gate in self._basis_gates:
            raise AttributeError(f"You already have {new_gate} as a basis gate.")
        self.remove_basis_gate(old_gate)
        self.add_basis_gate_distribution(new_gate, distribution_error, distribution_length, freeze_property)


    def swap_basis_gate_numeric(self, old_gate : str, new_gate : str, error_vals : "list[float]", length_vals : "list[float]", freeze_property: bool = False):
        """Change the basis gates from `old_gate` to `new_gate` based on directly inputted values for `error_vals` and `length_vals`.
        
        Args:
            old_gate (str): the gate that requires replacing
            new_gate (str): the gate that replaces `old_gate`
            error_vals (list[float]): a list of error values for the gates.
            length_vals (list[float]): a list of length values for the gates. 
            freeze_property (bool): `Optional`, when set true set value will be frozen as static value (to not be resampled upon generation)"""


        if new_gate in self._two_qubit_lut:
            raise LookupError(f"Please use a two qubit basis gate function to modify {new_gate}")
        elif old_gate in self._two_qubit_lut:
            raise LookupError(f"Please use a two qubit basis gate function to modify {old_gate}")
        elif old_gate not in self._basis_gates:
            raise LookupError(f"{old_gate} is not in the basis gates.")
        elif new_gate in self._basis_gates:
            raise AttributeError(f"You already have {new_gate} as a basis gate.")
        elif len(length_vals) != self._num_qubits != len(error_vals):
            raise AttributeError("error_val AND length_vals should be the same size as many qubits as your backend has. Please use self.increase_qubits(qubit) to change the size of your backend.")
        self.remove_basis_gate(old_gate)
        self.add_basis_gate_numeric(new_gate, error_vals, length_vals, freeze_property)
    
    def swap_2q_basis_gate(self, old_gate : str, new_gate : str, freeze_property:bool = False):
        """Change the 2-qubit gates from `old_gate` to `new_gate`.
        
        Args:
            old_gate (str): the gate that requires replacing
            new_gate (str): the gate that replaces `old_gate`
            freeze_property (bool): `Optional`, when set true set value will be frozen as static value (to not be resampled upon generation)"""

        if old_gate not in self._basis_gates:
            raise LookupError(f"{old_gate} is not in the basis gates.")
        elif old_gate not in self._two_qubit_lut:
            raise LookupError(f"{old_gate} is not a two qubit gate.")
        elif new_gate in self._basis_gates:
            raise AttributeError(f"You already have {new_gate} as a basis gate.")
        elif new_gate not in self._two_qubit_lut:
            raise LookupError(f"{new_gate} is not a two qubit gate.")

        replace = self._gate_properties.loc[self._gate_properties.gate==old_gate]
        replace['gate'].values[:] = [new_gate] * len(replace)
        self._gate_properties.update(replace)
        self._frozen_gate_properties.loc[self._frozen_gate_properties.gate==old_gate,'gate_error'][:] = freeze_property
        self._frozen_gate_properties.loc[self._frozen_gate_properties.gate==old_gate,'gate_length'][:] = freeze_property

        index = self._basis_gates.index(old_gate)
        self._basis_gates.pop(index)
        self._basis_gates.append(new_gate)


    def swap_2q_basis_gate_distribution(self, old_gate : str, new_gate : str, distribution_error : "list[float, float]", distribution_length : "list[float, float]", freeze_property: bool = False):
        """Applies a new 2-qubit gate with a specified distribution for the `gate_error`s and the `gate_length`s.
    
        Args:
            old_gate (str): the gate that requires replacing
            new_gate (str): the gate that replaces `old_gate`
            distribution_error (list[float, float]): A list with the standard deviation and mean of the distribution for the error rate. 
            Formatted: [std, mean]
            distribution_length (list[float, float]): Standard deviation of distribution. 
            Formatted: [std, mean]
            freeze_property (bool): `Optional`, when set true set value will be frozen as static value (to not be resampled upon generation)"""

        if old_gate not in self._basis_gates:
            raise LookupError(f"{old_gate} is not in the basis gates.")
        elif old_gate not in self._two_qubit_lut:
            raise LookupError(f"{old_gate} is not a two qubit gate.")
        elif new_gate in self._basis_gates:
            raise AttributeError(f"You already have {new_gate} as a basis gate.")
        elif new_gate not in self._two_qubit_lut:
            raise LookupError(f"{new_gate} is not a two qubit gate.")

        replace = self._gate_properties.loc[self._gate_properties.gate==old_gate]
        qubits = replace.qubits

        count = len(replace)
        gate = [new_gate] * count

        gate_error = self.sample_distribution(distribution_error[0], distribution_error[1], count)
        gate_length = self.sample_distribution(distribution_length[0], distribution_length[1], count)

        self._gate_properties.drop(replace.index)
        self._frozen_gate_properties.drop(replace.index)
        temp_df = pd.DataFrame({
            "gate" : gate,
            "qubits": qubits,
            "gate_error": gate_error,
            "gate_length": gate_length
        })
        self._gate_properties.update(temp_df)

        self._frozen_gate_properties.loc[self._frozen_gate_properties.gate==old_gate, 'gate_error'][:] = freeze_property
        self._frozen_gate_properties.loc[self._frozen_gate_properties.gate==old_gate, 'gate_length'][:]= freeze_property
        self._frozen_gate_properties.loc[self._frozen_gate_properties.gate==old_gate, 'gate'][:] = new_gate
        
        index = self._basis_gates.index(old_gate)
        self._basis_gates.pop(index)
        self._basis_gates.append(new_gate)

    
    def swap_2q_basis_gate_numeric(self, old_gate : str, new_gate : str, gate_error : "list[float]", gate_length : "list[float]", freeze_property:bool=False):
        """Changes the basis gates of a 2-qubit system from the `old_gate`s to the `new_gate`s, while also applying the required `gate_errors`
        with the respective `gate_length`s.
    
        Args:     
        old_gate (str): The gate that requires replacing
        new_gate (str): The gate that replaces `old_gate`
        gate_error (list[float]): A list of errors corresponding to the `new_gates`
        gate_length (list[float]): A list of length of the gate
        freeze_property (bool): `Optional`, when set true set value will be frozen as static value (to not be resampled upon generation)"""

        if old_gate not in self._basis_gates:
            raise LookupError(f"{old_gate} is not in the basis gates.")
        elif old_gate not in self._two_qubit_lut:
            raise LookupError(f"{old_gate} is not a two qubit gate.")
        elif new_gate in self._basis_gates:
            raise AttributeError(f"You already have {new_gate} as a basis gate.")
        elif new_gate not in self._two_qubit_lut:
            raise LookupError(f"{new_gate} is not a two qubit gate.")

        replace = self._gate_properties.loc[self._gate_properties.gate==old_gate]
        qubits = replace.qubits

        count = len(replace)
        gate = [new_gate] * count

        self._gate_properties.drop(replace.index)
        self._frozen_gate_properties.drop(replace.index)
        temp_df = pd.DataFrame({
            "gate" : gate,
            "qubits": qubits,
            "gate_error": gate_error,
            "gate_length": gate_length
        })
        self._gate_properties.update(temp_df)

        self._frozen_gate_properties.loc[self._frozen_gate_properties.gate==old_gate, 'gate_error'][:] = freeze_property
        self._frozen_gate_properties.loc[self._frozen_gate_properties.gate==old_gate, 'gate_length'][:]= freeze_property
        self._frozen_gate_properties.loc[self._frozen_gate_properties.gate==old_gate, 'gate'][:] = new_gate

        index = self._basis_gates.index(old_gate)
        self._basis_gates.pop(index)
        self._basis_gates.append(new_gate)

### Freezing for static values
    def _gen_frozen_props(self):
        """Generate pd.DataFrame for gates and qubits to freeze, where frozen properties are set with a specified error type."""

        qubit_frozen = self._qubit_properties.copy()
        gate_frozen = self._gate_properties.copy()

        for col in qubit_frozen.columns:
            qubit_frozen[col].values[:] = False
        
        gate_frozen['gate_error'].values[:] = False
        gate_frozen['gate_length'].values[:] = False


        self._frozen_qubit_properties = qubit_frozen
        self._frozen_gate_properties = gate_frozen


    def _apply_freeze(self, dfs : pd.DataFrame) -> "list[np.ndarray, np.ndarray]":
        """
        Keeps frozen properties static while sampling for unfrozen properties
    
        Args:
            dfs (pd.DataFrame): The DataFrames to which the frozen properties are applied.

        Returns: 
            frozen_props: A list of the DataFrames with both sampled and static values.""" 

        qubit_df = dfs[0]
        gate_df = dfs[1]
        for col in qubit_df.columns:
            freeze_row = self._frozen_qubit_properties[col].values[:]

            qubit_df[col].values[:] = qubit_df[col].values[:] * (1-freeze_row) + self._qubit_properties[col].values[:] * freeze_row

        for col in ['gate_error', "gate_length"]:
            freeze_row = self._frozen_gate_properties[col].values

            gate_df[col].values[:] = gate_df[col].values[:] * (1-freeze_row) + self._gate_properties[col].values[:] * freeze_row
        return [qubit_df, gate_df]


### Samplers
    def _sample_props(self) -> "list[np.ndarray, np.ndarray]":
        """Generates DataFrames from samples for qubits and gates.
        
        Returns:
            sampled_props (list[np.ndarray, np.ndarray]): List of DataFrames with sampled values""" 

        qubit_df = pd.DataFrame()
        i = 0
        for prop in self._qubit_properties.columns:
            qubit_df.insert(i, prop, self._sample_qubits(prop, self._num_qubits), True)

        gate_df = pd.DataFrame(columns=["gate", "qubits","gate_error", "gate_length"])

        for gate in self._basis_gates:

            count = len(self._edge_list) if gate in self._two_qubit_lut else self._num_qubits

            gate_list = [gate] * count
            qubits = list(range(count)) if gate not in self._two_qubit_lut else self._edge_list

            # try:
            gate_error = self._sample_gates(gate, 'gate_error', count) if 'reset' not in gate else [np.nan] * count
            # except:
            #     print(gate)
            gate_length = self._sample_gates(gate, 'gate_length', count)

            temp_dict = {"gate": gate_list, 'qubits': qubits, 'gate_error': gate_error, "gate_length":gate_length}
            temp_df = pd.DataFrame(temp_dict)
            gate_df = pd.concat((gate_df, temp_df), ignore_index=True, sort=False)

            gate_df['gate_error'] = gate_df['gate_error']
            gate_df['gate_length'] = gate_df['gate_length']


        return qubit_df, gate_df

    def _sample_gates(self, gate : str, prop_key : str, count : int) -> np.ndarray:
        """Generates a normal distribution using the specified `prop_key` for the properties of the gates, and samples a `count` number of values from 
        the normal distribution.
    
        Args:
            prop_key (str): The property for which the sample is being generated.
            count (int): number of samples that will be generated.

        Returns:
            sampled_gates (np.ndarray): Numpy array containing sampled values using distribution found in `self.gate_properties`""" 
        mini_df = self._gate_properties.loc[self._gate_properties['gate'] == gate]

        data = mini_df[prop_key]

        mu = np.mean(data)
        sigma = np.std(data)

        distribution = stats.norm(
            loc=mu,
            scale=sigma
        )

        sample = distribution.rvs(size=10000)

        series = pd.Series(sample)
        sample = series.sample(n=count) # random_state for seed?

        sample_output = np.array(sample.abs())

        return sample_output # returns a pd series (or does it have to be a df?)


    def _sample_qubits(self, prop_key : str, count : int) -> np.ndarray: 
        """Generates a normal distribution using the specified `prop_key` for the properties of the qubits, and samples a `count` number of values from 
        the normal distribution.
    
        Args:    
            prop_key (str): The property for which the sample is being generated.
            count (int): number of samples that will be generated.

        Returns:
            sampled_qubits (np.ndarray): Numpy array containing sampled values using distribution found in `self.qubit_properties`"""
        data = self._qubit_properties[prop_key]

        mu = np.mean(data)
        sigma = np.std(data)

        distribution = stats.norm(
            loc=mu,
            scale=sigma
        )

        sample = distribution.rvs(size=10000)

        series = pd.Series(sample)
        sample = series.sample(n=count) # random_state for seed?

        sample_output = np.array(sample.abs())

        return sample_output # returns a pd series (or does it have to be a df?)
    
    def sample_distribution(self, std : int, mean : int, count : int) -> np.ndarray:
        """Generates a normal distribution with an inputted `mean`, `standard deviation`, and a sample `count` number of values from the normal
        distribution.
    
        Args:
            std (float): Standard deviation of the distribution.
            mean (float): Mean of the distribution.
            count (int): number of samples that will be generated.

        sampled_distribution (np.ndarray): Numpy array containing sampled values using distribution provided by user"""
        distribution = stats.norm(
            loc=mean,
            scale=std
        )
        sample = distribution.rvs(size=10000)

        series = pd.Series(sample)
        sample = series.sample(n=count) # random_state for seed?
        sample_output = np.array(sample.abs())

        return sample_output


### New backend generation code

    def _gen_target(self, qubits_df : pd.DataFrame) -> Target:
        """Generates a `Target` from `BackendSpec` based on the backend's `QubitProperties`.

        Args:
            `qubits_df`: DataFrame containing QubitProperties

        Returns: 
            target (Target): Target object with qubit properties from `qubit_df`"""
        
        num_qubits = self._num_qubits
        _target = Target(
                num_qubits = num_qubits,
                dt = self._dt,
                qubit_properties = [
                    IBMQubitProperties( 
                        t1 = qubits_df['T1'][i],
                        t2 = qubits_df['T2'][i],
                        frequency = qubits_df['frequency'][i],
                        anharmonicity= qubits_df['anharmonicity'][i],
                    )
                    for i in range(num_qubits)
                ],
            )
        return _target

    def _gen_inst_props(self, props : pd.DataFrame) -> dict:
        """Generates a dictionary from `BackendSpec` of the `InstructionProperties` of the gates.
    
        Args:
            props (pd.DataFrame): DataFrame containing the gate properties of the backend.

        Returns: 
            instruction_dict (dict): Dictionary containing all gates described in `BackendSpec`'s `self.gate_properties`.
        """
        gates_lut = {
                'x': XGate,
                'y': YGate,
                's': SGate,
                't': TGate,
                'u' : UGate,
                'u1': U1Gate,
                'p': PhaseGate,
                'u2': U2Gate,
                'u3': U3Gate,
                'rz': RZGate,
                'sx': SXGate,
                'cx': CXGate,
                'cz': CZGate,
                'swap': SwapGate,
                'iswap': iSwapGate,
                'ecr': ECRGate,
                'id': IGate,
                'reset': Reset,
                "measure": Measure
        }

        inst_dict = {}
        
        gates = list(set(props['gate']))

        for gate in gates:
            gate_class = gates_lut[gate]

            mini_df = props.loc[props['gate'] == gate].transpose()

            gate_called = None
            if gate == 'rz':
                gate_called = gate_class(Parameter('theta'))
            elif gate == 'u':
                gate_called = gate_class(Parameter('theta'), Parameter('phi'), Parameter('lambda'))
            elif gate == 'u1':
                gate_called = gate_class(Parameter('theta'))
            elif gate == 'p':
                gate_called = gate_class(Parameter('theta'))
            elif gate == 'u2':
                gate_called = gate_class(Parameter('theta'), Parameter('phi'))
            elif gate == 'u3':
                gate_called = gate_class(Parameter('theta'), Parameter('phi'), Parameter('lambda'))
            else:
                gate_called = gate_class()

            
            mini_dict = {
                gate: (
                gate_called,
                {
                (mini_df[i]['qubits'],) if isinstance(mini_df[i]['qubits'], int) else tuple(mini_df[i]['qubits']) : InstructionProperties(
                    error = mini_df[i]['gate_error'],
                    duration = mini_df[i]['gate_length']
            )
                for i in mini_df
            })
            }
            inst_dict[gate] = mini_dict[gate]

        props = self._qubit_properties
        measure_dict = {
            'measure': (
            Measure(),
            {
                (i,) : InstructionProperties(
                error = props['readout_error'][i],
                duration = props['readout_length'][i]
                )
                for i in list(props.index)
            })
        }
        inst_dict['measure'] = measure_dict['measure']

        return inst_dict


    def new_backend(self) -> FakeBackendV2:
        """Generates new backend using the properties stored in the `BackendSpec` (along with static freezing)
 
        Returns: 
            new_backend (FakeBackendV2): A `FakeBackendV2` object with properties that follow the BackendSpec."""

        new_backend = FakeBackendV2()
        new_backend._coupling_map = self._coupling_map

        qubits_df, gates_df = self._apply_freeze(self._sample_props())

        basis_gates = list(self._basis_gates)
        # new_backend.qubits_df = qubits_df
        # new_backend.gates_df = gates_df
        
        _target = self._gen_target(qubits_df)

        inst_props = self._gen_inst_props(gates_df)

        if "measure" not in self._basis_gates:
                basis_gates.append("measure")

        for gate in basis_gates:
                try:
                    _target.add_instruction(*inst_props[gate])
                except:
                    raise QiskitError(f"{gate} is not a valid basis gate")
        new_backend._target = _target
        new_backend._basis_gates = basis_gates
        return new_backend

