import numpy as np
import math
import pandas as pd
from scipy import stats
import warnings
from typing import Optional, List, Tuple, Union
from qiskit.transpiler import CouplingMap
from qiskit.providers.basicaer import BasicAer
from qiskit.providers.ibmq import IBMQBackend
from qiskit.transpiler import Target, InstructionProperties, QubitProperties
from qiskit.providers.fake_provider import FakeBackendV2
from qiskit.providers.backend import Backend, BackendV2
from qiskit.providers.options import Options
from qiskit.exceptions import QiskitError
from qiskit_ibm_provider.ibm_qubit_properties import IBMQubitProperties
from qiskit.circuit.library import XGate, RZGate, SXGate, CXGate, ECRGate, IGate, CZGate, ECRGate, YGate, iSwapGate, U1Gate, UGate, U2Gate, U3Gate, SGate, TGate, SwapGate, PhaseGate
from qiskit.circuit import Measure, Parameter, Delay, Reset
from qiskit.compiler.transpiler import target_to_backend_properties


# TODO: Make it so users do not have to flag data added by hand

pd.options.mode.chained_assignment = None  # default='warn'


class BackendSpec:
    def __init__(self, parent: Union[FakeBackendV2, Backend, BackendV2]=None):
        
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

            self._gen_flag()
            del self.parent
            del self.false
            return

            
        if isinstance(parent, BackendV2):
            self._load_IBM(parent)
            self._coupling_type = 'hexagonal'
            
        self._load_data()
        self._load_edges(self._coupling_map.graph)
        self._tuple_remover()

        self._gen_flag()
        del self.parent
        del self.properties
        del self.false


    def _load_base(self):
        self._basis_gates = ['x', 'sx', 'cx', 'rz', 'id', 'reset']
        self._coupling_type = 'hexagonal' 
        self._edge_list = [(0,1), (1,0)]
        self._coupling_map = CouplingMap()
        self._coupling_map.graph.extend_from_edge_list([(0,1), (1,0)])
        self._num_qubits = 2
        


        self._qubit_properties = pd.DataFrame({"T1": [0,0],
                                            "T2": [0,0],
                                            "frequency": [0,0],
                                            "anharmonicity": [0,0],
                                            "readout_error": [0,0],
                                            "prob_meas0_prep1": [0,0],
                                            "prob_meas1_prep0": [0,0],
                                            "readout_length": [0,0]
                                            })
        
        self._gate_properties = pd.DataFrame({"gate": ['x', 'sx', 'rz', 'id', 'x', 'sx', 'rz', 'id','cx', 'cx', 'reset', 'reset'],
                                        "qubits": [0, 0, 0, 0,  1, 1, 1 ,1, (0,1), (1,0), 0, 1],
                                        "gate_error": [0] *  12,
                                        "gate_length": [0] * 12
                                        })
        
        self._max_circuits = 100
        self._dt = 0
        self._dtm = 0
        self._bidirectional = True
        self._gen_flag()

        return

#TODO: Add calibartion
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

    
### Getter
    @property
    def qubit_properties(self) -> pd.DataFrame:
        return self._qubit_properties

    @property
    def gate_properties(self) -> pd.DataFrame:
        return self._gate_properties
    
    @property
    def bidirectional(self):
        return self._bidirectional

    @property
    def max_circuits(self) -> int:
        return self._max_circuits

    @property
    def edge_list(self) -> list[tuple[int, int]]:
        return self._edge_list

    @property
    def coupling_map(self) -> CouplingMap:
        return self._coupling_map

    @property
    def coupling_type(self) -> str:
        return self._coupling_type

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def basis_gates(self) -> list[str]:
        return self._basis_gates

    @property
    def dtm(self) -> float:
        return self._dtm

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def qubit_flag(self) -> pd.DataFrame:
        return self._qubit_flag

    @property
    def gate_flag(self) -> pd.DataFrame:
        return self._gate_flag
    
    @property
    def seed(self) -> float:
        return self._seed

    def get_qubit_property(self, qubit: int, qubit_property: str) -> float:
      """Get `qubit_property` of `qubit` from `self.qubit_properties`.
    
        Parameters
        ----------
        
        qubit : int
        The integer id of the qubit in the coupling map and `self.qubit_properties` DataFrame.
        qubit_property: str
        The specific qubit property to retrieve from `self.qubit_properties`.
        #### Returns: float
         The specified `qubit_property`. 
        """
      return self._qubit_properties[qubit_property][qubit]


    def get_gate_property(self, gate_name : str, qubits : Union[int, tuple[int, int]], gate_property: str) -> str:
        """Get `gate_property` of gate with `gate_name`   from `self.gate_properties`.
        
        Parameters
        ----------

        gate_name : str
            The name of the gate.
        qubits: int | tuple(int, int)
            The qubit or pair of qubits the gate acts on.
        gate_property: str
            The specific property wanted. 
        #### Returns: float
            The specified `qubit_property`.
        """

        temp_df = pd.DataFrame(self._gate_properties)
        temp_df = temp_df[temp_df["gate"] == gate_name]
        if gate_name in self._two_qubit_lut:
            temp_df = temp_df[temp_df["qubits"] == qubits]
        else:
            temp_df = temp_df[temp_df["qubits"] == tuple(qubits)]

        property_index = temp_df.index[0]

        return self._gate_properties[gate_property][property_index]

    def qubit_selector(self, qubit_property: str, lower_bound : Union[int, float], upper_bound: Union[int, float]):
        """Get the index of a qubit with properties within the inclusive range [`lower_bound`, `upper_bound`].
        
        Parameters
        ----------

        qubit_property : str
            The name of the `QubitProperty` wanted.
        lower_bound: int
            The minimum the property can be for selection.
        upper_bound: int
            The maximum the property can be for selection.
    #### Returns: float
         The specified `qubit_property`.
        """
        
        qubit_indices = []
        for i in range(len(self._qubit_properties.index)):
            if upper_bound >= self._qubit_properties[qubit_property][i] >= lower_bound:
                qubit_indices.append(i)

        return qubit_indices


    def from_backend(self, parent : Union[Backend, BackendV2, FakeBackendV2]):
        """Initialize a `BackendSpec` using a parent Backend.
        
        Parameters
        ----------

        parent : Backend | BackendV2 | FakeBackendV2
        """
        self.__init__(parent)
        
 # Modifiers

    def increase_qubits(self, increase_amount : int, coupling_type: str):
        """Increase the size of a backend's coupling map by extending by `increase_amount` of qubits 
        - Note: This function uses `BackendSpec.coupling_change` meaning the `qubit_flags` and `gate_flags` will be 
        reset. `num_qubits` may also be greater than expected due to the round up done in `BackendSpec.coupling_change`.
        
        Parameters
        ----------

        increase_amount : int
            The amount of qubits to add (at minimum)
        coupling_type: str
            The coupling_scheme to extend the qubits in
    #### Returns: Image
            A rgb image of the CouplingMap object. (CouplingMap.draw() object)
        """
        if increase_amount < 0:
            raise ValueError("Please provide a number greater than zero. To decrease see self.decrease_qubits(decrease_amount, coupling_type)")
        self._num_qubits += increase_amount
        
        return self.coupling_change(coupling_type)
        
    def decrease_qubits(self, decrease_amount: int, coupling_type: str):
        """Decrease the size of a backend's coupling map by extending by `decrease_amount` of qubits 
        - Note: This function uses `BackendSpec.coupling_change` meaning the `qubit_flags` and `gate_flags` will be 
        reset. `num_qubits` may also be greater than expected due to the round up done in `BackendSpec.coupling_change`.
        
        Parameters
        ----------

        decrease_amount : int
            The amount of qubits to remove (at maximum)
        coupling_type: str
            The coupling scheme to extend the qubits in
    #### Returns: Image
            A rgb image of the CouplingMap object. (CouplingMap.draw() object)
        """
        if decrease_amount > 0:
            raise ValueError("Please provide a number less than zero. To increase see self.increase_qubits(increase_amount, coupling_type)")
        self._num_qubits += decrease_amount
        return self.coupling_change(coupling_type)

# Methods for coupling changes 

    def coupling_change(self, coupling_type: str):
        """Changes coupling scheme of `BackendSpec` to `couping_type`.
        - Note: The `num_qubits` could change after this function, the static value flags will reset along with the properties being resampled.

        Parameters
        ----------

        coupling_type: str
            The coupling scheme to create coupling map.
    #### Returns: Image
            A rgb image of the CouplingMap object. (CouplingMap.draw() object)
        """
        self._coupling_type = coupling_type
        num_qubits = self._num_qubits
        
        rgb = None
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

            rgb = hex_map.draw()
            rgb = rgb.convert("RGB")
            



        elif (coupling_type == 'square'):

            row = math.ceil(math.sqrt(num_qubits))
            square_map = updated_map.from_grid(row, row, bidirectional=self._bidirectional)

            graph = square_map.graph
            rgb = square_map.draw()
            rgb = rgb.convert("RGB")


        elif (coupling_type == 'grid'):

            row = math.ceil(np.sqrt(num_qubits))
            col = math.ceil(num_qubits/row)

            grid_map = updated_map.from_grid(row, col, bidirectional=self._bidirectional)
            graph = grid_map.graph

            rgb = grid_map.draw()
            rgb = rgb.convert("RGB")
            

        elif (coupling_type == 'ata'):

            ata_map = updated_map.from_full(num_qubits, bidirectional=self._bidirectional)
            graph = ata_map.graph
            rgb = ata_map.draw()
            rgb = rgb.convert("RGB")

        else:
            raise LookupError("Please use a valid coupling type such as: hexagonal, square, ata or grid")
            
        updated_map.graph.extend_from_edge_list(graph.edge_list())
        self._coupling_map = updated_map
        self._edge_list = list(updated_map.graph.edge_list())
        self._num_qubits = updated_map.size()
            
        self._qubit_properties, self._gate_properties = self._sample_props()
        self._gen_flag()

        return rgb
    
    

    def _generate_couple(self, num_qubits: int, coupling_type: str) -> CouplingMap:
        """Creates coupling scheme of `couping_type` with `num_qubits` (at minimum).
        
        Parameters
        ----------
        num_qubits: int
            Minimum amount of qubits to build up map.
        coupling_type: str
            The coupling scheme to create coupling map.
    #### Returns: CouplingMap
            CouplingMap object.
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


    ###############################################
    #         Scaling methods

    

    def scale_qubit_property(self, property_key:str, scale_factor:float):
        """Scales all qubit properties by `scale_factor`
        
        Parameters
        ----------

        property_key: str
            The qubit property to scale.
        scale_factor: float
            The factor to scale properties by.
        """
        self._qubit_properties[property_key] *= scale_factor

    def scale_gate_property(self, gate_name:str, property_key:str, scale_factor:float):
        """Scales all gate properties by `scale_factor`
        
        Parameters
        ----------
        
        gate_name: str
            Name of gate to scale property.
        property_key: str
            The qubit property to scale.
        scale_factor: float
            The factor to scale properties by.
        """
        selection = self._gate_properties[self._gate_properties.gate==gate_name]
        selection[property_key] *= scale_factor

        self._gate_properties.update(selection)


### Setters:

    def set_flag_gates_property(self, flag: bool, gate_name: str, gate_property: str):
        """Sets flag for ALL gate properties to be static (or values to not be sampled for in backend generation).
        
        Parameters
        ----------

        flag: bool
            True to make property static, False to make property sampled.
        gate_name: str
            The name of the gate to flag.
        gate_property: str
            Gate property to flag.
        """
        try:
            temp_df = self._gate_flag[self._gate_flag.gate == gate_name]
            temp_df[gate_property] = [flag] * len(temp_df[gate_property])
            self._gate_flag.update(temp_df)
        except:
            raise KeyError(f"Gate: {gate_name} with property: {gate_property} not found")


    def set_flag_gate_property(self, flag: str, gate_name: str, gate_property: str, qubits: Union[int, list[int, int]]):
        """Sets flag for gate property to be static (or values to not be sampled for in backend generation).
        
        Parameters
        ----------

        flag: bool
            True to make property static, False to make property sampled.
        gate_name: str
            The name of the gate to flag.
        gate_property: str
            Gate property to flag.
        qubits: int | list(int, int)
            The qubits that the gate is applied to
        """
        
        try:
            qubits = tuple(qubits) if isinstance(qubits, list) else qubits
            temp_df = self._gate_flag.loc[self._gate_flag["gate"] == gate_name]
            
            temp_df = temp_df.loc[temp_df["qubits"] == qubits]

            property_index = temp_df.index[0]

            self._gate_flag[gate_property][property_index] = flag
        except:
            raise KeyError(f"Gate: {gate_name} with qubits: {str(qubits)} and property: {gate_property} not found")
        
    def set_flag_qubits_property(self, flag: bool, qubit_property: str):
        """Sets flag for ALL qubit properties to be static (or values to not be sampled for in backend generation).
        
        Parameters
        ----------

        flag: bool
            True to make property static, False to make property sampled.
        qubit_property: str
            Qubit property to flag. """
        try:
            self._qubit_flag[qubit_property] = [flag] * len(self._qubit_flag[qubit_property])
        except:
            raise KeyError(f"Qubits with property {str(qubit_property)} not found")

        
    def set_flag_qubit_property(self, flag: bool, prop_key: str, qubit_id: int):
        """Sets flag for qubit property to be static (or values to not be sampled for in backend generation).
        
        Parameters
        ----------

        flag: bool
            True to make property static, False to make property sampled.
        qubit_property: str
            Qubit property to flag.
        qubit_id: int
            Integer value that identifies qubit."""
        try:
            self._qubit_flag[prop_key][qubit_id] = flag
        except:
            raise KeyError(f'Qubit {str(qubit_id)} not found with {str(prop_key)}')

    def set_seed(self, seed: float):
        """Function to set seed of samplers"""
        np.random.seed(seed)
        self._seed = seed

    def set_coupling_map(self, coupling_map: CouplingMap, coupling_type: str):
        """Set Coupling Map object of backend 
        - Note: Changing the coupling map could have the affected of changing the
         number of qubits as well as resetting the flags and resampling the property values.
        
        Parameters
        ---------

        coupling_map : CouplingMap
            Coupling Map object to set
        coupling_type : str
            Name of the coupling scheme"""
        self._coupling_map = coupling_map
        self._edge_list = list(coupling_map.graph.edge_list())
        self._coupling_type = coupling_type
        self._num_qubits = coupling_map.size()
        
        self._qubit_properties, self._gate_properties = self._sample_props()
        self._gen_flag()

    def set_bidirectional(self, bidirectional: bool):
        """Sets bidirectional coupling edges to true or false"""
        self._bidirectional = bidirectional

    def set_qubit_property(self, qubit_id: int, qubit_property: str, value: float, set_static_flag : bool = False ):
        """Sets qubit property value of specific qubit
        
        Parameters
        ---------

        qubit_id : int
            Integer value that specifies the qubit.
        qubit_property : str
            Qubit property to be set
        value: float
            Value to be set
        set_static_flag : bool = False
            Default: False, when set true set value will be flagged as static value (to not be resampled upon generation)"""
        
        self._qubit_properties[qubit_property][qubit_id] = value
        if set_static_flag:
            self.set_flag_qubit_property(True, qubit_property, qubit_id)

    def set_multi_qubit_property(self, qubits: list[int], qubit_property: str, values: list[Union[int, float]], set_static_flag :bool = False):
       """Sets qubit property value of specified qubits
        
        Parameters
        ---------

        qubits : list[int]
            List of integer values that specifies the qubits.
        qubit_property : str
            Qubit property to be set
        values: list[float]
            list of values to be set
        set_static_flag : bool = False
            Default: False, when set true set values will be flagged as static values (to not be resampled upon generation)"""
       
       for i in qubits:
           self._qubit_properties[qubit_property][qubits[i]] = values[i]
       if set_static_flag:
            self.set_flag_qubits_property(True, qubit_property, qubits)

    def set_gate_properties(self, gate_name: str, gate_property: str, values: list[Union[int, float]], set_static_flag: bool = False):
        """Sets all gate property values of specific gate type
        
        Parameters
        ---------

        gate_name : str
            Name of gate.
        gate_property : str
            Gate property to be set.
        values: list[float]
            list of values to be set
        set_static_flag : bool = False
            Default: False, when set true set values will be flagged as static values (to not be resampled upon generation)"""
        
        temp_df = self._gate_properties[self._gate_properties.gate == gate_name]
        temp_df[gate_property].values[:] = values
        
        self._gate_properties.update(temp_df)

        if set_static_flag:
            self.set_flag_gates_property(True, gate_name, gate_property)

    def set_gate_property(self, gate_name: str, gate_property: str, qubits: Union[int, tuple[int, int]], value: list[Union[int, float]], set_static_flag: bool = False):
        """Sets gate property values of specific gates under gate type
        
        Parameters
        ---------

        gate_name : str
            Name of gate.
        gate_property : str
            Gate property to be set.
        qubits: int | list[int, int]
            Qubits that the gate is applied to (specification of gates to set)
        values: list[float]
            list of values to be set
        set_static_flag : bool = False
            Default: False, when set true set values will be flagged as static values (to not be resampled upon generation)"""
        
        temp_df = pd.DataFrame(self._gate_properties)
        temp_df = temp_df[temp_df["gate"] == gate_name]
        if gate_name not in self._two_qubit_lut:  # case for any gate other than cx
            temp_df = temp_df[temp_df["qubits"] == qubits]
        else:   # case for cx gate, acts on two qubits
            temp_df = temp_df[temp_df["qubits"] == tuple(qubits)]

        property_index = temp_df.index[0]
        self._gate_properties[gate_property][property_index] = value

        if set_static_flag:
            self.set_flag_gate_property(True, gate_name, gate_property, qubits)

    def set_gate_properties_distribution(self,gate_name: str, gate_property: str, std: float, mean: float, set_static_flag : bool = False):
        """Sets gate property values of specific gates using distribution
        
        Parameters
        ---------

        gate_name : str
            Name of gate.
        gate_property : str
            Gate property to be set.
        std: float
            Standard deviation of distribution.
        mean: float
            Mean of distribution.
        set_static_flag : bool = False
            Default: False, when set true set values will be flagged as static values (to not be resampled upon generation)"""
        
        count = len(self._edge_list) if gate_name in self._two_qubit_lut else self._num_qubits
        vals = self.sample_distribution(std, mean, count)
        self.set_gate_properties(gate_name, gate_property, vals)

        if set_static_flag:
            self.set_flag_gates_property(True, gate_name, gate_property)

    def set_qubits_properties_distribution(self, qubits: list[int],  qubit_property: str, std: float, mean: float, set_static_flag: bool = False):
        """Sets gate property values of specific gates using distribution
        
        Parameters
        ---------

       qubits : list[int]
            List of integer values that specifies the qubits.
        qubit_property : str
            Qubit property to be set
        std: float
            Standard deviation of distribution.
        mean: float
            Mean of distribution.
        set_static_flag : bool = False
            Default: False, when set true set values will be flagged as static values (to not be resampled upon generation)"""
       
        vals = self.sample_distribution(std, mean, self._num_qubits)
        self.set_multi_qubit_property(qubits, qubit_property, vals)

        if set_static_flag:
            for qubit in qubits:
                self.set_flag_qubit_property(True,qubit,  gate_property, qubits)


    def set_max_circuits(self, max_circuits: int):
        """Set the max circuits of backend"""
        self._max_circuits = max_circuits

    def set_dt(self, dt: float):
        """Set dt of backend"""
        self._dt = dt

    # distribution_xxx => [std_xxx, mean_xxx]

    # Methods for basis gates
        
    def add_basis_gate_distribution(self, new_gate : str, distribution_error : list[float, float], distribution_length : list[float, float], set_static_flag: bool = False):
        """Adds new basis gate with gate errors and gate lengths sampled from distribution.
        
        Parameters
        ---------

       new_gate : list[int]
            The name of the new gate to add.
        distribution_error : list[float, float] <-> [std, mean]
            A list with the standard deviation and mean of the distribution for the error rate.
        distribution_length: list[float, float] <-> [std, mean]
            Standard deviation of distribution.
        set_static_flag : bool = False
            Default: False, when set true set values will be flagged as static values (to not be resampled upon generation)"""
        
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

        temp_df['gate_error'][:] = set_static_flag
        temp_df['gate_length'][:] = set_static_flag
        self._basis_gates.append(new_gate) 
        self._gate_flag = pd.concat((self._gate_flag, temp_df), ignore_index= True, sort= False)

    
    def add_basis_gate_numeric(self, gate_name : str, error_vals : list[float], length_vals : list[float], set_static_flag: bool = False):
        """Adds new basis gate with gate errors and gate lengths using provided values.
        
        Parameters
        ---------

       new_gate : list[int]
            The name of the new gate to add.
        error_vals : list[float]
            A list of error values.
        length_vals: list[float]
            A list of the length values.
        set_static_flag : bool = False
            Default: False, when set true set values will be flagged as static values (to not be resampled upon generation)"""
        
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

        temp_df['gate_error'][:] = set_static_flag
        temp_df['gate_length'][:] = set_static_flag 
        self._gate_flag = pd.concat((self._gate_flag, temp_df), ignore_index= True, sort= False)
        self._basis_gates.append(gate_name)

    def remove_basis_gate(self, gate_name : str):
        if gate_name not in self._basis_gates:
            raise LookupError(f"{gate_name} is not in the basis gates.")
        remove = self._gate_properties.loc[self._gate_properties.gate == gate_name].index
        self._gate_properties = self._gate_properties.drop(remove)
        self._gate_properties.index = range(len(self._gate_properties.index))

        self._gate_flag = self._gate_flag.drop(remove)
        self._gate_flag.index = range(len(self._gate_flag.index))
        index = self._basis_gates.index(gate_name)
        self._basis_gates.pop(index)

    def swap_basis_gate(self, old_gate : str, new_gate : str, set_static_flag: bool = False):
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
        replace['gate_error'][:] = set_static_flag
        replace['gate_length'][:] = set_static_flag

        index = self._basis_gates.index(old_gate)
        self._basis_gates.pop(index)
        self._basis_gates.append(new_gate)

    def swap_basis_gate_distribution(self, old_gate : str, new_gate : str, distribution_error : list[float, float], distribution_length : list[float, float], set_static_flag :bool=False):
        if new_gate in self._two_qubit_lut:
            raise LookupError(f"Please use a two qubit basis gate function to modify {new_gate}")
        elif old_gate in self._two_qubit_lut:
            raise LookupError(f"Please use a two qubit basis gate function to modify {old_gate}")
        elif old_gate not in self._basis_gates:
            raise LookupError(f"{old_gate} is not in the basis gates.")
        elif new_gate in self._basis_gates:
            raise AttributeError(f"You already have {new_gate} as a basis gate.")
        self.remove_basis_gate(old_gate)
        self.add_basis_gate_distribution(new_gate, distribution_error, distribution_length, set_static_flag)


    def swap_basis_gate_numeric(self, old_gate : str, new_gate : str, error_vals : list[float], length_vals : list[float], set_static_flag: bool = False):
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
        self.add_basis_gate_numeric(new_gate, error_vals, length_vals, set_static_flag)
    
    def swap_2q_basis_gate(self, old_gate : str, new_gate : str, set_static_flag:bool = False):
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
        replace['gate_error'][:] = set_static_flag
        replace['gate_length'][:] = set_static_flag

        index = self._basis_gates.index(old_gate)
        self._basis_gates.pop(index)
        self._basis_gates.append(new_gate)


    def swap_2q_basis_gate_distribution(self, old_gate : str, new_gate : str, distribution_error : list[float, float], distribution_length : list[float, float], set_static_flag: bool = False):
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
        self._gate_flag.drop(replace.index)
        temp_df = pd.DataFrame({
            "gate" : gate,
            "qubits": qubits,
            "gate_error": gate_error,
            "gate_length": gate_length
        })
        self._gate_properties.update(temp_df)
        temp_df['gate_error'][:] = set_static_flag
        temp_df['gate_length'][:] = set_static_flag
        self._gate_flag.update(temp_df)
        
        index = self._basis_gates.index(old_gate)
        self._basis_gates.pop(index)
        self._basis_gates.append(new_gate)

    
    def swap_2q_basis_gate_numeric(self, old_gate : str, new_gate : str, gate_error : list[float], gate_length : list[float], set_static_flag:bool=False):
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
        self._gate_flag.drop(replace.index)
        temp_df = pd.DataFrame({
            "gate" : gate,
            "qubits": qubits,
            "gate_error": gate_error,
            "gate_length": gate_length
        })
        self._gate_properties.update(temp_df)
        temp_df['gate_error'][:] = set_static_flag
        temp_df['gate_length'][:] = set_static_flag
        self._gate_flag.update(temp_df)

        index = self._basis_gates.index(old_gate)
        self._basis_gates.pop(index)
        self._basis_gates.append(new_gate)

        # stop here for typing

### Flagging for static values

    def _gen_flag(self):
        qubit_flag = self._qubit_properties.copy()
        gate_flag = self._gate_properties.copy()

        for col in qubit_flag.columns:
            qubit_flag[col].values[:] = False
        
        gate_flag['gate_error'].values[:] = False
        gate_flag['gate_length'].values[:] = False


        self._qubit_flag = qubit_flag
        self._gate_flag = gate_flag


    def _apply_flags(self, dfs : pd.DataFrame) -> list[np.ndarray, np.ndarray]:
        qubit_df = dfs[0]
        gate_df = dfs[1]
        for col in qubit_df.columns:
            flag_row = self._qubit_flag[col].values[:]

            qubit_df[col].values[:] = qubit_df[col].values[:] * (1-flag_row) + self._qubit_properties[col].values[:] * flag_row

        for col in ['gate_error', "gate_length"]:
            flag_row = self._gate_flag[col].values

            gate_df[col].values[:] = gate_df[col].values[:] * (1-flag_row) + self._gate_properties[col].values[:] * flag_row
        return [qubit_df, gate_df]


### Samplers
    def _sample_props(self) -> list[np.ndarray, np.ndarray]:

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

    def _sample_gates(self, gate : str, prop_key : str, sample_count : int) -> np.ndarray: 
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
        sample = series.sample(n=sample_count) # random_state for seed?

        sample_output = np.array(sample.abs())

        return sample_output # returns a pd series (or does it have to be a df?)


    def _sample_qubits(self, prop_key : str, sample_count : int) -> np.ndarray: # never mind i need backendspec

        """
      The method signature of this function follows: normal_distribution(self, error_type)
      with the following parameter:
        * error_type (str)

      At the moment, the function covers 4 functions:
      - T1 (loss of coherence)
      - T2 (loss of phase)
      - Readout error
      - Single Gate error

      We use NumPy, Scipy sampling and Seaborn to conduct the sampling. Each of the errors
      have an assoicated normal distribution generating 10000 random samples, and an additional
      uniform random sample, that randomly chooses ONE sample from the overall normal distribution
      to use as the respective error value for the created experimental backend.

      Return type: Pandas Series

      """
        data = self._qubit_properties[prop_key]

        mu = np.mean(data)
        sigma = np.std(data)

        distribution = stats.norm(
            loc=mu,
            scale=sigma
        )

        sample = distribution.rvs(size=10000)

        series = pd.Series(sample)
        sample = series.sample(n=sample_count) # random_state for seed?

        sample_output = np.array(sample.abs())

        return sample_output # returns a pd series (or does it have to be a df?)
    
    def sample_distribution(self, std : int, mean : int, count : int) -> np.ndarray:
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
        # qubits_df = self._qubit_properties
        num_qubits = self._num_qubits
        _target = Target(
                num_qubits = num_qubits,
                dt = self._dt,
                qubit_properties = [
                    IBMQubitProperties( # only grabs t1 t2 and freq because that's what qubitprop has
                        t1 = qubits_df['T1'][i],
                        t2 = qubits_df['T2'][i],
                        frequency = qubits_df['frequency'][i],
                        anharmonicity= qubits_df['anharmonicity'][i],
                    )
                    for i in range(num_qubits)
                ],
            )
        return _target

    def _gen_inst_props(self, props : pd.DataFrame) -> pd.DataFrame:
#  YGate, iSwapGate, U1Gate, UGate, U2Gate, U3Gate, SGate, TGate
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
                'delay': Delay,
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
        test_backend = FakeBackendV2()
        test_backend._coupling_map = self._coupling_map

        qubits_df, gates_df = self._apply_flags(self._sample_props())

        basis_gates = list(self._basis_gates)
        test_backend.qubits_df = qubits_df
        test_backend.gates_df = gates_df
        
        _target = self._gen_target(qubits_df)

        inst_props = self._gen_inst_props(gates_df)

        if "measure" not in self._basis_gates:
                basis_gates.append("measure")

        for gate in basis_gates:
                try:
                    _target.add_instruction(*inst_props[gate])
                except:
                    raise QiskitError(f"{gate} is not a valid basis gate")
        test_backend._target = _target
        test_backend._basis_gates = basis_gates
        return test_backend

