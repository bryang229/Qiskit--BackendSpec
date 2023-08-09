import numpy as np
import math
import pandas as pd
from scipy import stats
import warnings
from typing import Optional, List, Tuple
from qiskit.transpiler import CouplingMap
from qiskit.providers.basicaer import BasicAer
from qiskit.providers.ibmq import IBMQBackend
from qiskit.transpiler import Target, InstructionProperties, QubitProperties
from qiskit.providers.fake_provider import FakeBackendV2
from qiskit.providers.backend import BackendV2
from qiskit.providers.options import Options
from qiskit.exceptions import QiskitError
from qiskit_ibm_provider.ibm_qubit_properties import IBMQubitProperties
from qiskit.circuit.library import XGate, RZGate, SXGate, CXGate, ECRGate, IGate, CZGate, ECRGate, YGate, iSwapGate, U1Gate, UGate, U2Gate, U3Gate, SGate, TGate, SwapGate
from qiskit.circuit import Measure, Parameter, Delay, Reset
from qiskit.compiler.transpiler import target_to_backend_properties, CouplingMap


# TODO: Make BackendSpec.new_backend() work with parameter gates


pd.options.mode.chained_assignment = None  # default='warn'
#TODO:
#   - Generalize scrapper to go from parent input to parent(s)
#   - Make it so that numpy arrays are used for lists

class BackendSpec:
    def __init__(self, parent=None):
        
        self.seed = np.random.randint(10000000,300000000)
        np.random.seed(self.seed)
        self.two_qubit_lut = ['cx', 'ecr', 'cz', 'swap', 'iswap'] #TODO: Add more cases
        
        if parent == None:          ## default None parameter for when user doesn't want to base it on an existing backend
            self._load_base()
            return 
        
        self.false = False
        self.parent = parent

        if isinstance(parent, IBMQBackend):
            self._load_IBMQ(parent)
            self.coupling_type = 'hexagonal'

        elif "fake" in parent.name.lower():
            self.false = True
            
        if isinstance(parent, BackendV2):
            self._load_IBM(parent)
            self.coupling_type = 'hexagonal'
            
        self._load_data()
        self._load_edges(self.coupling_map.graph)
        self._tuple_remover()

        self._gen_flag_df()
        del self.parent
        del self.properties
        del self.false


    def _load_base(self):
        self.basis_gates = ['x', 'sx', 'cx', 'rz', 'id', 'reset']
        self.coupling_type = 'hexagonal' 
        self.coupling_list = [(0,1), (1,0)]
        self.coupling_map = CouplingMap()
        self.coupling_map.graph.extend_from_edge_list([(0,1), (1,0)])
        self.num_qubits = 2
        


        self.qubit_props_df = pd.DataFrame({"T1": [0,0],
                                            "T2": [0,0],
                                            "frequency": [0,0],
                                            "anharmonicity": [0,0],
                                            "readout_error": [0,0],
                                            "prob_meas0_prep1": [0,0],
                                            "prob_meas1_prep0": [0,0],
                                            "readout_length": [0,0]
                                            })
        
        self.gate_props_df = pd.DataFrame({"gate": ['x', 'sx', 'rz', 'id', 'x', 'sx', 'rz', 'id','cx', 'cx', 'reset', 'reset'],
                                        "qubits": [0, 0, 0, 0,  1, 1, 1 ,1, (0,1), (1,0), 0, 1],
                                        "gate_error": [0] *  12,
                                        "gate_length": [0] * 12
                                        })
        
        self.parent_type = 'user-made'
        self.max_circuits = 100
        self.dt = 0
        self.dtm = 0
        self.bidirectional = True
        self._gen_flag_df()

        return

#TODO: Add calibartion
    def _load_IBMQ(self,parent):
        config = parent.configuration()
        self.basis_gates = config.basis_gates
        self.num_qubits = config.num_qubits
        self.dt = config.dt
        self.dtm = config.dtm
        self.parent_type = 'ibmq'

        # Since IBMQBackend does not have a CouplingMap obj we create one (based off the edges given)
        coupling_map = config.coupling_map
        self.coupling_list = [tuple(pair) for pair in coupling_map]
        self.coupling_map = CouplingMap()
        self.coupling_map.graph.extend_from_edge_list(self.coupling_list)
        
        self.max_circuits = config.max_experiments
        self.properties = parent.properties()


    def _load_IBM(self,parent):
        if self.false:
          self.basis_gates = ['id', 'rz', 'sx', 'x', 'cx', 'reset'] 
          self.properties =  target_to_backend_properties(parent.target)
        
        else:
          self.basis_gates = parent.basis_gates
          if 'reset' not in parent.basis_gates:
            parent.basis_gates.append('reset')
          self.properties = parent.properties()
          self.dtm = parent.dtm


        self.num_qubits = parent.num_qubits
        self.dt = parent.dt
        self.coupling_map = parent.coupling_map
        self.coupling_list = list(parent.coupling_map.graph.edge_list())
        self.max_circuits = parent.max_circuits
        self.parent_type = 'ibm'
         


    def _load_data(self):

        qubit_props = self.properties._qubits #loading qubit props
        
        qubit_props_df = pd.DataFrame(data=qubit_props)
        qubit_props_df = qubit_props_df.transpose() #transposing so proper keys are on columns

        self.qubit_props_df = qubit_props_df # setting attribute


        gate_props = self.properties._gates

        gate_prop_holder = pd.DataFrame(columns=["gate", "qubits","gate_error", "gate_length"])

        for props in gate_props:
            gate = [props] * len(gate_props[props])
            qubits = list(gate_props[props].keys())

            if props not in self.two_qubit_lut:
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


        self.gate_props_df = gate_prop_holder



    def _load_edges(self, graph):
        in_edges =  np.empty(self.num_qubits)
        out_edges = np.empty(self.num_qubits)

        for i in range(self.num_qubits):
            in_edges[i]  = len(graph.in_edges(i))
            out_edges[i] = len(graph.out_edges(i))
        self.bidirectional = not False in (in_edges == out_edges)
        self.qubit_props_df['total_edges'] = in_edges + out_edges
    def _tuple_remover(self):
        for key in self.qubit_props_df:
            self.qubit_props_df[key] = self.qubit_props_df[key].apply(lambda x: x[0] if isinstance(x, tuple) else x)
        for key in self.gate_props_df:
            if key == 'qubits':
                continue
            self.gate_props_df[key] = self.gate_props_df[key].apply(lambda x: x[0] if isinstance(x, tuple) else x)



### Setters:

    def set_coupling_map(self, coupling_map, coupling_type):
        self.coupling_map = coupling_map
        self.coupling_list = list(coupling_map.graph.edge_list())
        self.coupling_type = coupling_type
        self.num_qubits = coupling_map.size()
        
        self.qubit_props_df, self.gate_props_df = self._sample_props()
        self._gen_flag_df()



    def set_qubit_property(self, qubit, qubit_property, value):
        self.qubit_props_df[qubit_property][qubit] = value

    def set_qubit_properties(self, qubits, qubit_property, values):
       for i in qubits:
           self.qubit_props_df[qubit_property][i] = values[i]

    def set_gate_properties(self, gate_name, gate_property, values):
        temp_df = self.gate_props_df[self.gate_props_df.gate == gate_name]
        temp_df[gate_property].values[:] = values
        
        self.gate_props_df.update(temp_df)


    def set_gate_property(self, gate_name, gate_property, qubits, value):
        temp_df = pd.DataFrame(self.gate_props_df)
        temp_df = temp_df[temp_df["gate"] == gate_name]
        if gate_name not in self.two_qubit_lut:  # case for any gate other than cx
            temp_df = temp_df[temp_df["qubits"] == qubits]
        else:   # case for cx gate, acts on two qubits
            temp_df = temp_df[temp_df["qubits"] == tuple(qubits)]

        property_index = temp_df.index[0]
        self.gate_props_df[gate_property][property_index] = value

    def set_gate_properties_dist(self,gate_name, prop_key, std, mean):
        count = len(self.coupling_list) if gate_name in self.two_qubit_lut else self.num_qubits
        vals = self.sample_dist(std, mean, count)
        self.set_gate_properties(gate_name, prop_key, vals)

    def set_qubits_properties_dist(self,qubits,  prop_key, std, mean):
        vals = self.sample_dist(std, mean, self.num_qubits)
        self.set_qubit_properties(qubits, prop_key, vals)



    def set_max_circuits(self, max_circuits):
        self.max_circuits = max_circuits

    def set_dt(self, dt):
        self.dt = dt

    

    def sample_dist(self, std, mean, count):
        distribution = stats.norm(
            loc=mean,
            scale=std
        )
        sample = distribution.rvs(size=10000)

        series = pd.Series(sample)
        sample = series.sample(n=count) # random_state for seed?
        sample_output = np.array(sample.abs())

        return sample_output
### Gettter

    def get_num_qubits(self):
      return self.num_qubits

    def get_qubit_property(self, qubit, qubit_property):
      return self.qubit_props_df[qubit_property][qubit]


    def get_gate_property(self, gate_name, qubits, gate_property):
      temp_df = pd.DataFrame(self.gate_props_df)
      temp_df = temp_df[temp_df["gate"] == gate_name]
      if gate_name in self.two_qubit_lut:
          temp_df = temp_df[temp_df["qubits"] == qubits]
      else:
          temp_df = temp_df[temp_df["qubits"] == tuple(qubits)]

      property_index = temp_df.index[0]

      return self.gate_props_df[gate_property][property_index]

    def qubit_selector(self, property, lower_bound, upper_bound):
        qubit_indices = []
        for i in range(len(self.qubit_props_df.index)):
            if self.qubit_props_df[property][i] >= lower_bound and self.qubit_props_df[property][i] <= upper_bound:
                qubit_indices.append(i)

        return qubit_indices


    def from_backend(self, parent):
        self.__init__(parent)
        
 # Modifiers

    def increase_qubits(self, increase_amount, coupling_type):
        
        if increase_amount < 0:
            raise ValueError("Please provide a number greater than zero. To decrease see self.decrease_qubits(decrease_amount, coupling_type)")
        self.num_qubits += increase_amount
        
        return self.coupling_change(coupling_type)
        


    def decrease_qubits(self, decrease_amount, coupling_type):
        if decrease_amount > 0:
            raise ValueError("Please provide a number less than zero. To increase see self.increase_qubits(increase_amount, coupling_type)")
        self.num_qubits += decrease_amount
        return self.coupling_change(coupling_type)

    def coupling_change(self, coupling_type):
        self.coupling_type = coupling_type
        num_qubits = self.num_qubits
        
        rgb = None
        updated_map = CouplingMap()
        graph = None

        if (coupling_type == 'hexagonal'):

            m = num_qubits
            col = 1

            hex_map = updated_map.from_hexagonal_lattice(1,col, bidirectional=self.bidirectional)
            

            while hex_map.size() < m:
                col += 1
                hex_map = updated_map.from_hexagonal_lattice(1,col, bidirectional=self.bidirectional) # increase cols if the thing starts to infinite loop

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
            square_map = updated_map.from_grid(row, row, bidirectional=self.bidirectional)

            graph = square_map.graph
            rgb = square_map.draw()
            rgb = rgb.convert("RGB")


        elif (coupling_type == 'grid'):

            row = math.ceil(np.sqrt(num_qubits))
            col = math.ceil(num_qubits/row)

            grid_map = updated_map.from_grid(row, col, bidirectional=False)
            graph = grid_map.graph

            rgb = grid_map.draw()
            rgb = rgb.convert("RGB")
            

        elif (coupling_type == 'ata'):

            ata_map = updated_map.from_full(num_qubits, bidirectional=False)
            graph = ata_map.graph
            rgb = ata_map.draw()
            rgb = rgb.convert("RGB")

        else:
            raise LookupError("Please use a valid coupling type such as: hexagonal, square, ata or grid")
            
        updated_map.graph.extend_from_edge_list(graph.edge_list())
        self.coupling_map = updated_map
        self.coupling_list = list(updated_map.graph.edge_list())
        self.num_qubits = updated_map.size()
            
        self.qubit_props_df, self.gate_props_df = self._sample_props()
        self._gen_flag_df()

        return rgb
    
    

    def _generate_couple(self, num_qubits, coupling_type):
        updated_map = CouplingMap()
        graph = None

        if (coupling_type == 'hexagonal'):
            m = num_qubits
            col = 1
            hex_map = updated_map.from_hexagonal_lattice(1,col, bidirectional=self.bidirectional)

            while hex_map.size() < m:
                col += 1
                hex_map = updated_map.from_hexagonal_lattice(1,col, bidirectional=self.bidirectional) # increase cols if the thing starts to infinite loop

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
            square_map = updated_map.from_grid(row, row, bidirectional=self.bidirectional)

            graph = square_map.graph


        elif (coupling_type == 'grid'):

            row = math.ceil(np.sqrt(num_qubits))
            col = math.ceil(num_qubits/row)

            grid_map = updated_map.from_grid(row, col, bidirectional=False)
            graph = grid_map.graph

    
            
        elif (coupling_type == 'ata'):
            ata_map = updated_map.from_full(num_qubits, bidirectional=False)
            updated_map = ata_map
        
        else:
            raise LookupError("Please use a valid coupling type such as: hexagonal, square, ata or grid")
            

        updated_map.graph.extend_from_edge_list(graph.edge_list())

        return updated_map


    ###############################################
    #         Scaling methods

    

    def scale_qubit_property(self, property_key, scale_factor):
        self.qubit_props_df[property_key] *= scale_factor

    # def scale_qubit_property_sel(self, property_key, scale_factor, expression):
    #     self.qubit_props_df[property_key][expression(self.qubit_props_df[property_key])] *= scale_factor

    # def scale_gate_property_sel(self, gate_name, property_key, scale_factor, expression):
    #     selection = self.gate_props_df[self.gate_props_df.gate == gate_name]
    #     selection[expression(selection[property_key])] *= scale_factor

    #     self.gate_props_df.update(selection)

    def scale_gate_property(self, gate_name, property_key, scale_factor):
        selection = self.gate_props_df[self.gate_props_df.gate==gate_name]
        selection[property_key] *= scale_factor

        self.gate_props_df.update(selection)



### Flagging for static values

    def _gen_flag_df(self):
        qubit_flag = self.qubit_props_df.copy()
        gate_flag = self.gate_props_df.copy()

        for col in qubit_flag.columns:
            qubit_flag[col].values[:] = False
        
        gate_flag['gate_error'].values[:] = False
        gate_flag['gate_length'].values[:] = False


        self.qubit_flag = qubit_flag
        self.gate_flag = gate_flag


    def set_flag_gates_property(self, flag, gate_name, prop_key):
        try:
            temp_df = self.gate_flag[self.gate_flag.gate == gate_name]
            temp_df[prop_key] = [flag] * len(temp_df[prop_key])
            self.gate_flag.update(temp_df)
            return True
        except:
            raise KeyError(f"Gate: {gate_name} with property: {prop_key} not found")


    def set_flag_gate_property(self, flag, gate_name, prop_key, qubits):
        try:
            qubits = tuple(qubits) if isinstance(qubits, list) else qubits
            temp_df = self.gate_flag.loc[self.gate_flag["gate"] == gate_name]
            
            temp_df = temp_df.loc[temp_df["qubits"] == qubits]

            property_index = temp_df.index[0]

            self.gate_flag[prop_key][property_index] = flag
            return True
        except:
            raise KeyError(f"Gate: {gate_name} with qubits: {str(qubits)} and property: {prop_key} not found")
        
    def set_flag_qubits_property(self, flag, prop_key):
        try:
            self.qubit_flag[prop_key] = [flag] * len(self.qubit_flag[prop_key])
            return True
        except:
            raise KeyError(f"Qubits with property {str(prop_key)} not found")

        
    def set_flag_qubit_property(self, flag, prop_key, qubit_id):
        try:
            self.qubit_flag[prop_key][qubit_id] = flag
            return True
        except:
            raise KeyError(f'Qubit {str(qubit_id)} not found with {str(prop_key)}')


    # dist_xxx => [std_xxx, mean_xxx]
        
    def add_basis_gate_dist(self, new_gate, dist_error, dist_length):
        if new_gate in self.two_qubit_lut:
            raise LookupError(f"Please use a two qubit basis gate function to modify {new_gate}")
        elif new_gate in self.basis_gates:
            raise AttributeError(f"You already have {new_gate} as a basis gate.")
        gates = [new_gate] * self.num_qubits
        error_vals = self.sample_dist(dist_error[0], dist_error[1], self.num_qubits)
        length_vals = self.sample_dist(dist_length[0], dist_length[1], self.num_qubits)

        temp_df = pd.DataFrame({"gate": gates,
                                "qubits": range(self.num_qubits),
                                "gate_error": error_vals,
                                "gate_length": length_vals       
        })

        self.gate_props_df = pd.concat((self.gate_props_df, temp_df), ignore_index= True, sort= False)

        temp_df['gate_error'][:] = False
        temp_df['gate_length'][:] = False
        self.basis_gates.append(new_gate) 
        self.gate_flag = pd.concat((self.gate_flag, temp_df), ignore_index= True, sort= False)
    
    def add_basis_gate_numeric(self, gate_name, error_vals, length_vals):
        if gate_name in self.two_qubit_lut:
            raise LookupError(f"Please use a two qubit basis gate function to modify {gate_name}")
        elif len(length_vals) != self.num_qubits != len(error_vals):
            raise AttributeError("error_val AND length_vals should be the same size as many qubits as your backend has. Please use self.increase_qubits(qubit) to change the size of your backend.")
        elif gate_name in self.basis_gates:
            raise AttributeError(f"You already have {gate_name} as a basis gate.")

        gates = [gate_name] * self.num_qubits
        qubits = range(self.num_qubits)

        temp_df = pd.DataFrame({"gate": gates,
                                "qubits": qubits,
                                "gate_error": error_vals,
                                "gate_length": length_vals       
        })

        self.gate_props_df = pd.concat((self.gate_props_df, temp_df), ignore_index= True, sort= False)

        temp_df['gate_error'][:] = False
        temp_df['gate_length'][:] = False 
        self.gate_flag = pd.concat((self.gate_flag, temp_df), ignore_index= True, sort= False)
        self.basis_gates.append(gate_name)

    def remove_basis_gate(self, gate_name):
        if gate_name not in self.basis_gates:
            raise LookupError(f"{gate_name} is not in the basis gates.")
        remove = self.gate_props_df.loc[self.gate_props_df.gate == gate_name].index
        self.gate_props_df = self.gate_props_df.drop(remove)
        self.gate_props_df.index = range(len(self.gate_props_df.index))

        self.gate_flag = self.gate_flag.drop(remove)
        self.gate_flag.index = range(len(self.gate_flag.index))
        index = self.basis_gates.index(gate_name)
        self.basis_gates.pop(index)

    def swap_basis_gate(self, old_gate, new_gate):
        if new_gate in self.two_qubit_lut:
            raise LookupError(f"Please use a two qubit basis gate function to modify {new_gate}")
        elif old_gate in self.two_qubit_lut:
            raise LookupError(f"Please use a two qubit basis gate function to modify {old_gate}")
        elif old_gate not in self.basis_gates:
            raise LookupError(f"{old_gate} is not in the basis gates.")
        elif new_gate in self.basis_gates:
            raise AttributeError(f"You already have {new_gate} as a basis gate.")
        
        replace = self.gate_props_df.loc[self.gate_props_df.gate==old_gate]
        replace['gate'].values[:] = [new_gate] * len(replace)
        self.gate_props_df.update(replace)
        replace['gate_error'][:] = False
        replace['gate_length'][:] = False

        index = self.basis_gates.index(old_gate)
        self.basis_gates.pop(index)
        self.basis_gates.append(new_gate)

    def swap_basis_gate_dist(self, old_gate, new_gate, dist_error, dist_length):
        if new_gate in self.two_qubit_lut:
            raise LookupError(f"Please use a two qubit basis gate function to modify {new_gate}")
        elif old_gate in self.two_qubit_lut:
            raise LookupError(f"Please use a two qubit basis gate function to modify {old_gate}")
        elif old_gate not in self.basis_gates:
            raise LookupError(f"{old_gate} is not in the basis gates.")
        elif new_gate in self.basis_gates:
            raise AttributeError(f"You already have {new_gate} as a basis gate.")
        self.remove_basis_gate(old_gate)
        self.add_basis_gate_dist(new_gate, dist_error, dist_length)


    def swap_basis_gate_numeric(self, old_gate, new_gate, error_vals, length_vals):
        if new_gate in self.two_qubit_lut:
            raise LookupError(f"Please use a two qubit basis gate function to modify {new_gate}")
        elif old_gate in self.two_qubit_lut:
            raise LookupError(f"Please use a two qubit basis gate function to modify {old_gate}")
        elif old_gate not in self.basis_gates:
            raise LookupError(f"{old_gate} is not in the basis gates.")
        elif new_gate in self.basis_gates:
            raise AttributeError(f"You already have {new_gate} as a basis gate.")
        elif len(length_vals) != self.num_qubits != len(error_vals):
            raise AttributeError("error_val AND length_vals should be the same size as many qubits as your backend has. Please use self.increase_qubits(qubit) to change the size of your backend.")
        self.remove_basis_gate(old_gate)
        self.add_basis_gate_numeric(new_gate, error_vals, length_vals)
    
    def swap_2q_basis_gate(self, old_gate, new_gate):
        if old_gate not in self.basis_gates:
            raise LookupError(f"{old_gate} is not in the basis gates.")
        elif old_gate not in self.two_qubit_lut:
            raise LookupError(f"{old_gate} is not a two qubit gate.")
        elif new_gate in self.basis_gates:
            raise AttributeError(f"You already have {new_gate} as a basis gate.")
        elif new_gate not in self.two_qubit_lut:
            raise LookupError(f"{new_gate} is not a two qubit gate.")

        replace = self.gate_props_df.loc[self.gate_props_df.gate==old_gate]
        replace['gate'].values[:] = [new_gate] * len(replace)
        self.gate_props_df.update(replace)
        replace['gate_error'][:] = False
        replace['gate_length'][:] = False

        index = self.basis_gates.index(old_gate)
        self.basis_gates.pop(index)
        self.basis_gates.append(new_gate)


    def swap_2q_basis_gate_dist(self, old_gate, new_gate, dist_error, dist_length):
        if old_gate not in self.basis_gates:
            raise LookupError(f"{old_gate} is not in the basis gates.")
        elif old_gate not in self.two_qubit_lut:
            raise LookupError(f"{old_gate} is not a two qubit gate.")
        elif new_gate in self.basis_gates:
            raise AttributeError(f"You already have {new_gate} as a basis gate.")
        elif new_gate not in self.two_qubit_lut:
            raise LookupError(f"{new_gate} is not a two qubit gate.")

        replace = self.gate_props_df.loc[self.gate_props_df.gate==old_gate]
        qubits = replace.qubits

        count = len(replace)
        gate = [new_gate] * count

        gate_error = self.sample_dist(dist_error[0], dist_error[1], count)
        gate_length = self.sample_dist(dist_length[0], dist_length[1], count)

        self.gate_props_df.drop(replace.index)
        self.gate_flag.drop(replace.index)
        temp_df = pd.DataFrame({
            "gate" : gate,
            "qubits": qubits,
            "gate_error": gate_error,
            "gate_length": gate_length
        })
        self.gate_props_df.update(temp_df)
        temp_df['gate_error'][:] = False
        temp_df['gate_length'][:] = False
        self.gate_flag.update(temp_df)
        
        index = self.basis_gates.index(old_gate)
        self.basis_gates.pop(index)
        self.basis_gates.append(new_gate)

    
    def swap_2q_basis_gate_numeric(self, old_gate, new_gate, gate_error, gate_length):
        if old_gate not in self.basis_gates:
            raise LookupError(f"{old_gate} is not in the basis gates.")
        elif old_gate not in self.two_qubit_lut:
            raise LookupError(f"{old_gate} is not a two qubit gate.")
        elif new_gate in self.basis_gates:
            raise AttributeError(f"You already have {new_gate} as a basis gate.")
        elif new_gate not in self.two_qubit_lut:
            raise LookupError(f"{new_gate} is not a two qubit gate.")

        replace = self.gate_props_df.loc[self.gate_props_df.gate==old_gate]
        qubits = replace.qubits

        count = len(replace)
        gate = [new_gate] * count

        self.gate_props_df.drop(replace.index)
        self.gate_flag.drop(replace.index)
        temp_df = pd.DataFrame({
            "gate" : gate,
            "qubits": qubits,
            "gate_error": gate_error,
            "gate_length": gate_length
        })
        self.gate_props_df.update(temp_df)
        temp_df['gate_error'][:] = False
        temp_df['gate_length'][:] = False
        self.gate_flag.update(temp_df)

        index = self.basis_gates.index(old_gate)
        self.basis_gates.pop(index)
        self.basis_gates.append(new_gate)


    def _apply_flags(self, dfs):
        qubit_df = dfs[0]
        gate_df = dfs[1]
        for col in qubit_df.columns:
            flag_row = self.qubit_flag[col].values[:]

            qubit_df[col].values[:] = qubit_df[col].values[:] * (1-flag_row) + self.qubit_props_df[col].values[:] * flag_row

        for col in ['gate_error', "gate_length"]:
            flag_row = self.gate_flag[col].values

            gate_df[col].values[:] = gate_df[col].values[:] * (1-flag_row) + self.gate_props_df[col].values[:] * flag_row
        return [qubit_df, gate_df]


### Seeding
    def set_seed(self, seed):
        np.random.seed(seed)
        self.seed = seed

### Samplers
    def _sample_props(self):

        qubit_df = pd.DataFrame()
        i = 0
        for prop in self.qubit_props_df.columns:
            qubit_df.insert(i, prop, self._sample_qubits(prop, self.num_qubits), True)

        gate_df = pd.DataFrame(columns=["gate", "qubits","gate_error", "gate_length"])

        for gate in self.basis_gates:

            count = len(self.coupling_list) if gate in self.two_qubit_lut else self.num_qubits

            gate_list = [gate] * count
            qubits = list(range(count)) if gate not in self.two_qubit_lut else self.coupling_list

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

    def _sample_gates(self, gate, prop_key, sample_count): 
        mini_df = self.gate_props_df.loc[self.gate_props_df['gate'] == gate]

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


    def _sample_qubits(self, prop_key, sample_count): # never mind i need backendspec

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
        data = self.qubit_props_df[prop_key]

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
    
### New backend generation code

    def _gen_target(self, qubits_df):
        # qubits_df = self.qubit_props_df
        num_qubits = self.num_qubits
        _target = Target(
                num_qubits = num_qubits,
                dt = self.dt,
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

    def _gen_inst_props(self, props):
#  YGate, iSwapGate, U1Gate, UGate, U2Gate, U3Gate, SGate, TGate
        gates_lut = {
                'x': XGate,
                'y': YGate,
                's': SGate,
                't': TGate,
                'u' : UGate,
                'u1': U1Gate,
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
        
        
    

        props = self.qubit_props_df
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


    def new_backend(self):
        test_backend = FakeBackendV2()
        test_backend._coupling_map = self.coupling_map

        qubits_df, gates_df = self._apply_flags(self._sample_props())

        basis_gates = list(self.basis_gates)
        test_backend.qubits_df = qubits_df
        test_backend.gates_df = gates_df
        
        _target = self._gen_target(qubits_df)

        inst_props = self._gen_inst_props(gates_df)

        if "measure" not in self.basis_gates:
                basis_gates.append("measure")

        for gate in basis_gates:
                try:
                    _target.add_instruction(*inst_props[gate])
                except:
                    raise QiskitError(f"{gate} is not a valid basis gate")
        test_backend._target = _target
        test_backend._basis_gates = basis_gates
        return test_backend
