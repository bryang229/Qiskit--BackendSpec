from qiskit.test import QiskitTestCase
from backendspec import BackendSpec
import unittest
from qiskit.compiler.transpiler import CouplingMap
from qiskit.providers.fake_provider import FakeBackendV2



class TestBackendSpec(QiskitTestCase):
    def test_add_basis_gate_distribution(self):
        spec = BackendSpec()
        spec.add_basis_gate_distribution('y', [3,.01], [4,.01])
        spec.set_seed(spec.seed)
        dist1 = spec.sample_distribution(3,.01,2)
        dist2 = spec.sample_distribution(4,.01,2)

        input_dist1 = spec.gate_properties.loc[spec.gate_properties.gate == 'y', 'gate_error'].values
        input_dist2 = spec.gate_properties.loc[spec.gate_properties.gate == 'y', 'gate_length'].values


        self.assertTrue('y' in spec.basis_gates)
        basis_gates = set(spec.gate_properties.gate)
        self.assertTrue('y' in basis_gates)
        for i in range(2):
            self.assertEqual(dist1[i], input_dist1[i])
            self.assertEqual(dist2[i], input_dist2[i])



    def test_add_basis_gate_numeric(self):
        spec = BackendSpec()
        dist1 = [1,2]
        dist2 = [3,4]
        spec.add_basis_gate_numeric('y', dist1, dist2)
        
        input_dist1 = spec.gate_properties.loc[spec.gate_properties.gate == 'y', 'gate_error'].values
        input_dist2 = spec.gate_properties.loc[spec.gate_properties.gate == 'y', 'gate_length'].values

        
        self.assertTrue('y' in spec.basis_gates)
        basis_gates = set(spec.gate_properties.gate)
        self.assertTrue('y' in basis_gates)
        for i in range(2):
            self.assertEqual(dist1[i], input_dist1[i])
            self.assertEqual(dist2[i], input_dist2[i])


    def test_coupling_change(self):
        spec = BackendSpec()

        old_cmap = spec.coupling_map
        spec.coupling_change('square')
        new_cmap = spec.coupling_map

        self.assertNotEqual(old_cmap, new_cmap)


    def test_decrease_qubits(self):
        spec = BackendSpec()
        ## assumes increase qubits works!
        spec.increase_qubits(10,'square') # should become 4x4 
        spec.decrease_qubits(-4, 'grid') # should become 4x3

        c_map = spec.coupling_map
        grid_map = CouplingMap.from_grid(4,3)

        self.assertEqual(c_map, grid_map)
        self.assertEqual(spec.num_qubits, 12)
    
    def test_get_gate_property(self):
        spec = BackendSpec()
        id0_gate_error1 = spec.get_gate_property('id', 'gate_error', 0)
        gate_props = spec.gate_properties
        id0_gate_error2 = gate_props.loc[gate_props.gate=='id', 'gate_error']
        id0_gate_error2 = gate_props['gate_error'][id0_gate_error2.index[0]]
        self.assertEqual(id0_gate_error1 ,id0_gate_error2)
    
    def test_get_qubit_property(self):
        spec = BackendSpec()
        qubit_t1_q1_1 = spec.get_qubit_property(1, 'T1')
        qubit_t1_q1_2 = spec.qubit_properties['T1'][1]
        self.assertEqual(qubit_t1_q1_1 ,qubit_t1_q1_2)

    def test_increase_qubits(self):
        spec = BackendSpec()
        spec.increase_qubits(10, 'square')
        c_map = CouplingMap.from_grid(4,4)
        square_map = spec.coupling_map
        self.assertEqual(c_map, square_map)

    def test_new_backend(self):
        spec = BackendSpec()
        backend = spec.new_backend()
        self.assertTrue(isinstance(backend, FakeBackendV2))

    def test_qubit_selector(self):
        spec = BackendSpec()
        spec.set_multi_qubit_property([0,1], 'T1', [0,1])
        qubit_id = spec.qubit_selector('T1', 1, 2)
        self.assertEqual(qubit_id, [1])

    def test_remove_basis_gate(self):
        spec = BackendSpec()
        spec.remove_basis_gate('id')
        self.assertFalse('id' in spec.basis_gates)

    # TODO:
    # def test_sample_distribution(self):
    #     spec = BackendSpec()
        
    def test_scale_gate_property(self):
        spec = BackendSpec()
        spec.set_gate_property('id', 'gate_error', 0, 1)
        spec.scale_gate_property('id', 'gate_error', 2)
        val = spec.get_gate_property('id', 'gate_error', 0)
        self.assertEqual(val, 2)

    def test_scale_qubit_property(self):
        spec = BackendSpec()
        spec.set_qubit_property(0, 'T1', 1)
        spec.scale_qubit_property('T1', 2)
        val = spec.get_qubit_property(0, 'T1')
        self.assertEqual(val, 2) 

    def test_set_bidirectional(self):
        spec = BackendSpec()
        pre = spec.bidirectional
        spec.set_bidirectional(not pre)
        new = spec.bidirectional
        self.assertNotEqual(pre, new)

    def test_set_coupling_map(self):
        spec = BackendSpec()
        init_num_qubits = spec.num_qubits
        init_c_map = spec.coupling_map

        new_map = CouplingMap.from_hexagonal_lattice(1,1)
        spec.set_coupling_map(new_map, 'hexagonal')

        new_num_qubits = spec.num_qubits
        c_map = spec.coupling_map
        self.assertNotEqual(init_c_map, c_map)
        self.assertEqual(c_map, new_map)
        self.assertNotEqual(init_num_qubits, new_num_qubits)

    def test_set_dt(self):
        spec = BackendSpec()
        init_dt = spec.dt
        spec.set_dt(10)
        new_dt = spec.dt
        self.assertNotEqual(init_dt, new_dt)
        self.assertEqual(10, new_dt)

    def test_set_frozen_gate_property(self):
        spec = BackendSpec()
        spec.set_gate_property('id', 'gate_error', 0, 1)
        spec.set_frozen_gate_property(True, 'id', 'gate_error', 0)
        backend = spec.new_backend()
        id_dict = backend.target['id']
        id0_inst_prop = list(id_dict.values())[0]
        val = id0_inst_prop.error
        self.assertEqual(1, val)

    def test_set_frozen_gates_property(self):
        spec = BackendSpec()
        spec.set_gate_properties('id', 'gate_error', [0,1])
        spec.set_frozen_gates_property(True, 'id', 'gate_error')
        backend = spec.new_backend()
        id_dict = backend.target['id']
        id0_inst_prop = list(map(lambda x: x.error, id_dict.values()))
        for val in [0,1]:
            self.assertEqual(val, id0_inst_prop[val])

    def test_set_frozen_qubit_property(self):
        spec = BackendSpec()
        spec.set_qubit_property(0, 'T1', 2)
        spec.set_frozen_qubit_property(True, 'T1', 0)
        backend = spec.new_backend()
        t1 = backend.target.qubit_properties[0].t1
        self.assertEqual(t1, 2)

    def test_set_frozen_qubits_property(self):
        spec = BackendSpec()
        vals = [2,3]
        spec.set_multi_qubit_property([0,1], 'T1', vals)
        spec.set_frozen_qubits_property(True, 'T1')
        backend = spec.new_backend()
        t1 = [0,0]
        for i in range(2):
            t1 = backend.target.qubit_properties[i].t1
            self.assertEqual(t1, vals[i])

    def test_set_gate_properties(self):
        spec = BackendSpec()
        vals = [10, 20]
        spec.set_gate_properties('cx', 'gate_error', vals)
        set_vals = spec.gate_properties.loc[spec.gate_properties.gate == 'cx', 'gate_error'].values
        for i in range(2):
            self.assertEqual(vals[i], set_vals[i])

    def test_set_gate_properties_distribution(self):
        spec = BackendSpec()
        spec.set_gate_properties_distribution('id', 'gate_error', 1, .01)
        spec.set_seed(spec.seed)
        dist = spec.sample_distribution(1, .01, 2)
        input_dist = spec.gate_properties.loc[spec.gate_properties.gate == 'id', 'gate_error'].values
        for i in range(2):
            self.assertEqual(dist[i], input_dist[i])

    def test_set_gate_property(self):
        spec = BackendSpec()
        spec.set_gate_property('id', 'gate_error', 0, 10)
        val = spec.get_gate_property('id','gate_error', 0)
        self.assertEqual(val, 10)
    def test_set_max_circuits(self):
        spec = BackendSpec()
        pre_max = spec.max_circuits
        spec.set_max_circuits(10)
        new_max = spec.max_circuits
        self.assertNotEqual(pre_max, new_max)
        self.assertEqual(new_max, 10)

    def test_set_multi_qubit_property(self):
        spec = BackendSpec()
        vals = [20,20]
        spec.set_multi_qubit_property([0,1], 'T1',vals )
        q_vals = [spec.get_qubit_property(i, 'T1') for i in range(2)]
        for i in range(2):
            self.assertEqual(vals[i], q_vals[i])

    def test_set_qubit_property(self):
        spec = BackendSpec()
        spec.set_qubit_property(0, 'T1', 10)
        val = spec.get_qubit_property(0, 'T1')
        self.assertEqual(val, 10)

    def test_set_qubits_properties_distribution(self):
        spec = BackendSpec()
        spec.set_qubits_properties_distribution([0,1], 'T1', 2, .01)
        spec.set_seed(spec.seed)
        dist = spec.sample_distribution(2, .01, 2)
        input_dist = [spec.get_qubit_property(i, 'T1') for i in range(2)]
        for i in range(2):
            self.assertEqual(dist[i], input_dist[i])    

    def test_set_seed(self):
        spec = BackendSpec()
        old_seed = spec.seed
        spec.set_seed(10)
        new_seed = spec.seed
        self.assertNotEqual(old_seed, new_seed)
        self.assertEqual(new_seed, 10)

    def test_swap_2q_basis_gate(self):
        spec = BackendSpec()
        spec.swap_2q_basis_gate('cx', 'ecr')
        self.assertTrue('cx' not in spec.basis_gates)
        self.assertTrue('ecr' in spec.basis_gates)
        basis_gates = set(spec.gate_properties.gate)
        self.assertTrue('cx' not in basis_gates)
        self.assertTrue('ecr' in basis_gates)      


    def test_swap_2q_basis_gate_distribution(self):
        spec = BackendSpec()
        spec.swap_2q_basis_gate_distribution('cx', 'ecr', [2, .01], [3, .01])

        spec.set_seed(spec.seed)
        dist1 = spec.sample_distribution(2, .01, 2)
        dist2 = spec.sample_distribution(3, .01, 2)
        input_dist1 = spec.gate_properties.loc[spec.gate_properties.gate == 'ecr', 'gate_error'].values
        input_dist2 = spec.gate_properties.loc[spec.gate_properties.gate == 'ecr', 'gate_length'].values

        self.assertTrue('cx' not in spec.basis_gates)
        self.assertTrue('ecr' in spec.basis_gates)
        basis_gates = set(spec.gate_properties.gate)
        self.assertTrue('cx' not in basis_gates)
        self.assertTrue('ecr' in basis_gates)      

        for i in range(2):
            self.assertEqual(dist1[i], input_dist1[i])
            self.assertEqual(dist2[i], input_dist2[i])

    def test_swap_2q_basis_gate_numeric(self):
        spec = BackendSpec()

        dist1 = [0,1]
        dist2 = [2,3]
        spec.swap_2q_basis_gate_numeric('cx', 'ecr', dist1, dist2)

        spec.set_seed(spec.seed)
 
        input_dist1 = spec.gate_properties.loc[spec.gate_properties.gate == 'ecr', 'gate_error'].values
        input_dist2 = spec.gate_properties.loc[spec.gate_properties.gate == 'ecr', 'gate_length'].values

        self.assertTrue('cx' not in spec.basis_gates)
        self.assertTrue('ecr' in spec.basis_gates)
        basis_gates = set(spec.gate_properties.gate)
        self.assertTrue('cx' not in basis_gates)
        self.assertTrue('ecr' in basis_gates)      

        for i in range(2):
            self.assertEqual(dist1[i], input_dist1[i])
            self.assertEqual(dist2[i], input_dist2[i])

    def test_swap_basis_gate(self):
        spec = BackendSpec()
        spec.swap_basis_gate('id', 'y')
        self.assertTrue('id' not in spec.basis_gates)
        self.assertTrue('y' in spec.basis_gates)
        basis_gates = set(spec.gate_properties.gate)
        self.assertTrue('id' not in basis_gates)
        self.assertTrue('y' in basis_gates)      
        
    def test_swap_basis_gate_distribution(self):
        spec = BackendSpec()
        spec.swap_basis_gate_distribution('id', 'y', [2,.01], [3,.01])
        spec.set_seed(spec.seed)
        dist1 = spec.sample_distribution(2, .01, 2)
        dist2 = spec.sample_distribution(3, .01, 2)
        input_dist1 = spec.gate_properties.loc[spec.gate_properties.gate == 'y', 'gate_error'].values
        input_dist2 = spec.gate_properties.loc[spec.gate_properties.gate == 'y', 'gate_length'].values

        self.assertTrue('id' not in spec.basis_gates)
        self.assertTrue('y' in spec.basis_gates)
        basis_gates = set(spec.gate_properties.gate)
        self.assertTrue('id' not in basis_gates)
        self.assertTrue('y' in basis_gates)      

        for i in range(2):
            self.assertEqual(dist1[i], input_dist1[i])
            self.assertEqual(dist2[i], input_dist2[i])

    def test_swap_basis_gate_numeric(self):
        spec = BackendSpec()
    
        dist1 = [5,6]
        dist2 = [9,10]

        spec.swap_basis_gate_numeric('id', 'y', dist1, dist2)


        input_dist1 = spec.gate_properties.loc[spec.gate_properties.gate == 'y', 'gate_error'].values
        input_dist2 = spec.gate_properties.loc[spec.gate_properties.gate == 'y', 'gate_length'].values

        self.assertTrue('id' not in spec.basis_gates)
        self.assertTrue('y' in spec.basis_gates)
        basis_gates = set(spec.gate_properties.gate)
        self.assertTrue('id' not in basis_gates)
        self.assertTrue('y' in basis_gates)      

        for i in range(2):
            self.assertEqual(dist1[i], input_dist1[i])
            self.assertEqual(dist2[i], input_dist2[i])



if __name__ == '__main__':
    unittest.main()