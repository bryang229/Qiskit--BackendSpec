# Framework for Generating Experimental Backends (BackendSpec)
### Bryan Bonilla Garay, Matthew Figueroa, Vea Iyer

________________

There are many aspects of IBM Quantum backends that require meticulous testing with the transpiler. The purpose of BackendSpec is to provide a framework for generating highly customized backends that can be used for more diverse testing methods. This framework would allow for exploring new hardware architectures with different coupling schemes, qubit types, qubit specific basis gates and  noise aware transpilation.

Below are a few tutorials explaining the various features of BackendSpec, including initialization and different use cases.

## Initialization

There are two ways to initialize a BackendSpec object:

1. A BackendSpec based off of an existing IBM/IBMQ Backend:

    ```
    IBMProvider.save_account('your_API_token')
    provider = IBMProvider()    
    parent_backend = provider.get_backend('ibm_backend')
    backend_spec = BackendSpec(parent_backend)
    ```

    This will return a BackendSpec object consisting of the qubit and gate properties of the specified IBM backend.

2. An empty BackendSpec with specifications to be added:
    ```
    backend_spec = BackendSpec()
    ```
    In this case, the BackendSpec object will be initialized with 2 qubits, and the rest of the qubit and gate properties will be empty. These properties can then be populated with the built-in modifier functions, that are assisted by sampling functions that sample values from a distribution corresponding to the attributes of QubitProperties and GateProperties.

    After the initialization, the BackendSpec undergoes three separate processes to create a backend to be passed into the transpiler. 

    ### Loaders

    The loaders are used to retrieve data to create the BackendSpec objects. There are four main types of loaders:
    
    - IBM: current IBM backend version
    - IBMQ: former IBM backend version (pre-migration)
    - `FakeBackendV2`: Object for FakeBackends
    - Base: empty `BackendSpec` object with no specifications

    The loaders are called in their specific initialization functions, and accordingly load the QubitProperties and GateProperties as DataFrames. 

    ### Modifiers

    `QubitProperties` and `GateProperties` objects contain the majority of information regarding any generated backend. 
    
    `QubitProperties` consist of various properties associated with qubits, such as:
    - T1 rates (decoherence)
        - Property key: `T1`
    - T2 rates (loss of phase)
        - Property key: `T2`
    - Frequency
        - Property key: `frequency`
    - Anharmonicity
        - Property key: `anharmoncity`
    - Readout errors
        - Property key: `readout_error`
    - Readout lengths
        - Property key: `readout_length`

    `GateProperties` consist of properties associated with single and multi-qubit gates, such as:

    - Gate errors
        - Property key: `gate_error`
    - Gate lengths
        - Property key: `gate_length`
    These properties can be access by providing the:
    - Qubits (the qubit(s) that the gate acts on)
        - Property key: `qubits`
    - Gate name:
        - Property key: `gate_name`

    The modifer functions allow for these properties to be selected, frozen, scaled and even replaced.

    ### Samplers

    Since there are a variety of qubit and gate properties, they can be modelled using a normal distribution, which generate about 10000 samples per attribute. Then, the corresponding number of values for the `QubitProperties` and `GateProperties` are chosen using a uniform random distribution. The sampled values are fed into the respective DataFrames, and are used when the modifiers are called. This is especially useful when more qubits are introduced in the backend, or if the backend is being newly generated. 

    The sampler functions help generate values for  `QubitProperties` and `GateProperties` attributes, either with respect to a probability distribution with a provided mean and standard deviation, or directly with pre-determined values. The values themselves can also be individually changed within the DataFrames. 

    After the BackendSpec is passed through the loader, modifiers, and samplers, a FakeBackendV2 object is created from the new information, and is then passed into the transpiler. 

    ## Example Code

    Following are some examples applications of BackendSpec.

    #### Creating and transpiling a BackendSpec from an existing backend

    ```
    IBMProvider.save_account('your_API_token')
    provider = IBMProvider()    
    parent_backend = provider.get_backend('ibm_backend')

    spec = BackendSpec(parent_backend)

    # increase qubits to 10
    spec.increase_qubits(1, 'hexagonal')

    # get circuit ready for transpiling
    qc = QuantumCircuit(spec.num_qubits,spec.num_qubits)
    qc.x(0)
    qc.cx(0,1)
    qc.h(1)

    backend = spec.new_backend()

    plot_error_map(backend)
    ```

## Freezing properties
When creating a new backend the user can either generate new instruction and qubit properties from a distribution created from the `BackendSpec.xxxx_properties`. This may not be the wanted flow at all times, to allow users to create static values that will be used by the backend (and not sampled for). Freezing a property will turn that value into a static one which will stay until backend generation. 

An example of this is creating an empty backend and setting all T1 times to 100µs. We can freeze these qubit properties (using` BackendSpec.set_frozen_qubit_property`), the resulting backend from `BackendSpec.new_backend()` will have qubits with T1 times of 100µs. (Rather than T1 times around 100µs if they were non-frozen properties)

# Methods

## Load methods:
- `BackendSpec(parent)`
    - Loads parent data into BackendSpec
- `BackendSpec.from_parent(parent)`
    - Loads parent data into BackendSpec

---
## Modifiers:
### Basis gates:
- `add_basis_gate_distribution(new_gate, distribution_error, distribution_length, freeze_property = False)`
    - Adds new basis gates with instruction properties from distributions provided in `distribution_error` and `distribution_length`. Both arguments are expected to be formatted `distribution_xxx` = [`std_xxx`, `mean_xxx`].  Allows for freezing of new properties. *Only intended for single qubit gates.*

- `add_basis_gate_numeric(gate_name, error_vals, length_vals, freeze_property = False)`
    - Adds new basis gate with user provider gate properties. Allow for freezing of new properties. *Only intended for single qubit gates.*
- `remove_basis_gate(gate_name)`
    - Removes the specified basis gate. *Only intended for single qubit gates.*
- `swap_basis_gate(old_gate, new_gate)`
    - Replaces `old_gate` with `new_gate` in internal properties and basis gates. `new_gate` inherits the `old_gate`'s properties. *Only intended for single qubit gates.*
- `swap_basis_gate_distribution(old_gate, new_gate, distribution_error, distribution_length, freeze_property = False)`
    - Replaces `old_gate` with `new_gate` in internal properties and basis gates. Generates instruction properties using distribution provided. Both distribution arguments are expected to be formatted `distribution_xxx` = [`std_xxx`, `mean_xxx`].Allows for freezing of new properties. *Only intended for single qubit gates.*
- `swap_basis_gate_numeric(old_gate, new_gate, error_vals, length_vals, freeze_property = False)`
    - Replaces `old_gate` with `new_gate` in internal properties and basis gates. Uses provided property values to set new instruction properties. Allows for freezing of new properties. *Only intended for single qubit gates.*
- `swap_2q_basis_gate(old_gate, new_gate, freeze_property = False)`
    - Replaces `old_gate` with `new_gate`.`new_gate` inherits the `old_gate`'s properties. *Only intended for two qubit gates.*
- `swap_2q_basis_gate_distribution(old_gate, new_gate, distribution_error, distribution_length, freeze_property = False)`
    - Replaces `old_gate` with `new_gate` in internal properties and basis gates. Generates instruction properties using distribution provided. Both distribution arguments are expected to be formatted `distribution_xxx` = [`std_xxx`, `mean_xxx`].Allows for freezing of new properties. *Only intended for two qubit gates.*
- `swap_2q_basis_gate_numeric`
    - Replaces `old_gate` with `new_gate` in internal properties and basis gates. Uses provided property values to set new instruction properties. Allows for freezing of new properties. *Only intended for two qubit gates.*
---
### Coupling Map (num_qubits)
- `coupling_change(coupling_type)`
    - Uses provided `coupling_type` to generate new coupling map with at minimum qubits as the number of qubits initially.
- `increase_qubits(increase_amount, coupling_type)`
    - Replaces spec coupling map with a coupling map with at least `BackendSpec.num_qubits + increase_amount` qubits. Regenerates internal properties dataframes with distribution based on initial internal properties dataframes.
- `decrease_qubits(decrease_amount, coupling_type)`
    - Replaces spec coupling map with a coupling map with at least `BackendSpec.num_qubits + decrease_amount` qubits. Regenerates internal properties dataframes with distribution based on initial internal properties dataframes. *decrease_amount must be negative*
- `set_coupling_map(coupling_map, coupling_type)`
    - Replaces internal coupling map, number of qubits, coupling type and resets all frozen properties. Also causes internal dataframe properties to be resampled.
---
### Scale
- `scale_gate_property(gate_name, property_key, scale_factor)`
    - Scales all gate's property with key:`property_key` and name:`gate_name` by `scale_factor`
- `scale_qubit_property(property_key, scale_factor)`
    - Scales all of the qubit's property: `property_key` by `scale_factor`
---
### Setters
#### Properties
- `set_bidirectional(bidirectional)`
    - Sets bidirectional coupling map, value is used for coupling_changes and xxcrease_qubits.
- `set_dt(dt)`
    - Sets the system time resolution of input signals
- `set_max_circuits(max_circuits)`
    - Sets the maximum number of circuits that can be run in a single job. 
- `set_seed(seed)`
    - Sets seed used for distribution generation and sampling
#### Freeze/Unfreeze Properties
- `set_frozen_gate_property(frozen, gate_name, gate_property, qubits)`
    - Freezes/Unfreezes specific gate property acting on `qubits` 
- `set_frozen_gates_property(frozen, gate_name, gate_property)`
    - Freezes/Unfreezes all `gate_property` values.
- `set_frozen_qubit_property(frozen, qubit_property, qubit_id)`
    - Freezes/Unfreezes specific qubit property on specific qubit.
- `set_frozen_qubits_property(frozen, qubit_property)`
    - Freezes/Unfreezes all `qubit_property` values.
#### Set Properties (DataFrame)
- `set_gate_property(gate_name, gate_property, qubits, value, freeze_property = False)`
    - Sets `gate_property` of `gate_name` acting on `qubits` to `value`. Allows for Freezing of new properties.
- `set_gate_properties(gate_name, gate_property, qubits, values, freeze_property = False)`
    - Sets all `gate_property` of `gate_name` to `values`. Allows for Freezing of new properties.
- `set_gate_properties_distribution(gate_name, gate_property, std, mean, freeze_property = False)`
    - Sets all `gate_property` of `gate_name` to values generated values using distribution based on `std` and `mean` arguments. Allows for Freezing of new properties.
- `set_qubit_property(qubit_id, qubit_property, value, freeze_property = False)`
    - Sets the specific qubit's (`qubit_id`) `qubit_property` to `value`. Allows for Freezing of new properties.
- `set_multi_qubit_property(qubits, qubit_property , values, freeze_property = False)`
    - Sets the specified qubits (`qubits`) `qubit_property` to `values`. Allows for Freezing of new properties.
- `set_qubits_properties_distribution(qubits,  qubit_property, std, mean, freeze_property = False)`
    - Sets the specified qubits (`qubits`) `qubit_property` to values sampled from distribution based on argument `std` and `mean` values.
---
## Sampler
- `sample_distribution(std, mean, count)`
    - Returns `np.array` of size count with values sampled from distribution based off input `std` and `mean`

### Backend Generation
- `new_backend()`
    - Returns `FakeBackendV2` object with target and coupling map objects based on BackendSpec properties.