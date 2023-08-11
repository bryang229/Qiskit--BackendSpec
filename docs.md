# Framework for Generating Experimental Backends (BackendSpec)
### Bryan Bonilla Garay, Matthew Figueroa, Vea Iyer

________________

There are many aspects of IBM Quantum backends that require meticulous testing with the transpiler. The purpose of BackendSpec is to provide a framework for generating highly customized backends that can be used for more diverse testing methods. This framework would allow for exploring new hardware architectures with different coupling schemes, qubit types, qubit specific basis gates and  noise aware transpilation. The overall flow of the framework is shown here:

![Framework](https://drive.google.com/file/d/1HKytMdon5ePxUF697L7sq0fdvWHmZx0o/view?usp=drive_link "Optional title")

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

    `QubitProperties` and `GateProperties` objects contain the majority of information regarding any generated backend. `QubitProperties` consist of various properties associated with qubits, such as:

    - T1 rates (dechoerence)
    - T2 rates (loss of phase)
    - Frequency
    - Anharmonicity
    - Readout errors
    - Readout lengths

    `GateProperties` consist of properties associated with single and multi-qubit gates, such as:

    - Gate errors
    - Gate lengths

    The modifer functions allow for these properties to be selected, flagged, and changed. 

    ### Samplers

    Since there are a variety of qubit and gate properties, they can be modelled using a normal distribution, which generate about 10000 samples per attribute. Then, the corresponding number of values for the `QubitProperties` and `GateProperties` are chosen using a uniform random distribution. The sampled values are fed into the respective DataFrames, and are used when the modifiers are called. This is especially useful when more qubits are introduced in the backend, or if the backend is being newly generated. 

    The sampler functions help generate values for  `QubitProperties` and `GateProperties` attributes, either with respect to a probability distribution with a provided mean and standard deviation, or directly with pre-determined values. The values themselves can also be individually changed within the DataFrames. 

    After the BackendSpec is passed through the loader, modifiers, and samplers, a FakeBackendV2 object is created from the new information, and is then passed into the transpiler. 

    ## Description of Data Structures

    The `QubitProperties` and `GateProperties` are stored in Pandas `DataFrames`. as seen below:

    ![QubitProperties](https://drive.google.com/file/d/10uO9iFV4uQfeYKOaJPw1yQ4gDR68kJoz/view?usp=drive_link "Optional title")

    ![GateProperties](https://drive.google.com/file/d/1znyv8mm9ScONt-5IZi5jpyn4thQJXtOq/view?usp=drive_link "Optional title")

    The `DataFrames` allow for the backend data to be more streamlined and accessible. 

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
