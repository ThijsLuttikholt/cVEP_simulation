# Description of the implementation

This folder contains various python files that were used in the master's thesis of Thijs Luttikholt. The descriptions of the main contents of the important files will now be provided. Any functions that are deemed to not be straightforward are explained. 

### EEG_generator.py

This file contains the base of the forward model (simulation framework) that was used in the study. To be precise, the EEG_generator class contains various functions used to generate data. However, the generator itself must be initialized with a version of the noiseGen class and a version of the signal_gen_infer class. The EEG_generator class contains 8 main functions:

- genN(): This function generates data using the transient responses recorded in the signal_gen_infer class and noise that is generated using pre-defined values.
- genNDrawn1(): This function generates data using the transient responses recorded in the signal_gen_infer class and noise that is generated by sampling parameter values from distributions.
- GenNDrawn2(): This function generates data using the transient responses simulated with the quadruple-Gamma function by drawing parameter values. This implementation was not used in the final analyses in the thesis. 
- GenNDrawn2_2(): This function generates data using the transient responses simulated with the quadruple-Gamma function by drawing parameter values in a relative design. This implementation was not used in the final analyses in the thesis. 
- GenNDrawn2_3(): This function generates data using the Sig version of the forward model and noise simulated by sampling parameters from distributions. 
- GenNDrawn2_3_rel(): This function generates data using the adapted version of the Sig version of the forward model, with parameters for long events being dependent on those for short events. Noise is simulated by sampling parameters from distributions. 
- overlay2(): A function that performs the overlaying (convolution) operation given signals and a code.
- find_duration_idx(): A helper function for the overlay2 function.

### EEGNet.py

This file mainly contains the EEGNet architecture and various helper functions/classes. 

### extra_functions.py

This file mainly contains various helper functions, either used in any of the .py files or in external analyses. 

- get_freq_dom(): Calculates a frequency spectrum.
- plot_accuracy(): Plots the accuracy given a score dataframe recorded in the used EEGNet training.
- plot_loss(): Plots the loss given a score dataframe recorded in the used EEGNet training.
- splice data(): this function is used for splitting data (31.5 seconds in the thesis) into smaller parts (2.1 seconds in the thesis)
- split_one(): this function is used to split one trial. 
- gaussian_drawer(): Implements the truncated normal distribution that was used in the thesis.

### final_evaluations_functions.py

This file contains the main functionality to perform the analyses that were reported in the thesis. 

- train_net_final(): This function is used to train the EEGNet architecture.
- test_net_final(): This function is used to test the EEGNet architecture.
- do_EEGNet_sim_emp_final(): Performs the sim>emp analysis with the EEGNet inverse model and a provided forward model. 
- do_CCA_sim_emp_final(): Performs the sim>emp analysis with the CCA inverse model and a provided forward model.
- do_EEGNet_emp_emp_final(): Performs the emp>emp analysis with the EEGNet inverse model. 
- do_CCA_emp_emp_final(): Performs the emp>emp analysis with the CCA inverse model.
- do_EEGNet_sim_sim_final(): Performs the sim>sim analysis with the EEGNet inverse model and a provided forward model. 
- do_CCA_sim_sim_final(): Performs the sim>sim analysis with the CCA inverse model and a provided forward model.

### LR_scheduler.py

This file contains only the implementation of the learning rate scheduler that was used in the thesis. 

### noise_generator.py

This file contains the implementation of the noise modeling aspect of the forward model. The class noiseGen contains various functions: 

- genPink(): Simulates pink noise
- genGauss(): Simulates Gaussian white noise
- genAlpha(): Simulates visual alpha rhythm noise
- genFreqSpike(): Simulates line noise
- convSame(): A helper function for the genBlink function, which was not used in the thesis. 
- genBlink(): Simulates eye blink artifacts, not used in the thesis.
- genEyeMov(): Simulates eye movement artifacts, not used in the thesis.
- genLogU(): Simulates log uniform noise, not used in the thesis.
- genNoise(): Generates a single trial of noise given parameters. Used in the genNoiseN function
- genNoiseN(): Generates N trials of noise given parameters.
- drawN(): Generates N trials of noise by using noise parameters that are sampled from distributions.
- drawNoiseParams(): Samples noise parameters from the corresponding distributions.

### preprocessor.py

This file only contains an implementation of the preprocessing that was used in the thesis. 

### signal_gen_infer.py

This file contains the main implementation of the signal modeling component that was used in the thesis. Due to the large amount of functions, they will be described in groups rather than individually. 

- changeSign() - full_templates(): These functions are used for acquiring the response vectors using CCA and empirical data. The changeSign function is used to ensure that the sign of each response is similar. 
- getN(): Randomly samples N sets of response pairs (for short and long events)
- getCodes()-get_needed_values(): Various get-functions for acquiring variables saved in the class. 
- drawN() - keepDraw(): Functions for generating signals with the quadruple-Gamma function, by sampling parameters from distributions. This was not used in the thesis.
- subGamma() - genCustS4Adapt(): Functions used for modeling a signal using the quadruple-Gamma function, given parameters. 
- genCustS4Adapt2() - keepDraw2(): Functions for generating signals with a relative version of the quadruple-Gamma function with sampling from distributions.
- drawN3(): Used for generating signals with the Sig model as introduced in the thesis. Makes use of the signal_simulator class.
- drawN3(): Used for generating signals with the adapted version of the Sig model, as discussed in the appendix in the thesis but not used in the final analyses. Makes use of the signal_simulator class.

### signal_simulator.py

This file contains the functionality for generating signals with the Sig version of the forward model. Once again, the functions will be discussed in groups. 

- make_s() - intermediate(): The functions used for the actual modeling of the signal, based on parameter values. 
- draw_wide_gaussian() - help_dist(): Functions that are used for creating the parameter distributions.
- drawN(): The main function that is used to generate signals with the Sig version of the forward model. It used the functions for parameter sampling and signal modeling. 
- drawPars() - drawX0s(): The functions that are used for the actual sampling of parameter values.
- drawN_rel() - intermediate_rel(): All functionality for the adapted version of the Sig model, as discussed in the appendix of the thesis. 