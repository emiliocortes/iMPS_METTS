# TO-DO


## Ancilla implementation
- [ ] finite MPS
- [ ] infinite MPS

## METTS benchmark
Goal: plots similar to [2]
![Heisenberg spin-1/2, collapse bases](papers/heisenberg_sampling_convergence.png)

- [ ] finite MPS
  - Heisenberg 
    - spin 1/2
    - spin 1
  - XXZ spin 1/2

- Do 100 independent runs
- Take N_samples = 10 (_thermal steps_)
- Take the average of the 100 runs at each thermal step

- Question:
  - sample over more thermal steps vs sample over independent runs


## METTS implementation
- Time-series processing
  - Averaging
  - Std estimation


- Symmetric tensors
  
## tanRG
## xTRG

# References
[1] M. Binder and T. Barthel, Minimally entangled typical thermal states versus matrix product purifications for the simulation of equilibrium states and time evolution, Phys. Rev. B 92, 125119 (2015).
[2] B. Bruognolo, J. von Delft, and A. Weichselbaum, Symmetric minimally entangled typical thermal states, Phys. Rev. B 92, 115105 (2015).
