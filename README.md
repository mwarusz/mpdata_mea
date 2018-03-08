### [SageMath](https://sagemath.org) scripts used in the paper *["MPDATA: Third-order accuracy for variable flows" (J. Comput. Phys. 359 2018)](https://www.sciencedirect.com/science/article/pii/S0021999118300159)*.

### Requirements
- SageMath (known to work in the version 8.0)
- Tests also require pytest installed in the SageMath own version of Python.  
  You can install it using `$ sage --python -m easy_install pytest`.

### Testing
`$ make` followed by `$ make test` runs the tests.

### Usage

The file **mea.sage** contains a routine to perform the modified equation analysis
for a given flux function. The flux function of MPDATA is defined in **schemes.sage**.
See **test_upwind.sage** and **test_mpdata.sage** for an example of combining the two.
