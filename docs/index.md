# Random Symplectic
Python library for generating uniformly random d-nary symplectic matrices for prime $d$. These matrices form a representation of the $n$-qudit Clifford  group [@hostens2005] and can be decomposed into elementary 1- and 2-qudit gates. Uniform sampling from the Clifford group is a critical subroutine to randomized benchmarking [@magesan2012;@hashim2025].

The algorithm implements a mapping from the integers to $\text{Sp}(2n, \mathbb{Z}_d)$, reducing sampling from the symplectic group to sampling an integer, and is an extension of the qubit-based method presented in \textcite{koenig2014}. A manuscript detailing the algorithm is forthcoming.

\bibliography

## Installation
To install the package for Python 3.8+, do
```
pip install random-symplectic
```


## Getting Started
The package

