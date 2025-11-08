# Qudit Cliffords
Python library for generating random qudit Clifford gates and decomposing them.

# To do:
- Initial release:
    - Finish documentation
    - Document index->matrix
    - Derive matrix->index algorithm
    - Check on a singleton class for each d.
    - Fix Transvection and .random_symplectic defined twice. 
- Rust implementation
    - Learn Rust
- Tableau simulation
    - Figure out Pauli frame tracking for even d (requried for d=2)
    - Implement with matrix multiplication or row operations?