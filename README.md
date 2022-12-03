# pyinla

INLA in Python

## Notes

- inla.mesh.2d
  - max.edge controls largest triangle length.
    - scalar controls inner domain length, length 2 vector controls inner and outer domain length
- on M1 Mac: run with environment variable `RPY2_CFFI_MODE=BOTH`

## Can't get working

- dimnames of ListVectors
- passing a Python function to inla.{e/t}marginal
