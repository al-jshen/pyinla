# pyinla

INLA in Python

**See [examples](examples).**

## Notes

- on M1 Mac: run with environment variable `RPY2_CFFI_MODE=BOTH`

## Can't get working

- getting dimnames of ListVectors
- passing a Python function to inla.{e/t}marginal
  - current workaround: pass an R function as a string. pretty intuitive/similar to python for simple functions (e.g., "exp(x\*\*2 + 1)")
- converting R's NA to Python, numpy has no int nan, and with numpy's converter NA goes to -2147483648
  - current workaround: convert int vector to numpy array, then if any values are -2147483648, then convert the array to a float array and turn the -2147483648 into a nan
