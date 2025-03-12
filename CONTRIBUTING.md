<!-- Add acknowledgements later. -->

## Versions
- This repository supports Python 3.9 or newer.
- Please satisfy the requirements for packages (to avoid conflicts).
- Specified in `requirements.txt`.

## Development

- Please **DO NOT** directly edit the main branch.
- The suggested PR procedures are the following.
  1. Implement codes in the branch named `(your name)-dev` or whatever.
  2. Create PR to the `dev` branch from your working branch.
  3. In the `dev` branch, we will make sure that implanted modules work as desired.
  4. Create PR from `dev` to `main`.

## Formatting and docstring
- Please make sure to apply `black` formatter and check `flake8`.
- Please use the numpy-style docstring. Or more specifically, please follow the documentation format used in OpenBanditPipeline.

## References
- black: https://github.com/psf/black
- PEP8: https://peps.python.org/pep-0008/
- flake8: https://flake8.pycqa.org/en/latest/
- numpy-style docstring: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html
- OpenBanditPipeline: https://github.com/st-tech/zr-obp
