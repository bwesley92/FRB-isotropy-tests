import os

_HPC_ENV_VARS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def configure_hpc_environment() -> None:

    for var, value in _HPC_ENV_VARS.items():

        os.environ.setdefault(
            var,
            value,
        )