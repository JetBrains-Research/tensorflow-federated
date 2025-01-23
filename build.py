import platform
import sys
import importlib

def get_plat_name():
    if platform.machine() == "aarch64":
        return "manylinux_2_31_aarch64"
    else:
        return "manylinux_2_31_x86_64"

if __name__ == "__main__":
    setuptools = importlib.import_module('setuptools')
    setuptools.setup(
        bdist_wheel={
            'plat_name': get_plat_name(),
        },
    )