import platform
import sys

def get_plat_name():
    if platform.machine() == "aarch64":
        return "manylinux_2_31_aarch64"
    else:
        return "manylinux_2_31_x86_64"

if __name__ == "__main__":
    plat_name = get_plat_name()
    sys.exit(plat_name)