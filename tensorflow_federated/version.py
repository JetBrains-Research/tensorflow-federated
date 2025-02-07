# Copyright 2019, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TensorFlow Federated version."""

import platform
import os
import sys
import re

def get_glibc_version():
    if sys.platform.startswith('linux'):
        try:
            output = os.popen('ldd --version 2>&1').readline()
            match = re.search(r"(\d+(?:\.\d+)+)", output)
            if match:
                return match.group(1).replace('.', '_').strip()

            return None

        except Exception as e:
            print(f"Error detecting glibc version: {e}")
            return  None
    else:
        return None


def get_system_architecture():
    arch = platform.machine()
    if arch == "x86_64" or arch == "amd64":
        return "x86_64"
    elif arch == "i386" or arch == "i686":
        return "x86"
    elif arch.startswith('armv'):
        return "arm"
    elif arch.startswith('aarch64') or arch == "arm64":
        return "aarch64"
    elif arch.startswith('ppc'):
        return "ppc"
    elif arch.startswith('s390'):
        return "s390"
    return arch

def get_platform_name():
    system = platform.system()
    arch = get_system_architecture()

    if system == "Linux":
        glibc_version = get_glibc_version()
        if glibc_version:
            return f"manylinux_{glibc_version}_{arch}"

    return f"unknown_{arch}"

__plat_name__ = get_platform_name()
__version__ = '0.86.4a1'

if __name__ == "__main__":
    print(f"__plat_name__ = {__plat_name__}")


