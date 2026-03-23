import os
import site
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install


def _install_pth():
    """Install .pth file for auto-patching in TP worker subprocesses."""
    try:
        dirs = site.getsitepackages()
    except AttributeError:
        # virtualenv without site-packages
        dirs = [site.getusersitepackages()]
    for d in dirs:
        try:
            pth = os.path.join(d, "verilm_capture.pth")
            with open(pth, "w") as f:
                f.write("import verilm._startup\n")
            break
        except OSError:
            continue


class PostDevelop(develop):
    def run(self):
        super().run()
        _install_pth()


class PostInstall(install):
    def run(self):
        super().run()
        _install_pth()


setup(
    name="verilm",
    version="0.3.0",
    description="VeriLM sidecar: capture, commit, and audit for verified LLM inference",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=["torch"],
    entry_points={
        "vllm.general_plugins": [
            "vi_capture = verilm:register",
        ],
    },
    cmdclass={
        "develop": PostDevelop,
        "install": PostInstall,
    },
)
