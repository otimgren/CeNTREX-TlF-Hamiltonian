import setuptools

setuptools.setup(
    name="CeNTREX_TlF_Hamiltonian",
    author="Olivier Grasdijk",
    author_email="olivier.grasdijk@yale.edu",
    description="general utility package for TlF molecular calculations used in the CeNTREX experiment",
    url="https://github.com/",
    packages=setuptools.find_packages(),
    install_requires=["numpy","scipy", "sympy>=1.9", "tqdm",],
    python_requires=">=3.6",
    version="0.1",
)
