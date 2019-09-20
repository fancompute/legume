from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    readme = f.read()

dependencies = [
        'numpy',
        'scipy',
        'matplotlib',
        'autograd'
]

setup(
    name='legume',
    version='0.0.1',
    description='GuidedModeExpansion',
    long_description=readme,
    long_description_content_type="text/markdown",
    author='Momchil Minkov',
    author_email='momchilmm@gmail.com',
    url='https://github.com/fancompute/legume',
    packages=find_packages(),
    install_requires=dependencies,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)