import pathlib

from setuptools import find_packages, setup

HERE = pathlib.Path(__file__).parent
README = (HERE / 'README.md').read_text()
REQUIRES = [r.strip() for r in (HERE / 'requirements.txt').read_text().split('\n')]

setup(
    name='klejbenchmark_baselines',
    version='0.1.0',
    author='Allegro: Machine Learning Research',
    author_email='klejbenchmark@allegro.pl',
    description='Baseline models for KLEJ Benchmark.',
    long_description=README,
    long_description_content_type='text/markdown',
    license='Apache 2 License',
    url='https://github.com/allegro/klejbenchmark-baselines',
    python_requires='>=3.6.0',
    install_requires=REQUIRES,
    packages=find_packages(),
)
