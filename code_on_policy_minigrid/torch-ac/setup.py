from setuptools import setup, find_packages

setup(
    name="torch_ac",
    version="1.1.0",
    keywords="reinforcement learning, actor-critic, a2c, ppo, multi-processes, gpu",
    packages=find_packages(),
    install_requires=[
        "numpy==1.19.5",
        "torch==1.9.0",
	"six==1.16.0"
    ]
)
