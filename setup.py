from setuptools import setup

__author__ = "Ethereal AI"

setup(
    name="xlr8",
    version="0.0.1",
    packages=["xlr8"],
    url="https://github.com/Ethereal-AI/xlr8",
    author=__author__,
    description="Fast cosine similarity for Python",
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
    ],
)
