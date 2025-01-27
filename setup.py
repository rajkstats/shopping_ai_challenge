from setuptools import setup, find_packages

setup(
    name="shopping_lens",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "fastapi",
        "uvicorn",
        "python-multipart",
        "Pillow",
        "numpy",
        "faiss-cpu",
        "scikit-learn",
        "tqdm",
        "opencv-python",
        "imagehash",
        "requests",
        "urllib3",
        "pathlib"
    ],
)
