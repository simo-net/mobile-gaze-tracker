import setuptools
# import runpy
# import os

# root = os.path.dirname(os.path.realpath(__file__))
# version = runpy.run_path(os.path.join(root, "mobile-gaze-tracker", "version.py"))["version"]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="MobileGazeTracker",
    version="0.1",
    author="Simone Testa",
    author_email="simonetesta994@gmail.com",
    description="Smartphone eye tracking based on deep learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=False,
    packages=setuptools.find_packages(),
    install_requires=["numpy", "tqdm", "sklearn", "torch", "torchvision", "matplotlib", "opencv-python", "dlib"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Linux",
        "Intended Audience :: Science/Research",
    ],
)
