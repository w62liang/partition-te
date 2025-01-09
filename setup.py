from setuptools import setup, find_packages

setup(
    name="partbte",  
    version="0.1.0",  
    description="Partitioning-based estimators for treatment effect\
    heterogeneity under randomization with binary outcomes",  
    long_description=open("README.md", encoding="utf-8").read(),  
    long_description_content_type="text/markdown",  
    author="Wei Liang",  
    author_email="w62liang@uwaterloo.ca",  
    url="https://github.com/w62liang/partition-te",  
    packages=find_packages(),  
    python_requires=">=3.6",  
    install_requires=[
        "numpy>=1.26.3",  
        "scipy>=1.11.4",
        "scikit-learn>=1.2.2",
        "joblib>=1.2.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": ["pytest>=8.3.4"],  
    },
    classifiers=[
        "Programming Language :: Python :: 3",  
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",  
        "Operating System :: OS Independent",  
        "Intended Audience :: Science/Research",  
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    license="MIT",  
    keywords="causal inference partitioning statistical inference treatment \
        harm rate treatment effect heterogeneity randomized clinical trials",  
)
