from setuptools import find_packages, setup

setup(
    name="stack-ai-rag",
    version="0.1.0",
    description="Local-first FastAPI RAG pipeline over PDF knowledge bases",
    packages=find_packages(include=["app", "app.*"]),
    python_requires=">=3.12",
    install_requires=[
        "fastapi>=0.115.0,<1.0.0",
        "uvicorn[standard]>=0.30.0,<1.0.0",
        "pydantic>=2.8.0,<3.0.0",
        "pydantic-settings>=2.4.0,<3.0.0",
        "python-multipart>=0.0.9,<1.0.0",
        "jinja2>=3.1.4,<4.0.0",
        "httpx>=0.27.0,<1.0.0",
        "pypdf>=4.3.1,<5.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.3.0,<9.0.0",
            "pytest-asyncio>=0.24.0,<1.0.0",
            "pytest-cov>=5.0.0,<6.0.0",
            "mypy>=1.11.0,<2.0.0",
            "ruff>=0.6.0,<1.0.0",
            "pre-commit>=3.8.0,<4.0.0",
        ]
    },
)
