from setuptools import find_packages, setup

setup(
    name="heracles_agents",
    version="0.0.1",
    url="",
    author="Aaron Ray",
    author_email="aray.york@gmail.com",
    description="Experimental evaluation framework for investigating interfaces between 3D scene graphs and LLMs.",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"": ["resources/*"]},
    install_requires=[
        "pydantic-settings",
        "plum-dispatch >= v2.7.0",
        "lark",
        "tiktoken",
        "typer",
        "spark-dsg",
        "textual",
        "heracles @ git+https://github.com/GoldenZephyr/heracles.git#subdirectory=heracles",
    ],
    extras_require={
        "openai": ["openai"],
        "anthropic": ["anthropic"],
        "bedrock": ["boto3"],
        "openrouter": ["openrouter"],
        "ollama": ["ollama"],
        "huggingface": [],
        "all": [
            "openai",
            "anthropic",
            "ollama",
            "boto3",
            "openrouter",
        ],
    },
)
