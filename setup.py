# 설정은 pyproject.toml 에 있다. 이 파일은 setuptools < 64 에서 `pip install -e .` (개발 설치)를 쓰기 위한 호환용.
from setuptools import find_packages, setup

setup(
    name="btqa-game",
    version="0.4.0",
    package_dir={"": "python"},
    packages=find_packages("python", include=["btqa_game*", "btqa_play*", "btqa_distill*"]),
    package_data={"btqa_play": ["*.json"], "btqa_distill": ["*.txt"]},
    extras_require={"distill": ["torch>=2.1", "transformers>=4.44", "peft>=0.12", "safetensors"]},
    python_requires=">=3.8",
    entry_points={"console_scripts": ["btqa-game = btqa_game.__main__:main", "btqa-play = btqa_play.__main__:main",
                                      "btqa-distill = btqa_distill.__main__:main"]},
)
