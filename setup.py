# 설정은 pyproject.toml 에 있다. 이 파일은 setuptools < 64 에서 `pip install -e .` (개발 설치)를 쓰기 위한 호환용.
from setuptools import find_packages, setup

setup(
    name="btqa-game",
    version="0.1.0",
    package_dir={"": "python"},
    packages=find_packages("python", include=["btqa_game*"]),
    python_requires=">=3.8",
    entry_points={"console_scripts": ["btqa-game = btqa_game.__main__:main"]},
)
