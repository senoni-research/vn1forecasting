# vn1forecasting

`vn1forecasting` is a Python library for multi-encoder transformer-based demand forecasting, originally developed for the VN1 Forecasting Accuracy Challenge. It addresses real-world forecasting challenges such as partial exogenous data, variable-length sequences, and multiple time-series for `(Client, Warehouse, Product)` triplets.

## ℹ️ Features
- **Multi-Encoder Transformer**: Separates sales and price signals into distinct encoders, handling partially known or unknown future prices.
- **Rolling Aggregates**: Implements short- and medium-term rolling sums (4-week, 13-week) for faster convergence on trend patterns.
- **Phased Training Approach**: Systematically adapts from scenarios with known future price to fully unknown price, improving real-world robustness.
- **Scalable Architecture**: Built with Python 3.11 and managed via [Poetry](https://python-poetry.org/), ensuring reproducible environments.

## ⚡ Getting Started

To get started with `vn1forecasting`, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/senoni-research/vn1forecasting.git
    cd vn1forecasting
    ```

2. Install dependencies and set up the development environment (see the next section).

3. Run the provided scripts and pipelines.

---

## 🌴 Set Up Development Environment

### 1. Install `pyenv` to Manage Python Versions

We recommend using `pyenv` to manage Python versions. If you prefer a different tool, ensure that the Python version matches `3.11.8`.

#### Install Dependencies for `pyenv`
```bash
sudo apt-get update && sudo apt-get install -y \
    make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```

#### Install `pyenv`
```bash
curl https://pyenv.run | bash
```

#### Update Shell Configuration
Add the following to `~/.bashrc` (or equivalent):
```bash
# pyenv
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"
```

#### Restart Your Shell
```bash
exec $SHELL
```

#### Install Python 3.11.8 Using `pyenv`
```bash
pyenv install 3.11.8
pyenv local 3.11.8
```

---

### 2. Install Poetry

Poetry is used to manage dependencies in this project. Install it via:
```bash
pip install --upgrade poetry
```

---

### 3. Install Dependencies

Install the package alongside its dependencies:
```bash
make install
```

This runs `poetry install`, resolving dependencies and generating the `poetry.lock` file.

---

### 4. Activate the Poetry Environment

To activate the virtual environment managed by Poetry:
```bash
poetry env use $(pyenv which python)
source $(poetry env info --path)/bin/activate
```

---

### 5. Add Dependencies

To add a new dependency, use:
```bash
poetry add <package-name>
```

For example:
```bash
poetry add numpy
```

---

## 🚀 Run Scripts

Define scripts in `pyproject.toml` under the `[tool.poetry.scripts]` section. To run a script, use:
```bash
poetry run <script-name>
```

---

## 📦 Build the Package

To build the package for distribution:
```bash
make build-whl
```

This runs `poetry build` and creates `.whl` and `.tar.gz` files in the `dist/` directory.

---

## 🔄 Update Dependencies

To update all project dependencies to their latest compatible versions:
```bash
poetry update
```

---

## 🧪 Testing

Run tests using:
```bash
make test
```

This runs `pytest` and generates coverage reports.

---

## 🤝 Contribution Guidelines

1. Fork the repository and clone it locally.
2. Create a feature branch:
    ```bash
    git checkout -b feature/your-feature-name
    ```
3. Commit your changes and push to your fork.
4. Open a pull request with a clear description of your changes.

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 📝 Contact

For questions or suggestions, please contact **Philippe Dagher** at [nasdag@senoni.com](mailto:nasdag@senoni.com).
