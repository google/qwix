# How to Contribute

We would love to accept your patches and contributions to this project.

## Development Environment Setup

### Prerequisites

-   Python 3.10 or higher
-   We recommend using [venv](https://docs.python.org/3/library/venv.html) to
    create a virtual environment.

### Setting Up Your Environment

#### 1. Clone the Repository

```bash
git clone https://github.com/google/qwix.git
cd qwix
```

#### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies and Qwix in Development Mode

This step installs Qwix in editable mode, along with its dependencies. This
installs the CPU version of JAX. For GPU/TPU support, please install the
[appropriate JAX version](https://docs.jax.dev/en/latest/installation.html).

If you are contributing to Qwix code, we recommend installing the `dev`
dependencies, which include tools for testing:

```bash
pip install -e ".[dev]"
```

If you only want to install Qwix and its core dependencies, you can run: `bash
pip install -e .`

### Running Tests

You can run the test suite using `pytest`.

Examples:

```bash
# Run all unit tests
python -m pytest tests/

# Run a specific test file
python -m pytest tests/_src/model_test.py

# Run tests with verbose output
python -m pytest tests/ -v

# Run integration tests
python -m pytest integration_tests/
```

Alternatively, you can run individual test files directly:

```bash
python tests/_src/model_test.py
```

## Building the Documentation Locally

### Prerequisites

*   Python 3.10+
*   A virtual environment is recommended.

### Setup

From the root of the repository, run the following commands to set up your
environment and install dependencies for documentation generation:

1.  **Create and activate a virtual environment:** ``shell python -m venv venv
    source venv/bin/activate # On Windows, use `venv\Scripts\activate` ``

2.  **Install the project and documentation dependencies:** `shell pip install
    -e ".[docs]"`

### Building and Viewing

Once the setup is complete, you can build and view the documentation.

1.  **Build the HTML:** From the repository root, run: `shell sphinx-build
    docs/source docs/build/html`

2.  **Serve the files locally:** `shell python -m http.server --directory
    docs/build/html` Open your web browser to `http://localhost:8000` to view
    the documentation.

## Before you submit your PR

### Sign our Contributor License Agreement

Contributions to this project must be accompanied by a
[Contributor License Agreement](https://cla.developers.google.com/about) (CLA).
You (or your employer) retain the copyright to your contribution; this simply
gives us permission to use and redistribute your contributions as part of the
project.

If you or your current employer have already signed the Google CLA (even if it
was for a different project), you probably don't need to do it again.

Visit <https://cla.developers.google.com/> to see your current agreements or to
sign a new one.

### Review our Community Guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google/conduct/).

## Contribution process

### Code Reviews

All submissions, including submissions by project members, require review. We
use [GitHub pull requests](https://docs.github.com/articles/about-pull-requests)
for this purpose.
