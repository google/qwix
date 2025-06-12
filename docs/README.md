# Building the Documentation Locally

### Prerequisites

* Python 3.8+
* A virtual environment is recommended.

### Setup

From the root of the repository, run the following commands to set up your
environment and install dependencies.

1.  **Create and activate a virtual environment:**
    ```shell
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

2.  **Install the project and documentation dependencies:**
    ```shell
    pip install -e .
    pip install -r docs/requirements.txt
    ```

### Building and Viewing

Once the setup is complete, you can build and view the documentation.

1.  **Build the HTML:**
    From the repository root, run:
    ```shell
    sphinx-build docs/source docs/build/html
    ```

2.  **Serve the files locally:**
    ```shell
    python -m http.server --directory docs/build/html
    ```
    Open your web browser to `http://localhost:8000` to view the
    documentation.
