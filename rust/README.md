Welcome to the experiment to add a Rust backend to CVXPY.

As a prerequisite to building this project, you need to use `pip install maturin`
in the virtual environment you're using for development and ensure that you have
rust installed.

To type-check the rust code, run `cargo check` (which is relatively fast).

To do correctness testing, run `maturin develop`. This compiles the code with a
debug profile and installs it in the virtual environment.

To do performance testing, run `maturin develop --release`. This compiles the
code with a release profile (which can have a 100x performance improvement)
and installs it in the virtual environment.
