# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit helps, and credit will always be given.

You can contribute in many ways:

- Report bugs at [https://github.com/pblankenau2/gapfill-landsat/issues](https://github.com/pblankenau2/gapfill-landsat/issues).
- Submit feedback and feature requests at [https://github.com/pblankenau2/gapfill-landsat/issues](https://github.com/pblankenau2/gapfill-landsat/issues).
- Write tests, fix bugs, or implement new features.

To set up `gapfill-landsat` for local development:

1.  Fork the `gapfill-landsat` repo on GitHub.
2.  Clone your fork locally:

    ```console
    $ git clone git@github.com:your_name_here/gapfill-landsat.git
    ```

3.  Install your local copy into a virtualenv. Assuming you have `uv` installed, this is how you set up your fork for local development:

    ```console
    $ cd gapfill-landsat/
    $ uv venv
    $ uv pip install -e .[dev]
    ```

4.  Create a branch for local development:

    ```console
    $ git checkout -b name-of-your-bugfix-or-feature
    ```

    Now you can make your changes locally.

5.  When you're done making changes, check that your changes pass the tests, including testing other Python versions with `tox`:

    ```console
    $ tox
    ```

6.  Commit your changes and push your branch to GitHub:

    ```console
    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature
    ```

7.  Submit a pull request through the GitHub website.
