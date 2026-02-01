# gapfill-landsat

[![PyPI](https://img.shields.io/pypi/v/gapfill-landsat.svg)](https://pypi.org/project/gapfill-landsat/)
[![CI](https://github.com/pblankenau2/gapfill-landsat/actions/workflows/ci.yml/badge.svg)](https://github.com/pblankenau2/gapfill-landsat/actions/workflows/ci.yml)
[![Documentation Status](https://github.com/pblankenau2/gapfill-landsat/actions/workflows/ci.yml/badge.svg)](https://pblankenau2.github.io/gapfill-landsat/)

Tools for filling missing data in satellite images caused by sensor malfunctions or masked out clouds.
Currently, this package only implements the nearest similar pixel interpolator (NSPI) algorithm
that was specifically designed to fill the Landsat 7 SLC off gap.  It can also be used to fill
small areas where clouds have been masked as nodata.

![Landsat 7 SLC off gap filled](docs/assets/NSPI3.png)

* Free software: MIT license
* Documentation: https://pblankenau2.github.io/gapfill-landsat/