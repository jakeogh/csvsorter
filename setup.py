# -*- coding: utf-8 -*-

import sys
import fastentrypoints
from setuptools import find_packages, setup
if not sys.version_info[0] == 3:
    sys.exit("Python 3 is required. Use: \'python3 setup.py install\'")

dependencies = ["icecream", "click", "colorama", "click-command-tree"]

config = {
    "version": "0.1",
    "name": "csvsorter",
    "url": "https://github.com/jakeogh/csvsorter",
    "license": "LGPL-2",
    "author": "Justin Keogh",
    "author_email": "github.com@v6y.net",
    "description": "@click version of https://github.com/richardpenman/csvsort",
    "long_description": __doc__,
    "packages": find_packages(exclude=['tests']),
    "package_data": {"csvsorter": ['py.typed']},
    "include_package_data": True,
    "zip_safe": False,
    "platforms": "any",
    "install_requires": dependencies,
    "entry_points": {
        "console_scripts": [
            "csvsorter=csvsorter.csvsorter:cli",
        ],
    },
}

setup(**config)