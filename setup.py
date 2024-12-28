import os
from pathlib import Path
import shutil
import sys

from packaging.version import Version
from setuptools import Command, Extension, setup, find_packages

CYTHON_MIN_VERSION = Version("0.29")


# adapted from bottleneck's setup.py
class clean(Command):
    user_options = [("all", "a", "")]

    def initialize_options(self):
        self.all = True
        self.delete_dirs = []
        self.delete_files = []

        for root, dirs, files in os.walk("gradient_growing_trees"):
            root = Path(root)
            for d in dirs:
                if d == "__pycache__":
                    self.delete_dirs.append(root / d)

            if "__pycache__" in root.name:
                continue

            for f in (root / x for x in files):
                ext = f.suffix
                if ext == ".pyc" or ext == ".so":
                    self.delete_files.append(f)

                if ext in (
                    ".c",
                    ".cpp",
                ):
                    source_file = f.with_suffix(".pyx")
                    if source_file.exists():
                        self.delete_files.append(f)

        build_path = Path("build")
        if build_path.exists():
            self.delete_dirs.append(build_path)

    def finalize_options(self):
        pass

    def run(self):
        for delete_dir in self.delete_dirs:
            shutil.rmtree(delete_dir)
        for delete_file in self.delete_files:
            delete_file.unlink()


EXTENSIONS = {
    "_criterion": {
        "sources": ["gradient_growing_trees/tree/_criterion.pyx"],
        "language": "c++",
    },
    "_parent_info": {
        "sources": ["gradient_growing_trees/tree/_parent_info.pyx"],
        "language": "c++",
    },
    "_tree": {
        "sources": [
            "gradient_growing_trees/tree/_tree.pyx",
        ],
        "language": "c++",
    },
    "_splitter": {
        "sources": [
            "gradient_growing_trees/tree/_splitter.pyx",
        ],
        "language": "c++",
    },
    "_partitioner": {
        "sources": [
            "gradient_growing_trees/tree/_partitioner.pyx",
        ],
        "language": "c++",
    },
    "_utils": {
        "sources": [
            "gradient_growing_trees/tree/_utils.pyx",
        ],
        "language": "c++",
    },
}


def get_module_from_sources(sources):
    for src_path in map(Path, sources):
        if src_path.suffix == ".pyx":
            return ".".join(src_path.parts[:-1] + (src_path.stem,))
    raise ValueError(f"could not find module from sources: {sources!r}")


def _check_cython_version():
    message = (
        f"Please install Cython with a version >= {CYTHON_MIN_VERSION} in order to build a scikit-learn from source."
    )
    try:
        import Cython
    except ModuleNotFoundError:
        # Re-raise with more informative error message instead:
        raise ModuleNotFoundError(message)

    if Version(Cython.__version__) < CYTHON_MIN_VERSION:
        message += f" The current version of Cython is {Cython.__version__} installed in {Cython.__path__}."
        raise ValueError(message)


def cythonize_extensions(extensions):
    """Check that a recent Cython is available and cythonize extensions"""
    _check_cython_version()
    from Cython.Build import cythonize

    # http://docs.cython.org/en/latest/src/userguide/source_files_and_compilation.html#cythonize-arguments
    directives = {"language_level": "3"}
    cy_cov = os.environ.get("CYTHON_COVERAGE", False)
    macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    if cy_cov:
        directives["linetrace"] = True
        macros.append(("CYTHON_TRACE", "1"))
        macros.append(("CYTHON_TRACE_NOGIL", "1"))

    for ext in extensions:
        if ext.define_macros is None:
            ext.define_macros = macros
        else:
            ext.define_macros += macros

    return cythonize(extensions, compiler_directives=directives)


def get_extensions():
    import numpy

    numpy_includes = [numpy.get_include()]
    # _check_eigen_source()

    DEFAULT_FLAGS = ['-O3']
    extensions = []
    for config in EXTENSIONS.values():
        name = get_module_from_sources(config["sources"])
        include_dirs = numpy_includes + config.get("include_dirs", [])
        extra_compile_args = DEFAULT_FLAGS + config.get("extra_compile_args", [])
        language = config.get("language", "c")
        ext = Extension(
            name=name,
            sources=config["sources"],
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            language=language,
        )
        extensions.append(ext)

    # Skip cythonization as we do not want to include the generated
    # C/C++ files in the release tarballs as they are not necessarily
    # forward compatible with future versions of Python for instance.
    if "sdist" not in sys.argv and "clean" not in sys.argv:
        extensions = cythonize_extensions(extensions)

    return extensions


with open("requirements.txt") as f:
    install_requirements = f.read().splitlines()


if __name__ == "__main__":
    setup(
        name='gradient_growing_trees',
        version='1.0.0',
        author='Polytech NTAILab',
        description='Gradient-based tree growth algorithms.',
        ext_modules=get_extensions(),
        cmdclass={"clean": clean},
        packages=find_packages(),
        install_requires=install_requirements,
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent"],
        license='MIT',
        python_requires='>=3.10',
    )