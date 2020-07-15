import tempfile

import nox

locations = "src", "tests", "noxfile.py"
nox.options.sessions = "lint", "safety", "tests"


def install_with_constraints(session, *args, **kwargs):
    with tempfile.NamedTemporaryFile() as requirements:
        session.run(
            "poetry",
            "export",
            "--dev",
            "--format=requirements.txt",
            f"--output={requirements.name}",
            external=True,
        )
        session.install(f"--constraint={requirements.name}", *args, **kwargs)


@nox.session()
def tests(session):
    args = session.posargs or ["--cov"]
    session.run("poetry", "install", external=True)
    session.run("pytest", *args)


@nox.session()
def lint(session):
    args = session.posargs or locations
    install_with_constraints(
        session, "flake8", "flake8-black", "flake8-bugbear", "flake8-bandit"
    )
    session.run("flake8", *args)


@nox.session()
def black(session):
    args = session.posargs or locations
    install_with_constraints(session, "black")
    session.run("black", *args)


@nox.session()
def safety(session):
    with tempfile.NamedTemporaryFile() as requirements:
        session.run(
            "poetry",
            "export",
            "--dev",
            "--format=requirements.txt",
            "--without-hashes",
            f"--output={requirements.name}",
            external=True,
        )
        install_with_constraints(session, "safety")
        session.run("safety", "check", f"--file={requirements.name}", "--full-report")
