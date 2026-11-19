"""Allow python -m marlite, including under torchrun."""

from marlite.cli import main

if __name__ == "__main__":
    main()
