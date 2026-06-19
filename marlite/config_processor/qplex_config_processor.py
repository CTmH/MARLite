"""QPLEX config processor.

QPLEX shares the same YAML structure as QMIX (no auxiliary ``V_net``
or extra modules are needed — the joint value is computed inside the
mixer itself).  We reuse :class:`QMIXConfigProcessor` directly and
register under a distinct trainer type.

When a YAML config has:

.. code-block:: yaml

    trainer:
      type: "QPLEX"
      ...

:class:`QPLEXConfigProcessor` is selected by the processor registry and
delegates all parsing to :class:`QMIXConfigProcessor`.
"""

from marlite.config_processor.qmix_config_processor import QMIXConfigProcessor


class QPLEXConfigProcessor(QMIXConfigProcessor):
    """Config processor for QPLEX experiments.

    Inherits all YAML-parsing logic from :class:`QMIXConfigProcessor`.
    No extra fields beyond the standard off-policy set are needed.
    """

    pass
