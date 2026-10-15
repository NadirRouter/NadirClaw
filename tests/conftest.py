import os

# Never send usage telemetry from the test suite.
os.environ["NADIRCLAW_TELEMETRY"] = "0"
