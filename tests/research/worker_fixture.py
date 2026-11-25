# Copyright (c) Microsoft. All rights reserved.

import time

from agentlightning.research.events import current_rollout, observed_call


class WaitingAgent:
    def run(self):
        with observed_call("tool", current_rollout(), tool_type="fixture_wait", request={"seconds": 60}):
            time.sleep(60)
