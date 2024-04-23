from collections.abc import Iterable, Mapping

import numpy as np

from cargpt.components.objectives.common import ObjectiveName


class ObjectiveScheduler:
    def __init__(self, *, schedule: Mapping[str, float], sample_size: int):
        super().__init__()

        objectives, probabilities = zip(*schedule.items())
        objectives = tuple(map(ObjectiveName, objectives))

        if sample_size not in (sample_size_range := range(1, len(objectives))):
            msg = f"sample_size must be in {sample_size_range}"
            raise ValueError(msg)

        if not np.isclose(sum(probabilities), total := 1.0):
            msg = f"probabilities must add up to {total}"
            raise ValueError(msg)

        self.objectives = objectives
        self.probabilities = probabilities
        self.sample_size = sample_size
        self.generator = np.random.Generator(np.random.PCG64())

    def sample(self) -> Iterable[ObjectiveName]:
        objectives = self.generator.choice(  # pyright: ignore[reportCallIssue]
            a=self.objectives,  # pyright: ignore[reportArgumentType]
            p=self.probabilities,
            size=self.sample_size,
            replace=False,
        )

        return map(ObjectiveName, objectives)  # pyright: ignore[reportArgumentType]
