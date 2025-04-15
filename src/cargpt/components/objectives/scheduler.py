from collections.abc import Iterable, Mapping, Sequence

import numpy as np
from pydantic import validate_call


class ObjectiveScheduler:
    @validate_call
    def __init__(self, *, schedule: Mapping[str, float], sample_size: int) -> None:
        super().__init__()

        objectives, probabilities = zip(*schedule.items(), strict=True)

        if sample_size not in (sample_size_range := range(1, len(objectives))):
            msg = f"sample_size must be in {sample_size_range}"
            raise ValueError(msg)

        if not np.isclose(sum(probabilities), total := 1.0):
            msg = f"probabilities must add up to {total}"
            raise ValueError(msg)

        self.objectives: Sequence[str] = objectives
        self.probabilities: Sequence[float] = probabilities
        self.sample_size: int = sample_size
        self.generator: np.random.Generator = np.random.Generator(np.random.PCG64())

    def sample(self) -> Iterable[str]:
        return self.generator.choice(
            a=self.objectives,
            p=self.probabilities,
            size=self.sample_size,
            replace=False,
        )
