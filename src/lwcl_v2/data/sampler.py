from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterator

import numpy as np
from torch.utils.data import Sampler


class CrossSubjectBatchSampler(Sampler[list[int]]):
    """Build batches with same-gesture examples drawn from different subjects."""

    def __init__(
        self,
        labels: list[int],
        subjects: list[str],
        gestures_per_batch: int,
        subjects_per_gesture: int,
        samples_per_subject: int,
        seed: int = 2025,
        batches_per_epoch: int | None = None,
    ) -> None:
        self.seed = seed
        self.epoch = 0
        self.gestures_per_batch = gestures_per_batch
        self.subjects_per_gesture = subjects_per_gesture
        self.samples_per_subject = samples_per_subject
        self.index: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for index, (label, subject) in enumerate(zip(labels, subjects, strict=True)):
            self.index[int(label)][subject].append(index)
        self.labels = sorted(self.index)
        if gestures_per_batch > len(self.labels):
            raise ValueError("gestures_per_batch exceeds the number of labels")
        self.batch_size = gestures_per_batch * subjects_per_gesture * samples_per_subject
        self.batches_per_epoch = int(batches_per_epoch or math.ceil(len(labels) / self.batch_size))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.batches_per_epoch):
            labels = rng.choice(self.labels, self.gestures_per_batch, replace=False)
            batch: list[int] = []
            for label in labels:
                subject_index = self.index[int(label)]
                available_subjects = sorted(subject_index)
                selected_subjects = rng.choice(
                    available_subjects,
                    self.subjects_per_gesture,
                    replace=len(available_subjects) < self.subjects_per_gesture,
                )
                for subject in selected_subjects:
                    candidates = subject_index[str(subject)]
                    selected = rng.choice(
                        candidates,
                        self.samples_per_subject,
                        replace=len(candidates) < self.samples_per_subject,
                    )
                    batch.extend(int(index) for index in selected)
            rng.shuffle(batch)
            yield batch
