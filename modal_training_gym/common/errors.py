class TrainingGymError(ValueError):
    pass


class TrainingGymConfigError(TrainingGymError):
    pass


class GpuAllocationError(TrainingGymConfigError):
    pass
