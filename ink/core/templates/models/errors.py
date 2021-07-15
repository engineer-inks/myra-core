class BaseError(RuntimeError):
    """Models Base Error.

    Generic base error class, used for centrality purposes during capturing.
    This class should not be used directly, in opposite of one of its subclasses below.
    """


class Convergence(BaseError):
    """Converge Error.

    Raised whenever a model or training procedure has failed to converge.
    """


class DegradedScore(BaseError):
    """Degraded Score Error.

    Raised whenever a model's score degrades over a new dataset sample.
    It can be used to signal the necessity of a new training.
    """


class Training(BaseError):
    """Training Error.

    Raised whenever a model cannot be fit properly.
    """


class TrainingNotFound(Training):
    """Training Not Found Error.

    Raise whenever a model's is required, but it's trained weights cannot be found.
    """
