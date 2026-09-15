"""Keep the historical data split independent of the optimizer step budget."""


def add_training_split_args(parser) -> None:
    parser.add_argument(
        "--train_split_batches", type=int, default=10,
        help=("Number of leading batches reserved for training, independent of "
              "num_iterations. Default 10 preserves the original data split."),
    )


def fixed_split_schedule(
    dataset_size: int,
    batch_size: int,
    num_iterations: int,
    *,
    train_split_batches: int = 10,
    optimize_enabled: bool = True,
) -> list[tuple[int, bool]]:
    """Return (dataset batch index, train flag) in execution order.

    Training revisits only the fixed prefix. Evaluation visits the original
    suffix once, in order. Incomplete trailing batches remain excluded to keep
    the historical evaluation sample IDs exactly unchanged.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if num_iterations < 0:
        raise ValueError("num_iterations must be non-negative.")
    if train_split_batches <= 0:
        raise ValueError("train_split_batches must be positive.")
    num_batches = dataset_size // batch_size
    if train_split_batches >= num_batches:
        raise ValueError(
            "The fixed training prefix leaves no full evaluation batch. "
            "Provide more data or reduce train_split_batches."
        )
    training = (
        [(step % train_split_batches, True) for step in range(num_iterations)]
        if optimize_enabled else []
    )
    return training + [(idx, False) for idx in range(train_split_batches, num_batches)]
