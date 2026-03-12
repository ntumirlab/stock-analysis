"""Single Stock Bayesian Executor (Oscar AND/OR).

This is a dedicated entrypoint for AND/OR parameter optimization.
"""

from tests.oscar_tw_strategy.single_stock_bayesian import run_cli


if __name__ == "__main__":
    run_cli(
        default_output_dir="assets/OscarTWStrategy/single_stock_bayesian_andor",
        default_optimize_composite=False,
        description="Single Stock Bayesian Executor (AND/OR)",
    )
