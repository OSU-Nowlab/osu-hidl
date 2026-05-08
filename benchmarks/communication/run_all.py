#!/usr/bin/env python
"""
Run all communication benchmarks.

This script runs all available communication benchmarks with the specified parameters.
"""
import sys
from .utils import benchmark_parser, init_torch_distributed

# Import individual benchmark modules
from . import all_reduce
from . import broadcast
from . import all_gather
from . import gather
from . import scatter
from . import barrier


def main():
    parser = benchmark_parser()
    args = parser.parse_args()

    rank, world_size = init_torch_distributed(args.backend)

    # Determine which benchmarks to run
    run_all = not any([
        args.all_reduce, args.broadcast, args.all_gather, args.gather,
        args.scatter, args.barrier
    ])

    benchmarks = []

    if run_all or args.all_reduce:
        benchmarks.append(("All-Reduce", all_reduce))

    if run_all or args.broadcast:
        benchmarks.append(("Broadcast", broadcast))

    if run_all or args.all_gather:
        benchmarks.append(("All-Gather", all_gather))

    if run_all or args.gather:
        benchmarks.append(("Gather", gather))

    if run_all or args.scatter:
        benchmarks.append(("Scatter", scatter))

    if run_all or args.barrier:
        benchmarks.append(("Barrier", barrier))

    # Print summary
    if rank == 0:
        print("\n" + "=" * 60)
        print("Communication Benchmarks Suite")
        print("=" * 60)
        print(f"World Size: {world_size}")
        print(f"Backend: {args.backend}")
        print(f"Trials: {args.trials}")
        print(f"Warmups: {args.warmups}")
        if args.scan:
            print(f"Scanning message sizes up to 2^{args.maxsize}")
        print(f"\nRunning {len(benchmarks)} benchmark(s):")
        for name, _ in benchmarks:
            print(f"  - {name}")
        print("=" * 60 + "\n")

    # Run each benchmark
    for name, module in benchmarks:
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Starting: {name}")
            print("=" * 60)

        try:
            module.main()
        except Exception as e:
            if rank == 0:
                print(f"Error running {name}: {e}")
                import traceback
                traceback.print_exc()

        if rank == 0:
            print(f"\nCompleted: {name}")
            print("=" * 60)

    if rank == 0:
        print("\n" + "=" * 60)
        print("All benchmarks completed")
        print("=" * 60 + "\n")


if __name__ == '__main__':
    # Override sys.argv to remove the script name so individual benchmarks
    # can parse args correctly
    original_argv = sys.argv.copy()

    # Parse args first to get configuration
    parser = benchmark_parser()
    args = parser.parse_args()

    # Keep args for individual modules
    sys.argv = original_argv

    main()
