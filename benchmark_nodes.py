import json
import os
import time
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

from gnosis.eth import EthereumClient


class NodeTimer:
    def __init__(self, node_version: str):
        self.node_version = node_version
        print("Init NodeTimer for", node_version)

    def get_timer(self, test_name: str) -> "Timer":
        return Timer(self.node_version, test_name)


class Timer:
    start: float = 0.0
    elapsed: float = 0.0
    benchmarks: Dict[str, Dict[str, Any]] = {}

    def __init__(self, node_version: str, test_name: str):
        self.node_version = node_version
        self.test_name = test_name

    def __enter__(self):
        print(self.test_name, ":", "Starting")
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = self.get_elapsed()
        self.benchmarks.setdefault(self.node_version, {})[self.test_name] = self.elapsed
        print(self.test_name, ":", self.elapsed, "seconds")

    def get_elapsed(self):
        return time.time() - self.start

    @classmethod
    def plot(cls, benchmarks: Optional[Dict[str, Any]] = None):
        benchmarks = benchmarks or cls.benchmarks
        node_versions = list(benchmarks.keys())

        test_names = [
            test_name for test_name, _ in benchmarks[node_versions[0]].items()
        ]
        bars1 = [benchmarks[node_versions[0]][test_name] for test_name in test_names]
        bars2 = [benchmarks[node_versions[1]][test_name] for test_name in test_names]
        # TODO Support more bars than 2

        x = np.arange(len(test_names))  # the label locations
        width = 0.2

        # Make the plot
        fig, ax = plt.subplots()
        rects1 = ax.bar(
            x - width / 2,
            bars1,
            color="#7f6d5f",
            width=width,
            edgecolor="white",
            label=node_versions[0],
        )
        rects2 = ax.bar(
            x + width / 2,
            bars2,
            color="#557f2d",
            width=width,
            edgecolor="white",
            label=node_versions[1],
        )

        ax.set_ylabel("Time elapsed in seconds (less is better)")
        ax.set_title("Benchmark of Ethereum RPC")
        ax.set_xticks(x, test_names)
        ax.tick_params(axis="x", which="both", labelrotation=90)
        ax.legend()

        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)
        fig.subplots_adjust(bottom=0.35, top=0.96)
        fig.set_size_inches(12, 7)
        # plt.show()
        plt.savefig("plot.png")

    @classmethod
    def test_plot(cls):
        with open("plot.json", "r") as f:
            cls.plot(benchmarks=json.load(f))


def perform_benchmark():
    # Timer.test_plot()
    load_dotenv()
    nodes = os.getenv("ETHEREUM_NODES").split(",")

    ethereum_clients = [
        EthereumClient(node, provider_timeout=300, slow_provider_timeout=300)
        for node in nodes
    ]

    # Use always same block number for all the benchmarks
    current_block_number = ethereum_clients[0].current_block_number
    last_block_number = current_block_number - 20  # Use confirmed block
    print("Last block number:", last_block_number)
    for i, ethereum_client in enumerate(ethereum_clients):
        node_version = ethereum_client.w3.clientVersion
        node_timer = NodeTimer(node_version)
        last_block = ethereum_client.get_block(
            last_block_number, full_transactions=False
        )
        with node_timer.get_timer("Txs in last block") as timer:
            ethereum_client.get_transactions(last_block["transactions"])

        with node_timer.get_timer("Tx traces in last block") as timer:
            ethereum_client.parity.trace_transactions(last_block["transactions"])

        # Get block from last year
        old_block = ethereum_client.get_block(
            last_block["number"] - 2102400, full_transactions=False
        )
        with node_timer.get_timer("Txs in last year block") as timer:
            ethereum_client.get_transactions(old_block["transactions"])

        with node_timer.get_timer("Tx traces in last year block") as timer:
            ethereum_client.parity.trace_transactions(old_block["transactions"])

        with node_timer.get_timer(
            "Erc20 transfers in last 500 blocks"  # We don't know the number of events beforehand
        ) as timer:
            number_events = len(
                ethereum_client.erc20.get_total_transfer_history(
                    from_block=last_block_number - 500, to_block=last_block_number
                )
            )
            print(number_events)
            timer.test_name = (
                f"Erc20 transfers in last \n500 blocks - {number_events} events"
            )

        with node_timer.get_timer(
            "Traces for Gnosis Safe v1.1.1 last 200 blocks"  # We don't know the number of events before hand
        ) as timer:
            # 0x34CfAC646f301356fAa8B21e94227e3583Fe3F5F was deployed on 9084503
            number_traces = len(
                ethereum_client.parity.trace_filter(
                    from_block=last_block_number - 200,
                    to_block=last_block_number,
                    to_address=["0x34CfAC646f301356fAa8B21e94227e3583Fe3F5F"],
                )
            )
            print(number_traces)
            timer.test_name = f"Traces for Gnosis Safe v1.1.1 \nlast 200 blocks (30 minutes)\n{number_traces} traces"

        with node_timer.get_timer(
            "Traces for Gnosis Safe v1.1.1 \nfirst 100,000 blocks (17 days)\n5 traces"
        ) as timer:
            # 0x34CfAC646f301356fAa8B21e94227e3583Fe3F5F was deployed on 9084503
            from_block = 9084503
            print(
                len(
                    ethereum_client.parity.trace_filter(
                        from_block=from_block,
                        to_block=from_block + 100_000,
                        to_address=["0x34CfAC646f301356fAa8B21e94227e3583Fe3F5F"],
                    )
                )
            )

        with node_timer.get_timer(
            "Traces for Gnosis Safe v1.1.1 \nfirst 200,000 blocks (1 month)\n24 traces"
        ) as timer:
            # 0x34CfAC646f301356fAa8B21e94227e3583Fe3F5F was deployed on 9084503
            from_block = 9084503
            print(
                len(
                    ethereum_client.parity.trace_filter(
                        from_block=from_block,
                        to_block=from_block + 200_000,
                        to_address=["0x34CfAC646f301356fAa8B21e94227e3583Fe3F5F"],
                    )
                )
            )

        print()

    Timer.plot()

    with open("plot.json", "w") as f:
        json.dump(Timer.benchmarks, f, indent=4)


if __name__ == "__main__":
    perform_benchmark()
