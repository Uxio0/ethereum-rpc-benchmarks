import os
import time
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

from gnosis.eth import EthereumClient


class Timer:
    start: float = 0.0
    elapsed: float = 0.0
    node_versions: List[str] = []
    benchmarks: Dict[str, Any] = {}

    def __init__(self, test_name: str):
        self.test_name = test_name

    def __enter__(self):
        print(self.test_name)
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.time() - self.start
        self.benchmarks.setdefault(self.test_name, []).append(self.elapsed)
        print(self.test_name, ":", self.elapsed, "seconds")

    @classmethod
    def plot(cls, node_versions=None, benchmarks=None):
        node_versions = node_versions or cls.node_versions
        benchmarks = benchmarks or cls.benchmarks

        bars1 = []
        bars2 = []
        test_names = []
        for test_name, values in benchmarks.items():
            bars1.append(values[0])
            bars2.append(values[1])
            test_names.append(test_name)

        x = np.arange(len(test_names))  # the label locations
        width = 0.25

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
        plt.savefig("plot.png")

    @classmethod
    def test_plot(cls):
        node_versions = [
            "OpenEthereum//v3.3.5-stable/x86_64-linux-musl/rustc1.59.0",
            "erigon/2022.04.2/linux-amd64/go1.17.9",
        ]
        benchmarks = {
            "Txs in last block": [0.281780481338501, 0.16505098342895508],
            "Tx traces in last block": [0.19403791427612305, 3.491976499557495],
            "Txs in last year block": [0.6601812839508057, 0.3808917999267578],
            "Tx traces in last year block": [0.8607962131500244, 16.695952653884888],
            "Erc20 transfers in last 500 blocks": [
                40.88732099533081,
                38.98498845100403,
            ],
            "Traces for Gnosis Safe v1.1.1 last 200 blocks (30 minutes)": [
                2.587017774581909,
                29.74629807472229,
            ],
            "Traces for Gnosis Safe v1.1.1 first 100,000 blocks (17 days)": [
                19.528945446014404,
                0.7050864696502686,
            ],
        }
        cls.plot(node_versions=node_versions, benchmarks=benchmarks)


if __name__ == "__main__":
    # Timer.test_plot()
    load_dotenv()
    nodes = [
        os.getenv("ETHEREUM_NODE_1"),
        os.getenv("ETHEREUM_NODE_2"),
    ]

    ethereum_clients = [
        EthereumClient(node, provider_timeout=300, slow_provider_timeout=300)
        for node in nodes
    ]

    # Use always same block number for all the benchmarks
    last_block_number: Optional[int] = None
    for i, ethereum_client in enumerate(ethereum_clients):
        Timer.node_versions.append(ethereum_client.w3.clientVersion)
        print(Timer.node_versions[-1])
        last_block = ethereum_client.get_block(
            last_block_number or "latest", full_transactions=False
        )
        last_block_number = last_block["number"]
        with Timer("Txs in last block") as timer:
            ethereum_client.get_transactions(last_block["transactions"])

        with Timer("Tx traces in last block") as timer:
            ethereum_client.parity.trace_transactions(last_block["transactions"])

        old_block = ethereum_client.get_block(
            last_block["number"] - 2102400, full_transactions=False
        )
        with Timer("Txs in last year block") as timer:
            ethereum_client.get_transactions(old_block["transactions"])

        with Timer("Tx traces in last year block") as timer:
            ethereum_client.parity.trace_transactions(old_block["transactions"])

        with Timer("Erc20 transfers in last 500 blocks") as timer:
            ethereum_client.erc20.get_total_transfer_history(
                from_block=last_block["number"] - 500
            )

        with Timer(
            "Traces for Gnosis Safe v1.1.1 last 200 blocks (30 minutes)"
        ) as timer:
            # 0x34CfAC646f301356fAa8B21e94227e3583Fe3F5F was deployed on 9084503
            print(
                len(
                    ethereum_client.parity.trace_filter(
                        from_block=last_block["number"] - 200,
                        to_address=["0x34CfAC646f301356fAa8B21e94227e3583Fe3F5F"],
                    )
                )
            )

        with Timer(
            "Traces for Gnosis Safe v1.1.1 first 100,000 blocks (17 days)"
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

        with Timer(
            "Traces for Gnosis Safe v1.1.1 first 200,000 blocks (1 month)"
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
