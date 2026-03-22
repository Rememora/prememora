"""Full integration test: ingest text → MiniMax entity extraction → graph search."""

import sys
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

from adapter import Zep

GRAPH_ID = "test_integration"

TEST_EPISODES = [
    "Bitcoin surged past $95,000 on March 15, 2026, driven by institutional demand from BlackRock and Fidelity ETFs. Trading volume hit $48 billion in 24 hours.",
    "The Federal Reserve held interest rates steady at 4.25% amid mixed inflation signals. Chair Jerome Powell indicated potential cuts in Q3 2026.",
    "Polymarket saw record trading volume of $2.1 billion in February 2026, with the US presidential election market attracting the most liquidity.",
    "MicroStrategy announced an additional $500 million Bitcoin purchase, bringing their total holdings to 250,000 BTC. CEO Michael Saylor called it a generational opportunity.",
    "Ethereum's Pectra upgrade completed successfully, reducing gas fees by 40%. DeFi TVL rebounded to $180 billion across all chains.",
]


def main():
    client = Zep(api_key="unused")

    print("1. Creating graph...")
    client.graph.create(graph_id=GRAPH_ID, name="Integration Test", description="Testing full pipeline")
    print("   OK\n")

    print("2. Ingesting episodes via MiniMax → Graphiti...")
    for i, text in enumerate(TEST_EPISODES, 1):
        print(f"   [{i}/{len(TEST_EPISODES)}] {text[:60]}...")
        client.graph.add(graph_id=GRAPH_ID, type="text", data=text)
        print(f"   Done")
    print()

    print("3. Checking extracted entities...")
    nodes = client.graph.node.get_by_graph_id(GRAPH_ID)
    print(f"   Nodes: {len(nodes)}")
    for n in nodes[:10]:
        print(f"     - {n.name} ({', '.join(n.labels) if hasattr(n, 'labels') and n.labels else 'Entity'})")
    print()

    print("4. Checking extracted relationships...")
    edges = client.graph.edge.get_by_graph_id(GRAPH_ID)
    print(f"   Edges: {len(edges)}")
    for e in edges[:10]:
        print(f"     - {e.name}: {e.fact[:80]}..." if len(e.fact) > 80 else f"     - {e.name}: {e.fact}")
    print()

    print("5. Searching graph: 'Bitcoin price'...")
    results = client.graph.search(graph_id=GRAPH_ID, query="Bitcoin price", limit=5)
    print(f"   Results: {len(results.edges)}")
    for e in results.edges:
        print(f"     - {e.fact[:100]}")
    print()

    print("6. Searching graph: 'Federal Reserve interest rates'...")
    results = client.graph.search(graph_id=GRAPH_ID, query="Federal Reserve interest rates", limit=5)
    print(f"   Results: {len(results.edges)}")
    for e in results.edges:
        print(f"     - {e.fact[:100]}")
    print()

    print("7. Cleaning up...")
    client.graph.delete(graph_id=GRAPH_ID)
    print("   OK\n")

    print("Integration test complete!")


if __name__ == "__main__":
    main()
