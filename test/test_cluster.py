from hw2skeleton import cluster
from hw2skeleton import io
import os

def test_similarity():
    filename_a = os.path.join("data", "276.pdb")
    filename_b = os.path.join("data", "1806.pdb")

    activesite_a = io.read_active_site(filename_a)
    activesite_b = io.read_active_site(filename_b)

    # update this assertion
    assert cluster.compute_similarity(activesite_a, activesite_b) == 0.0

def test_partition_clustering():
    # tractable subset
    pdb_ids = [276, 39299, 38031]

    active_sites = []
    for id in pdb_ids:
        filepath = os.path.join("data", "%i.pdb"%id)
        active_sites.append(io.read_active_site(filepath))
    sim_matrix = cluster.similarity_matrix(active_sites)

    assert cluster.cluster_by_partitioning([], {}) == []
    assert cluster.cluster_by_partitioning([active_sites[0]], {}) == [[active_sites[0]]]



def test_hierarchical_clustering():
    # tractable subset
    pdb_ids = [276, 52954, 34088]

    active_sites = []
    for id in pdb_ids:
        filepath = os.path.join("data", "%i.pdb"%id)
        active_sites.append(io.read_active_site(filepath))

    sim_matrix = cluster.similarity_matrix(active_sites)

    # update this assertion
    assert cluster.cluster_hierarchically([], {}) == []
    assert cluster.cluster_hierarchically([active_sites[0]], {}) == [[active_sites[0]]]
