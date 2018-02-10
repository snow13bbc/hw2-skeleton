import sys
from .io import read_active_sites, write_clustering, write_mult_clusterings
from .cluster import cluster_by_partitioning, cluster_hierarchically, similarity_matrix, quality_score, comparison_score, compare

# Some quick stuff to make sure the program is called correctly
if len(sys.argv) < 4:
    print("Usage: python -m hw2skeleton [-P| -H| -C] <pdb directory> <output file>")
    sys.exit(0)

active_sites = read_active_sites(sys.argv[2])

# Choose clustering algorithm
if sys.argv[1][0:2] == '-P':
    print("Clustering using Partitioning method")
    sim_matrix = similarity_matrix(active_sites)
    clustering = cluster_by_partitioning(active_sites, sim_matrix)
    write_clustering(sys.argv[3], clustering)

if sys.argv[1][0:2] == '-H':
    print("Clustering using hierarchical method")
    sim_matrix = similarity_matrix(active_sites)
    clustering = cluster_hierarchically(active_sites, sim_matrix)
    write_clustering(sys.argv[3], clustering)

if sys.argv[1][0:2] == '-C':
    print("Comparing hierarchical and partioning methods")
    sim_matrix = similarity_matrix(active_sites)
    hierarchical_clustering = cluster_hierarchically(active_sites, sim_matrix)
    partioning_clustering = cluster_by_partitioning(active_sites, sim_matrix)
    compare(hierarchical_clustering, partioning_clustering, active_sites, sim_matrix)
