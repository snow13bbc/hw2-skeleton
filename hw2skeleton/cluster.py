from .utils import Atom, Residue, ActiveSite
import random
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import os
import matplotlib.pyplot as plt
import math

def compute_similarity(site_a, site_b):
    """
    Compute the similarity between two given ActiveSite instances.

    Input: two ActiveSite instances
    Output: the similarity between them (a floating point number)

    I am using Euclidean distance between different amino acids
    categories in each active site as similarity metric.

    The categories of amino acids have been defined in io.py.

    """

    # First, get the differences between active sites in terms of residue categories.
    # Then, calculate Euclidean distance.
    differences = site_a.categories - site_b.categories
    similarity = (sum(differences**2))**0.5
    return similarity

 ##--------------Functions for Partition Clustering--------------------------

def similarity_matrix(active_sites):
    """
    Give a similarity score between every pair of active sites.
    Similarity score of 1 is the best.

    Input: List of ActiveSites
    Output: Similarity Matrix
    """
    sim_matrix = {}
    maximum_score = 0.0
    for i in range(len(active_sites)):
        for j in range(i, len(active_sites)):
            site_a = active_sites[i]
            site_b = active_sites[j]
            similarity = compute_similarity(site_a, site_b)
            if similarity > maximum_score:
                maximum_score = similarity

            if site_a in sim_matrix:
                sim_matrix[site_a][site_b] = similarity
            else:
                sim_matrix[site_a] = {site_b : similarity}

            if site_b in sim_matrix:
                sim_matrix[site_b][site_a] = similarity
            else:
                sim_matrix[site_b] = {site_a : similarity}

    #Normalizing all the similarity scores by maximum score.
    for outer_matrix in sim_matrix:
        for inner_matrix in sim_matrix[outer_matrix]:
            current_sim = sim_matrix[outer_matrix][inner_matrix]
            sim_matrix[outer_matrix][inner_matrix] = 1.0 - (current_sim / maximum_score)

    return sim_matrix

def average_similarity(activesite, cluster, sim_matrix):
    """
    Give an average similarity score of the cluster.

    Input: ActiveSite
    Output: list of ActiveSites in cluster
    """

    similarity = 0.0
    for cluster_site in cluster:
        similarity += sim_matrix[activesite][cluster_site]

    similarity = similarity / float(len(cluster))
    return similarity

def cluster_center(clusters, sim_matrix):
    """
    Give active sites that score the best in similarity score in a cluster.
    i.e. Which active sites are the most representives of the cluster?

    Input: Clusters, a matrix of similarity scores

    Output: cluster_centers (A list of ActiveSite instances)
    """
    cluster_centers = []

    for cluster in clusters:
        center = cluster[0]
        best_similarity = average_similarity(cluster[0], cluster, sim_matrix)
        for active_site in cluster:
            similarity = average_similarity(active_site, cluster, sim_matrix)
            if similarity > best_similarity:
                best_similarity = similarity
                center = active_site
        cluster_centers.append(center)

    return cluster_centers

def k_means(k, active_sites, sim_matrix):
    """
    Calculate k_means for paritioning cluster.

    Input: k value, ActiveSites, sim_matrix

    Output: k clusters
    """

    active_sites_list = list(active_sites)

    clusters = []
    cluster_centers = []
    for i in range(k):
        cluster_centers.append(active_sites_list.pop())
        clusters.append([cluster_centers[i]])

    while(True):
        #make new empty clusters and add cluster centers to them
        new_clusters = []
        for i in range(k):
            new_clusters.append([])
        active_sites_list = list(active_sites)
        for i, center in enumerate(cluster_centers):
            active_sites_list.remove(center)
            new_clusters[i].append(center)
        for active_site in active_sites_list:
            highest_similarity_score = sim_matrix[active_site][cluster_centers[0]]
            best_cluster = 0
            for i in range(1, k):
                cluster = clusters[i]
                center_similarity = sim_matrix[active_site][cluster_centers[i]]
                if center_similarity > highest_similarity_score:
                    highest_similarity_score = center_similarity
                    best_cluster = i
            new_clusters[best_cluster].append(active_site)
        if new_clusters == clusters:
            break
        else:
            cluster_centers = cluster_center(new_clusters, sim_matrix)
            clusters = new_clusters

    return clusters

def best_k_value(active_sites, sim_matrix, plot = True):
    """
    Looking for best k value for k-means by iterating over all possible k k_values.
    Best k value is the lowest k value at which
    the intra-cluster similarity is minimized.

    Input: ActiveSites, sim_matrix
    Output: Best k value (integer)
    """

    cluster_similarity = []

    #Making the highest possible k value to the 4/5th of numbers of active sites.
    highest_k_value =  int(math.ceil(len(active_sites) * (4.0/5)))
    k_values = range(1, highest_k_value)

    for k in k_values:
        average_cluster_similarity = 0.0
        clustering = k_means(k, active_sites, sim_matrix)
        for cluster in clustering:
            cluster_sim = 0.0
            for active_site in cluster:
                cluster_sim += average_similarity(active_site, cluster, sim_matrix)
            cluster_sim = cluster_sim / float(len(cluster))
            average_cluster_similarity += cluster_sim

        average_cluster_similarity = average_cluster_similarity / float(k)
        cluster_similarity.append(average_cluster_similarity)


    max_cluster_similarity = np.amax(cluster_similarity)
    best_k = 1
    for k in k_values:
        if cluster_similarity[k - 1] >= max_cluster_similarity * 0.80:
            # This is a random number. Here, I am making it so that the best
            # k-value would give average intra-cluster similiarty to be
            # 80% of the maximum intra-cluster similarity.
            best_k_value = k
            break
    if plot == True:
        fig = plt.figure()
        plt.plot(k_values, cluster_similarity)
        plt.plot([best_k_value], [cluster_similarity[best_k_value - 1]], 'bo', linewidth=2, markersize=12)
        plt.xticks(np.arange(min(k_values), max(k_values)+1, 9.0))
        plt.xlabel('k values')
        plt.ylabel('Inter-Cluster Similarity')
        plt.title('Identifying best k value for Clustering by Partitioning')
        plt.savefig("K-Means Partition Clustering")
        plt.show()

    return best_k_value


 ##-----------------Functions for Hierarchical Clustering--------------------------

def poor_similarity(cluster1, cluster2, sim_matrix):
    """
    Gives the poorest similarity score between any pair of nodes in any two clusters.

    Input: 2 clusters, sim_matrix
    Output: a value for poor similairty score
    """
    #The poorest similairty score possible is 1.

    least_similarity = 1.0
    for site1 in cluster1:
        for site2 in cluster2:
            current_sim = sim_matrix[site1][site2]
            if current_sim < least_similarity:
                least_similarity = current_sim

    return least_similarity


def furthest_neighbor(active_sites, sim_matrix, similarity_cutoff):
    """
    Cluster active sites so that the best similairty score between
    the nodes of the clusters is less than the simialirty cutoff values.

    Input: active_sites, sim_matrix, a similairty cutoff value

    Output: clusters
    """
    current_clustering = []
    for site in active_sites:
        current_clustering.append([site])

    while True:
        best_sim = 0.0
        join_index_1 = 0
        join_index_2 = 1

        for i in range(len(current_clustering)):
            current_cluster = current_clustering[i]
            for j in range(i + 1, len(current_clustering)):
                comparison_cluster = current_clustering[j]
                worst_sim = poor_similarity(current_cluster, comparison_cluster, sim_matrix)
                if worst_sim > best_sim:
                    best_sim = worst_sim
                    join_index_1 = i
                    join_index_2 = j

        if best_sim <= similarity_cutoff:
            break
        # Initialize a new clustering
        new_clustering = []
        # Combine the two clusters with the best furthest-neighbor similarity score
        joined_cluster = current_clustering[join_index_1] + current_clustering[join_index_2]
        # Add all other clusters to the new clustering
        for cluster_index, current_cluster in enumerate(current_clustering):
            if (cluster_index != join_index_1 and cluster_index != join_index_2):
                new_clustering.append(current_cluster)
        # Add the new joined cluster
        new_clustering.append(joined_cluster)
        # clusterings.append(current_clustering)
        current_clustering = new_clustering

    return current_clustering


def get_similarity_cutoff(active_sites, sim_matrix, plot = True):
    """
    Get a similarity cutoff value for hierarchical clustering.

    Input: ActiveSites, similarity_matrix
    Output: a value for similarity cutoff
    """
    sim_cutoffs = np.linspace(0.0,1, 30)
    clustering_sims = []

    # Compare sim_cutoff values by building hierarchical clusters and
    # calculating intra-cluster similarity.
    for current_sim_cutoff in sim_cutoffs:
        avg_cluster_sim = 0.0
        clustering = furthest_neighbor(active_sites, sim_matrix, current_sim_cutoff)
        for cluster in clustering:
            cluster_sim = 0.0
            for active_site in cluster:
                cluster_sim += average_similarity(active_site, cluster, sim_matrix)
            cluster_sim = cluster_sim / float(len(cluster))
            avg_cluster_sim += cluster_sim

        avg_cluster_sim = avg_cluster_sim / len(clustering)
        clustering_sims.append(avg_cluster_sim)



    best_sim_cutoff = 0.0
    best_cluster_sim = 0.0
    best_avg_diff = 0.0
    clustering_sims_array = np.asarray(clustering_sims)

    for index, clustering_sim in enumerate(clustering_sims):
        if index == 0 or index == (len(clustering_sims) - 1):
            continue

        prev_point = clustering_sims[index - 1]
        next_point = clustering_sims[index + 1]
        if (clustering_sim > prev_point) and (clustering_sim > next_point):
            avg_diff = ((clustering_sim - prev_point) + (clustering_sim - next_point)) / 2.0
            if avg_diff > best_avg_diff:
                best_avg_diff = avg_diff
                best_sim_cutoff = sim_cutoffs[index]
                best_cluster_sim = clustering_sim


    if plot == True:
        fig = plt.figure()
        plt.plot(sim_cutoffs, clustering_sims)
        plt.plot(best_sim_cutoff, best_cluster_sim, 'bo', linewidth=2, markersize=12)
        plt.xlabel('Similarity cutoff')
        plt.ylabel('Inter-Cluster Similarity')
        plt.title('Similarity Cutoffs for Hierarchical clustering')
        plt.savefig("Similarity Cutoffs")
        plt.show()

    return best_sim_cutoff

##---------------Clusters-----------------------------------------------------
def cluster_by_partitioning(active_sites, sim_matrix):
    """
    Cluster a given set of ActiveSite instances using a partitioning method.

    Input: a list of ActiveSite instances, k (int)
    Output: a clustering of ActiveSite instances
            (this is really a list of clusters, each of which is list of
            ActiveSite instances)
    """

    if len(active_sites) <= 1:
        clusters = []
        for site in active_sites:
            clusters.append([site])
        return clusters
    # For determining good k value to use
    best_k = best_k_value(active_sites, sim_matrix)
    # Use k_means implementation above to get clusters from partitioning
    clusters = k_means(best_k, active_sites, sim_matrix)
    return clusters

def cluster_hierarchically(active_sites, sim_matrix):
    """
    Cluster the given set of ActiveSite instances using a hierarchical algorithm.                                                                  #

    Input: a list of ActiveSite instances
    Output: a list of clusterings
            (each clustering is a list of lists of Sequence objects)
    """

    if len(active_sites) <= 1:
        clusters = []
        for site in active_sites:
            clusters.append([site])
        return clusters

    # Determine a good similarity cutoff score to use as defined above
    sim_cutoff = get_similarity_cutoff(active_sites, sim_matrix)
    # Use furthest neighbor agglomerative clustering to get clusters from
    # hierarchical clustering
    clusterings = furthest_neighbor(active_sites, sim_matrix, sim_cutoff)
    return clusterings

 ##-----------------Cluster Quality--------------------------

def quality_score(clustering, active_sites, sim_matrix):
    """
    Using silhouette score as a clustering quality score.

    Input: List of lists of ActiveSites
    Output: Silhouette Score
    """

    s = 0.0

    for site in active_sites:
        for cluster in clustering:
            if site in cluster:
                break
# Compute average similarity of the ActiveSite with all other sites in its cluster.
        a = average_similarity(site, cluster, sim_matrix)
# Compute average similarity of the ActiveSite with all other sites in its closest cluster.
        b = 0.0
        for test_cluster in clustering:
            if test_cluster != cluster:
                test_sim = average_similarity(site, test_cluster, sim_matrix)
                if test_sim > b:
                    b = test_sim

        s += (a - b) / min(a, b)

    s = s / len(active_sites)
    return s

 ##-----------------Comparing Hierarchical and Partition Clustering--------------------------

def comparison_score(hierarchical_clustering, partioning_clustering, active_sites, sim_matrix):
    """
    Get a comparison score between hierarchical and partioning clustering
    using Silhouette score.

    Input: types of clustering methods, activesites, sim_matrix
    Output: comparison_score
    """

    # Get the two respective silhouette scores
    h_quality = quality_score(hierarchical_clustering, active_sites, sim_matrix)
    p_quality = quality_score(partioning_clustering, active_sites, sim_matrix)

    # Find their difference and normalize to [0,1]
    comparison_score = 1.0 - abs(h_quality - p_quality) / 2.0
    return comparison_score

def compare(hierarchical_clustering, partioning_clustering, active_sites, sim_matrix):
    """
    Given a hierarchical and partioning clustering, produce a plot that measures their
    respesctive qualities

    Intput: hierarchical_clustering, partioning_clustering (list of lists of ActiveSites)
            active_sites (list of ActiveSites)
            sim_matrix (dictionary of dictionaries)

    Output: None
    """

    partitioning_qualities = []
    partioning_clusterings = []
    top_limit =  int(math.ceil(len(active_sites) * (3.0/5)))
    k_vals = range(1, top_limit)

    hierarchical_qualities = []
    hierarchical_clusterings = []
    sim_cutoffs = np.linspace(0.0,1, len(k_vals))

    comparison_scores = []

    # Calculate partioning and hierarchical clusterings across range of k values and
    # similarity cutoffs. For each resulting clustering, calculate the quality scores
    for k in k_vals:
        partioning_clustering = k_means(k, active_sites, sim_matrix)
        partioning_clusterings.append(partioning_clustering)
        quality = quality_score(partioning_clustering, active_sites, sim_matrix)
        partitioning_qualities.append(quality)
    for current_sim_cutoff in sim_cutoffs:
        hierarchical_clustering = furthest_neighbor(active_sites, sim_matrix, current_sim_cutoff)
        hierarchical_clusterings.append(hierarchical_clustering)
        quality = quality_score(hierarchical_clustering, active_sites, sim_matrix)
        hierarchical_qualities.append(quality)

    # Calculate the comparison scores for clusterings produced at the same position in the
    # range [0,1] of input values (k value for partioning or similary cutoff score for hierarchical)
    for i in range(len(k_vals)):
        score = comparison_score(hierarchical_clusterings[i], partioning_clusterings[i], active_sites, sim_matrix)
        comparison_scores.append(score)

    # Plot the resulting scores
    k_vals_normalized = np.asarray(k_vals) / np.max(k_vals)
    fig = plt.figure()
    plt.plot(k_vals_normalized, partitioning_qualities, label = 'Partitioning Qualities')
    plt.plot(sim_cutoffs, hierarchical_qualities, label = 'Hierarchical Qualities')
    plt.plot(k_vals_normalized, comparison_scores, label = 'Comparison Scores')
    plt.xlabel('normalized k (Partitioning) and similarity cutoffs (Hierarchical)')
    plt.ylabel('Quality and Comparison Scores')
    plt.title('Quality and Comparison Scores for Partioning and Hierarchical Clustering')
    plt.legend()
    plt.savefig("Quality and Comparison of Partioning and Hierarchical Clustering")
    plt.show()
