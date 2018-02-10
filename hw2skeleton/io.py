import glob
import os
import numpy as np
from .utils import Atom, Residue, ActiveSite


def read_active_sites(dir):
    """
    Read in all of the active sites from the given directory.

    Input: directory
    Output: list of ActiveSite instances
    """
    files = glob.glob(dir + '/*.pdb')

    active_sites = []
    # iterate over each .pdb file in the given directory
    for filepath in glob.iglob(os.path.join(dir, "*.pdb")):

        active_sites.append(read_active_site(filepath))

    print("Read in %d active sites"%len(active_sites))

    return active_sites


def read_active_site(filepath):
    """
    Read in a single active site given a PDB file

    Input: PDB file path
    Output: ActiveSite instance
    """
    basename = os.path.basename(filepath)
    name = os.path.splitext(basename)

    if name[1] != ".pdb":
        raise IOError("%s is not a PDB file"%filepath)

    active_site = ActiveSite(name[0])

    r_num = 0

    # open pdb file
    with open(filepath, "r") as f:
        # iterate over each line in the file
        for line in f:
            if line[0:3] != 'TER':
                # read in an atom
                atom_type = line[13:17].strip()
                x_coord = float(line[30:38])
                y_coord = float(line[38:46])
                z_coord = float(line[46:54])
                atom = Atom(atom_type)
                atom.coords = (x_coord, y_coord, z_coord)

                residue_type = line[17:20]
                residue_number = int(line[23:26])

                # make a new residue if needed
                if residue_number != r_num:
                    residue = Residue(residue_type, residue_number)
                    r_num = residue_number

                # add the atom to the residue
                residue.atoms.append(atom)

            else:  # I've reached a TER card
                active_site.residues.append(residue)
                """
                Categorizing the types of amino acid residues by their chemical properties.
                Residue categories = Acidic, Basic, hydrophobic, Non-Polar, Polar.
                Counting the number of amino acid residues in an active site in a dictionary.
                """
        AA_residues = {}
        aminoacids = ['ASP','GLU','HIS','ARG','LYS','SER','THR',
                    'ASN','GLN','CYS','PHE','TRP','TYR','GLY',
                    'ALA','VAL','LEU','ILE','MET','PRO']
#Initialize the dictionary by listing all amino acids as keys first.
        for AA in aminoacids:
            AA_residues[AA] = 0
#Counting the occurance of amino acids in actve site as a value to the key.
        for res in active_site.residues:
            if res.type not in AA_residues:
                AA_residues[res.type] = 1
            else:
                AA_residues[res.type] += 1
        total_residues = sum(AA_residues.values()) # total residues in active site
#Categorizing the residues.
    residue_cat = np.array([0.0]*5)
    residue_cat[0] = AA_residues['ASP'] + AA_residues['GLU'] #'acidic'
    residue_cat[1] = AA_residues['HIS'] + AA_residues['ARG'] + AA_residues['LYS'] #'basic'
    residue_cat[2] = AA_residues['PHE'] + AA_residues['TRP'] + AA_residues['TYR'] #'hydrophobic'
    residue_cat[3] = AA_residues['GLY'] + AA_residues['ALA'] + AA_residues['VAL'] + AA_residues['LEU'] + AA_residues['ILE'] + AA_residues['MET'] + AA_residues['PRO'] #'nonpolar'
    residue_cat[4] = AA_residues['SER'] + AA_residues['THR'] + AA_residues['ASN'] + AA_residues['GLN'] + AA_residues['CYS'] #'polar'
#Calculating each residue categories observed in active site as a fraction.
    residue_cat = residue_cat/total_residues

    active_site.categories = residue_cat # add categories to the active site
    return active_site


def write_clustering(filename, clusters):
    """
    Write the clustered ActiveSite instances out to a file.

    Input: a filename and a clustering of ActiveSite instances
    Output: none
    """

    out = open(filename, 'w')

    for i in range(len(clusters)):
        out.write("\nCluster %d\n--------------\n" % i)
        for j in range(len(clusters[i])):
            out.write("%s\n" % clusters[i][j])

    out.close()


def write_mult_clusterings(filename, clusterings):
    """
    Write a series of clusterings of ActiveSite instances out to a file.
    Input: a filename and a list of clusterings of ActiveSite instances
    Output: none
    """

    out = open(filename, 'w')

    for i in range(len(clusterings)):
        clusters = clusterings[i]

        for j in range(len(clusters)):
            out.write("\nCluster %d\n------------\n" % j)
            for k in range(len(clusters[j])):
                out.write("%s\n" % clusters[j][k])

    out.close()
