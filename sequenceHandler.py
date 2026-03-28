import numpy as np
from scipy.spatial.distance import cdist
import Bio.PDB
from Bio.PDB import PDBParser, Polypeptide
from Bio import SeqIO, Align
import os
import subprocess
import requests



def mapPDBToSequence(pdbFile, chainId, sequence1, seq1Mapping, mapFile):
    """ Map a PDB chain to a reference sequence and compute residue distance map.
        Only standard amino-acid residues (ATOM records) are used; HETATM/ions are skipped.
    """

    # parse structure (suppress warnings)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdbFile)
    model = structure[0]

    # find the chain
    if chainId not in [c.id for c in model]:
        raise ValueError(f"Chain {chainId} not found in PDB")
    chain = model[chainId]

    # keep only standard amino-acids (this excludes HETATM/ions/waters)
    residues_filtered = []
    seq_letters = []
    for res in chain:
        if Polypeptide.is_aa(res, standard=True):
            # convert three-letter name to one-letter safely
            try:
                aa = Polypeptide.protein_letters_3to1[res.get_resname()]     #if we use a newer version on biophyton
                #aa = Polypeptide.three_to_one(res.get_resname())
            except Exception:
                # skip non-standard / unusual residue names if any
                continue
            residues_filtered.append(res)
            seq_letters.append(aa)

    if len(residues_filtered) == 0:
        raise ValueError("No standard amino-acid residues found on chain " + chainId)

    pdb_seq = ''.join(seq_letters)

    # compute distance map (minimal atom-atom distance between residues)
    n = len(residues_filtered)
    distanceMap = np.zeros((n, n), dtype=float)
    for i, res1 in enumerate(residues_filtered):
        atoms1 = np.array([a.get_coord() for a in res1.get_atoms()])
        for j, res2 in enumerate(residues_filtered):
            atoms2 = np.array([a.get_coord() for a in res2.get_atoms()])
            if atoms1.size == 0 or atoms2.size == 0:
                distanceMap[i, j] = np.inf
            else:
                distanceMap[i, j] = cdist(atoms1, atoms2).min()

    # Align PDB sequence to reference FASTA sequence
    ref_record = list(SeqIO.parse(sequence1, 'fasta'))[0]
    ref_seq = ref_record.seq

    aligner = Align.PairwiseAligner()
    # Using a simple global alignment here (adjust aligner params if needed)
    alignment = aligner.align(ref_seq, pdb_seq)[0]  # best alignment
    # alignment object has .aligned or .sequence depending on Biopython version.
    # You may need to extract mapping indexes with your getAlignmentIndexes function.
    # ali = alignment; ... (keep your existing alignment-to-index code here)

    # load seq1 mapping if provided
    if seq1Mapping != "None":
        with open(seq1Mapping, 'r') as f:
            refMapping = [int(l.strip()) for l in f if l.strip()]
    else:
        refMapping = np.arange(len(ref_seq))

    # Save output map if requested
    if mapFile != 'None':
        np.savetxt(mapFile, distanceMap, fmt='%.2f')

    return distanceMap, pdb_seq, residues_filtered  




def win_to_wsl_path(path):
    import os
    path = os.path.abspath(path)

    # normalize slashes
    path = path.replace("\\", "/")

    # convert drive letter
    if path[1:3] == ":/":
        drive = path[0].lower()
        path = f"/mnt/{drive}" + path[2:]

    return path




def alignSequenceToHMM(sequence,hmmFile):
    """ Aligns a sequence to a hmmer-generated HMM.

    Keyword Arguments:
        sequence (str): The sequence to be alignd as a simple string.

        hmmFile (str): The family hmm file.


    Returns:
        mapIndexes (ndarray): 1-D array containing the mapping indexes between the sequence and the hmmFile. 
                              The array is such that mapIndexes[i] maps sequence[i] to its position in the hmm.
    """
    
    # Align the pdbSequence to the HMM
    Bio.SeqIO.write(Bio.SeqRecord.SeqRecord(Bio.Seq.Seq(sequence),'tmp','tmp'),'__tmpFasta.fasta','fasta')
    out=subprocess.check_output(["hmmalign",hmmFile,'__tmpFasta.fasta']).splitlines()
    os.remove('__tmpFasta.fasta')
    rawMap=''
    gapMap=''
    for l in out:
        if l.startswith(b'#=GC RF'):
            rawMap+=str(l.split()[2])
        if l.startswith(b'#=GC PP_cons'):
            gapMap+=str(l.split()[2])

    mapIndexes=[]
    aaCounter=0
    for idx,c in enumerate(rawMap):
        if c == "x" and gapMap[idx] != '.' :
            mapIndexes.append(aaCounter)
            aaCounter+=1
        elif c=="x" and gapMap[idx] == '.':
            mapIndexes.append(-1)
        elif c==".":
            aaCounter+=1

    return np.asarray(mapIndexes)






def mapPDBToHMM(pdbFile, chainIds, hmmFile1, hmmFile2,mapFile,distType='all'):
    """ Maps a PDB structure to a hmm to compare DCA predictions with structural contacts. 
        This can be used either to only compute the PDB distance map (without hmmFile1='None' and hmmFile2='None') 
        or used to futher map the distance map such to aligned it to the hmm models.

    Keyword arguments:
        pdbFile (str)  : The structure file to map in PDB format 
    
        chainIds (str) : The chain Id(s) of the chain(s) to be mapped. For multiple chains, pass a unique string (e.g. AB)

        hmmFile1 (str) : Optional hmm file onto which the pdb map is aligned.
 
        hmmFile2 (str) : An optional second hmm file for mapping of hetero-dimers. 

        mapFile (str)  : The output name for the mapped distance map

       distType (str)  : Type of distance to compute: 'all': minimal distance between any atoms, 'alpha': Carbon-alpha distance, 'beta': Carbon-beta distance
    Return:
        map (ndarray) : A NxN array containing the (possibly mapped) distance map.
    """

    # Parse the command line hmms
    hmmFiles=[]
    if(hmmFile1 != 'None'):
        hmmFiles.append(hmmFile1)
    if(len(chainIds)==2 and hmmFile1!='None' and hmmFile2 != 'None'):
        hmmFiles.append(hmmFile2)
    elif(len(chainIds)==2 and hmmFile1!='None' and hmmFile2=='None'):
        hmmFiles.append(hmmFile1)

    # Parse the PDB and extract the residues of the target chains
    chainIds=list(chainIds)
    structure = Bio.PDB.PDBParser().get_structure('void', pdbFile)
    
    chains = structure[0].get_chains()

    residues=[chain.get_list() for chain in chains if chain.id in chainIds]
    residues = [res for resList in residues for res in resList]

    # Remove waters and hetero_residues
    #residues = [res for res in residues if (res.id[0][0] not in ["W", "H"])]

    filtered_residues = []
    filtered_seq = []

    for res in residues:
        if Polypeptide.is_aa(res, standard=True):
            try:
                aa = Polypeptide.protein_letters_3to1[res.get_resname()]
            except Exception:
                continue
            filtered_residues.append(res)
            filtered_seq.append(aa)

    residues = filtered_residues
    #print(residues)
    catPdbSeq = ''.join(filtered_seq)

    # Compute the distance map between all residues (minimal distance between atoms of the chain)
    distanceMap=np.zeros([len(residues),len(residues)],dtype=float)
    #catPdbSeq=''
    #from Bio.SCOP.Raf import protein_letters_3to1

    for idx1,res1 in enumerate(residues):
        if distType=='all':
            atoms1 = np.array([a.get_coord() for a in res1.get_atoms()])
        elif distType=='alpha':
            atoms1 = np.asarray([a.get_coord() for a in res1.get_atoms() if a.name=='CA'])
        elif distType=='beta':
            if res1.resname=='GLY':
                atoms1 = np.asarray([a.get_coord() for a in res1.get_atoms() if a.name=='CA'])
            else:
                atoms1 = np.asarray([a.get_coord() for a in res1.get_atoms() if a.name=='CB'])

        #catPdbSeq+=Polypeptide.three_to_one(res1.resname)
        #catPdbSeq+=Polypeptide.protein_letters_3to1[res1.resname]
        for idx2,res2 in enumerate(residues):
            if distType=='all':
                atoms2 = np.array([a.get_coord() for a in res2.get_atoms()])
            elif distType=='alpha':
                atoms2 = np.asarray([a.get_coord() for a in res2.get_atoms() if a.name=='CA'])
            elif distType=='beta':
                if res2.resname=='GLY':
                    atoms2 = np.asarray([a.get_coord() for a in res2.get_atoms() if a.name=='CA'])
                else:
                    atoms2 = np.asarray([a.get_coord() for a in res2.get_atoms() if a.name=='CB'])
            #atomDist = scipy.spatial.distance.cdist(atoms1,atoms2)
            #distanceMap[idx1,idx2]=atomDist.min()
            if len(atoms1) == 0 or len(atoms2) == 0:
                distanceMap[idx1, idx2] = np.inf
            else:
                distanceMap[idx1, idx2] = cdist(atoms1, atoms2).min()
    
    # Align the pdb sequences to the HMMs
    if hmmFiles:
        pdbSeqs=[]
        currentLength=0
        for chainId in chainIds:
                if chainId in structure[0]:
                    chain_res = [res for res in structure[0][chainId] if Polypeptide.is_aa(res, standard=True)]
                else:
                    print(f"Chain {chainId} not found in structure")
                    available_chains = [c.id for c in structure[0].get_chains()]
                    print(f"Available chains: {available_chains}")
                    print(f"Mapping chain {available_chains[0]} instead of {chainId}")
                    print("CHECK THE MATCH WITH THE HMM SEQUENCE")
                    chain_res = [res for res in structure[0][available_chains[0]] if Polypeptide.is_aa(res, standard=True)]
                d = len(chain_res)
                pdbSeqs.append(catPdbSeq[currentLength:currentLength + d])
                currentLength += d

            # for chain in structure[0].get_chains():
            #     if chain.id==chainId:
            #         d=len([res for res in chain.get_list() if res.id[0][0] not in ["W","H"]])
            #         pdbSeqs.append(catPdbSeq[currentLength:(currentLength+d)])
            #         currentLength+=d

        #print(pdbSeqs)
        mapIndexes=np.array([],dtype=int)
        offset=0
        for i in range(len(chainIds)):
            mapIndexes=np.concatenate([mapIndexes,offset+np.array(alignSequenceToHMM(pdbSeqs[i],hmmFiles[i]))])
            offset+=len(pdbSeqs[i])
        
        # Align the PDB distance map onto the HMM
        mi=mapIndexes
        dm=np.zeros([len(mi), len(mi)])
        dm[:]=np.inf
        dm[np.ix_(mi>=0,mi>=0)]=distanceMap[np.ix_(mi[mi>=0],mi[mi>=0])]  
    else:
        dm=distanceMap

    # For homo-dimeric mappings, reduce the distance map to a NxN map containing also the homo-dimeric contacts.
    if(len(chainIds)==2 and hmmFile1!='None' and hmmFile2=='None'):
        dm=np.minimum(np.minimum(np.minimum(dm[0:(len(dm)//2),0:(len(dm)//2)],dm[(len(dm)//2):,0:(len(dm)//2)]),dm[(len(dm)//2):,(len(dm)//2):].T),dm[0:(len(dm)//2),(len(dm)//2):].T)

    # Save output map
    if mapFile!='None':
        np.savetxt(mapFile,dm,fmt='%.2f')

    aligned_sequence=[]
    for element in mi:
        if int(element) == 0:
            raise RuntimeError("There is a problem on the mapping between pdb and hmm")
        elif int(element) == -1:
            aligned_sequence.append("-")
        else:
            aligned_sequence.append(pdbSeqs[0][int(element)-1])

    return dm,mi,aligned_sequence




def stockholm2fasta(stoFile,fastaFile,noFilterInserts=False):
    """ Converts an MSA in stockholm format to fasta format.

    Keyword Arguments:
        stoFile (str): The MSA file in stockholm format.

        fastaFile (str): The output fasta MSA filename.

        noFilterInserts(bool): If true, do not filter inserts (lower case and .)
    """

    # Extract all sequences from the Stockholm file
    seqs={}
    for line in open(stoFile,'r'):
        if line[0] not in ['#','/','\n']:
            seq=line.rstrip().split()
            if seq[0] in seqs:
                seqs[seq[0]]+=seq[1]
            else:
                seqs[seq[0]]=seq[1]

    # Build fasta MSA, removing inserts and delete symbols
    msa=[]
    for seqId,seq in seqs.items():
        # Filter inserts: delete '.' and lower case symbols
        if not noFilterInserts:
            seq=seq.replace('.','')
            seq=''.join([s for s in seq if not s.islower()])
        s=Bio.SeqRecord.SeqRecord(Bio.Seq.Seq(seq),id=seqId,description='')
        msa.append(s)
    Bio.SeqIO.write(msa,fastaFile,'fasta')


def filterSequenceByGapContent(inMSA,gapThreshold,filteredMSA,verbose=True):
    """ Filter MSA based on gap content.

    Keep only sequences having a maximum allowed gap fraction.
    Input sequences must be in fasta format.

    Keyword Arguments:
        inMSA (str): Input MSA file in fasta format.

        gapThreshold (float): Maximum allowed fraction of gaps. 

        filteredMSA (str): Output name for the filtered MSA.
    """
    
    sequences=Bio.SeqIO.parse(inMSA,"fasta")
    seq1=next(sequences)

    
    realLength=float(len(seq1.seq)-seq1.seq.count(".")-len([c for c in str(seq1.seq) if c.islower()]))
    sequences=list(Bio.SeqIO.parse(inMSA,"fasta"))
    filteredSequences=[seq for seq in sequences if float(seq.seq.count("-"))/realLength <= float(gapThreshold)]
    Bio.SeqIO.write(filteredSequences,filteredMSA,"fasta")
    if verbose:
        print("Original number of sequences ",len(sequences))
        print("Sequences after filtering : ",len(filteredSequences))
        print("Filtered sequences saved to ",filteredMSA)



def get_pfam_msa(pfam_acc: str, kind: str = "seed") -> bytes:
    """
    kind: 'seed' (curato, ~decine/centinaia di seq)
    'full' (tutte le sequenze UniProt che matchano l'HMM, può essere enorme)
    """
    url = (
        f"https://www.ebi.ac.uk/interpro/wwwapi//"
        f"entry/pfam/{pfam_acc}/?annotation=alignment:{kind}"
    )
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return resp.content  # formato Stockholm (.sto)


def do_DCA(msa_fasta):
    julia_code = f"""
    using PlmDCA, DelimitedFiles
    X = plmdca("{msa_fasta}")
    writedlm("scores.csv", X.score, ',')
    """

    # subprocess.run(["julia", "-e", julia_code])

    result = subprocess.run(
        ["julia", "-e", julia_code],
        capture_output=True,
        text=True
    )

    # Filter output
    for line in result.stdout.splitlines():
        if (
            "Original number of sequences" in line or
            "Sequences after filtering" in line or
            "Filtered sequences saved" in line or
            "removing duplicate sequences" in line
        ):
            print(line)

    data = np.loadtxt("scores.csv", delimiter=",")
    # data columns: i, j, score
    i = data[:, 0].astype(int)
    j = data[:, 1].astype(int)
    s = data[:, 2]

    # Build matrix
    N = int(max(i.max(), j.max()))
    score_map = np.zeros((N, N))

    for ii, jj, val in zip(i, j, s):
        score_map[ii-1, jj-1] = val
        score_map[jj-1, ii-1] = val

    if os.path.exists("scores.csv"):
        os.remove("scores.csv")
        
    return score_map

def extract_scop_ids(sto_file_path):
    scop_ids = []

    with open(sto_file_path, "r") as f:
        for line in f:
            line = line.strip()

            # Look for SCOP annotation lines
            if line.startswith("#=GF DR") and "SCOP;" in line:
                parts = line.split(";")

                # Expected format:
                # #=GF DR   SCOP; 5pti; fa;
                if len(parts) >= 2:
                    scop_id = parts[1].strip()
                    scop_ids.append(scop_id)
    if scop_ids == []:
        print("No sequences with SCOP annotation found in the .sto file.")
    return scop_ids