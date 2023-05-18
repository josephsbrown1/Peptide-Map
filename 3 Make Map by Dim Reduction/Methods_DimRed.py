import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from rdkit.Chem import AllChem as Chem
import esm
import torch
import json
from tqdm import tqdm

def lister(list1,list2): # Makes combinations of each element in both lists for plotting
    out = []
    for i in list1:
        for l in list2:
            out.append((i,l))
    return out

# Fingerprint encoding database and function, canonicalized / standardized previously using RDKit
Std_All_AA_Smiles = {
'A' : 'CNC(=O)[C@H](C)NC(C)=O',
'C' : 'CNC(=O)[C@H](CS)NC(C)=O', # note that Cys is not used in AS-MS libraries for ths work, but is included for other use cases
'D' : 'CNC(=O)[C@H](CC(=O)O)NC(C)=O',
'E' : 'CNC(=O)[C@H](CCC(=O)O)NC(C)=O',
'F' : 'CNC(=O)[C@H](Cc1ccccc1)NC(C)=O',
'G' : 'CNC(=O)CNC(C)=O',
'H' : 'CNC(=O)[C@H](Cc1c[nH]cn1)NC(C)=O',
'I' : 'CC[C@H](C)[C@H](NC(C)=O)C(=O)NC', # note that Ile is not used in AS-MS libraries for ths work, but is included for other use cases
'K' : 'CNC(=O)[C@H](CCCCN)NC(C)=O',
'L' : 'CNC(=O)[C@H](CC(C)C)NC(C)=O',
'M' : 'CNC(=O)[C@H](CCSC)NC(C)=O',
'N' : 'CNC(=O)[C@H](CC(N)=O)NC(C)=O',
'P' : 'CNC(=O)[C@@H]1CCCN1C(C)=O',
'Q' : 'CNC(=O)[C@H](CCC(N)=O)NC(C)=O',
'R' : 'CNC(=O)[C@H](CCCNC(=N)N)NC(C)=O',
'S' : 'CNC(=O)[C@H](CO)NC(C)=O',
'T' : 'CNC(=O)[C@@H](NC(C)=O)[C@@H](C)O',
'V' : 'CNC(=O)[C@@H](NC(C)=O)C(C)C',
'W' : 'CNC(=O)[C@H](Cc1c[nH]c2ccccc12)NC(C)=O',
'Y' : 'CNC(=O)[C@H](Cc1ccc(O)cc1)NC(C)=O',
'B' : 'CNC(=O)[C@H](CCCCN(Cc1ccccn1)Cc1ccccn1)NC(C)=O', # if desired, this and the other noncanonicals below can be commented out, but it does not negatively affect the results of canonical peptides to leave them in
'Z' : 'CNC(=O)[C@H](CCCNC(=O)N[C@@H]1O[C@H](CO)[C@H](O)[C@H](O)[C@H]1O)NC(C)=O',
'X' : 'CNC(=O)[C@H](CC(=O)N[C@@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1NC(C)=O)NC(C)=O',
'p' : 'CNC(=O)[C@H](CCCCNC(N)=O)NC(C)=O',
'n' : 'CNC(=O)[C@H](COP(=O)(O)O)NC(C)=O',
'z' : 'CCOP(=O)(Cc1ccc(C[C@H](NC(C)=O)C(=O)NC)cc1)OCC',
'v' : 'CNC(=O)[C@H](Cc1ccccc1C(F)(F)F)NC(C)=O',
'y' : 'CNC(=O)[C@H](Cc1c(F)c(F)c(F)c(F)c1F)NC(C)=O',
'm' : 'CNC(=O)[C@H](Cc1cccc(F)c1)NC(C)=O',
'r' : 'CNC(=O)[C@H](Cc1ccc(F)c(F)c1)NC(C)=O',
'u' : 'CNC(=O)[C@H](Cc1cccc2ccccc12)NC(C)=O',
'w' : 'N[C@H](C(=O)O)C(c1ccccc1)c1ccccc1',
'i' : 'CNC(=O)[C@H](Cc1cscn1)NC(C)=O',
'x' : 'CNC(=O)[C@H](CNC(C)=O)NS(=O)(=O)c1ccccc1',
'k' : 'CNC(=O)[C@H](Cc1ccc(N)cc1)NC(C)=O',
'j' : 'CNC(=O)[C@@H]1Cc2ccccc2CN1C(C)=O',
's' : 'CNC(=O)C1(c2ccccc2)CCN(C(C)=O)CC1',
'f' : 'CNC(=O)C1(NC(C)=O)CCNCC1',
't' : 'CNC(=O)[C@H](Cc1ccc(C(=O)O)cc1)NC(C)=O',
'o' : 'CNC(=O)[C@H](CCCCNC(=N)N)NC(C)=O',
'd' : 'CNC(=O)[C@H](CC1CC1)NC(C)=O',
'g' : 'CNC(=O)C1(NC(C)=O)CCOCC1',
'l' : 'CNC(=O)[C@H](CCS(C)(=O)=O)NC(C)=O',
'e' : 'CNC(=O)[C@@H]1C[C@@H](O)CN1C(C)=O',
'a' : 'CNC(=O)C1CN(C(C)=O)C1',
'h' : 'CNC(=O)c1ccc(CNC(C)=O)cc1',
'b' : 'CNC(=O)C(C)(C)NC(C)=O'
}

class Fingerprint_Generation:
    def __init__(self, smiles,radius,nbits):
        self.lookupfps = {}
        for key, value in Std_All_AA_Smiles.items():
            mol = Chem.MolFromSmiles(value)
            fp = np.array(Chem.GetMorganFingerprintAsBitVect(mol,radius,nbits))
            self.lookupfps[key] = fp
        self.lookupfps[' '] = np.zeros(self.lookupfps['A'].shape)
        self.lookupfps['x'] = np.ones(self.lookupfps['A'].shape)
    def seq(self, seq):
        fp = np.asarray([self.lookupfps[seq[i]] for i in range(len(seq))])
        return fp

def fingerprint(sequence,NBITS,fp):
    fp_seq = fp.seq(sequence)
    return fp_seq.reshape(-1)

# OneHot encoding function
enc_one = preprocessing.OneHotEncoder(handle_unknown='ignore')
enc_one.fit(np.array(list(Std_All_AA_Smiles.keys())).reshape(-1, 1))
def one_hot(sequence):
    return enc_one.transform(np.array(list(sequence)).reshape(-1,1)).toarray().reshape(-1)


# NGrams encoding functions
# By default minlengthfinder will be used to identify the minimum peptide length from the input dataset
# The maximum N-gram length can also be manually set
# Find the minimum NGram encoding length parameter
def minlengthfinder(lst):
    lengths = []
    for x in lst: lengths.append(len(x))
    the_min = min(lengths)
    return the_min

def cal_n_mers(seq_list, n): # enumerate and index of all the unique n_mers in the input dataset
    theo_max = len(Std_All_AA_Smiles)**n

    n_mers = set()
    for seq in seq_list:
        all_mers = [seq[i:i+n] for i in range(len(seq)-n+1)]
        n_mers.update(all_mers)
        
        if n_mers == theo_max:
            print('NGrams pre-calculation reached the theoretical maximum and stopped')
            break
    return n_mers

def n_grams(seq,n_mers):
    n_gram = []
    for n_mer in n_mers:
        n = len(n_mer[0])
        seq_mers = [seq[i:i+n] for i in range(len(seq)-n+1)]
        for mer in n_mer:
            n_gram.append(seq_mers.count(mer))
    return np.array(n_gram)

def enc_ngrams(seqs,n_mers):
    return [n_grams(seq,n_mers) for seq in seqs]

# Extended physical properties database and encoding
extended_prop_aa = {
'A':[0.620140363440966,0.17686056841772,-0.188749006724763,-0.224734173518473,-0.0836274309924442,-0.553429077935438,-1.32955152740811,-1.38172110737113,-1.02627398746377,-0.509282137090254,-0.585795803359558,-0.781110425203712,-0.520029443227509,0.213921592667873],
'C':[0.290065653867549,0.066322713156645,-0.440414349024447,-0.226412219190666,-1.04998885579402,-0.476729477285148,-0.489295456136705,-0.77494270388318,-0.578373331393242,-0.427542694668279,-1.63430191054009,-1.21748496442366,0.881041618691234,2.14004615278222],
'D':[-0.900203753382047,1.42435922064985,1.57290838937303,-0.225916432969336,1.73759217728745,1.55580993994754,-0.724977037103076,-0.501892422313604,-0.698861227178295,0.712268419104827,0.73623363612894,0.746200462066116,1.58587491364422,-0.881976863948911],
'E':[-0.740167530558572,1.07695453268647,1.57290838937303,-0.224734173518473,1.47741794753318,2.03310948183212,-0.253613875170334,0.0940506525406323,-0.122614769075867,0.794007861526803,1.0097569684369,0.964387731676091,-0.623175779074287,-0.964999474298668],
'F':[1.1902694072496,-1.60753623793963,-1.1954103759235,-0.22355191406761,-1.16149209711728,-0.535290658862734,1.1707226358873,0.887196708528451,1.27871184494595,-0.90435610879647,-0.904906357718851,-0.999297694813687,0.322332299521183,0.363362291297433],
'G':[0.480108668470425,0.745340966903249,0.062916335574921,-0.218174540436266,0.250882292977333,-0.553429077935438,-1.80091468934086,-2.03184082539393,-1.7465820600918,0.912075945025212,-0.631383025410886,-0.999297694813687,-0.30514124354672,-0.837698138429041],
'H':[-0.400090557058688,0.17686056841772,-0.188749006724763,-0.225420646748007,0.771230752485876,-0.370490165573597,0.555901120322853,0.447282365999688,0.322666584912373,-0.304933531035315,1.0097569684369,1.00802518559809,0.262163603610562,0.440850060957206],
'I':[1.38031242185247,-1.41804277177779,-0.843078896703942,-0.224162112493862,-1.16149209711728,-0.545655469761422,0.105032008908926,-0.0186367652499869,0.702465386843518,-1.24493711888804,-1.26960413412947,-0.868385333047702,-1.14750298629541,0.700987573386442],
'K':[-1.50033958897008,0.745340966903249,1.57290838937303,-0.224314662100425,1.10574047645565,2.01186161948981,0.443183842469371,0.952208680330731,0.872720022191963,1.37526611874974,1.60239085510416,2.27351134933594,-0.743513170895529,-0.527747059789951],
'L':[1.06023997620552,-1.60753623793963,-0.843078896703942,-0.22301799044464,-1.27299533844054,-0.53010825341339,0.105032008908926,0.243578187685877,0.702465386843518,-0.972472310814783,-0.996080801821506,-0.737472971281717,-0.700535530959371,1.24340196100485],
'M':[0.6401448912939,-0.817980128931955,-0.591413554404258,-0.224886723125036,-0.975653361578516,-0.47932068000982,0.463677892988186,0.466785957540372,0.718181199337221,-0.768123704759844,-0.631383025410886,-0.562923155593736,-1.10452534635925,-1.10337049154826],
'N':[-0.780176586264441,0.950625555245245,1.06957770477366,-0.224810448321755,1.21724371777891,-0.378263773747613,-0.42781330458026,-0.354531952895101,-0.209051737791231,0.871206223814225,0.872995302282922,0.702563008144121,2.74627119192048,-1.38011252604745],
'P':[0.120027167117606,0.17686056841772,0.062916335574921,-0.215848158936181,-0.120795178100197,-0.553429077935438,-0.458554380358482,-0.759773243795982,-0.649094487614904,0.303571206994949,-0.585795803359558,-0.781110425203712,-0.520029443227509,0.213921592667873],
'Q':[-0.850192433749711,0.792714333443709,0.163582472494795,-0.223132402649562,0.808398499593628,-0.370490165573597,0.0435498573524813,0.245745253412619,0.367194720311197,0.603282495875526,1.0097569684369,1.00802518559809,0.262163603610562,-0.0572856011413323],
'R':[-2.5305727733962,0.508474134200945,1.57290838937303,-0.223323089657766,0.808398499593628,2.1414217557234,1.18096966114671,1.60666252980702,1.00368512630615,1.33439639753875,1.51121641100151,1.48803717874003,-0.820872922780613,-0.843232979119025],
'S':[-0.180040750676409,0.871669944344477,0.213915540954732,-0.224810448321755,0.325217787192839,-0.466882906931394,-1.16559912325759,-1.12817441734224,-0.979126549982658,1.42975908036439,0.143599749461682,-0.0392737085297955,1.31941354604004,-0.073890123211283],
'T':[-0.050011319632336,0.666385356002481,-0.138415938264826,-0.224886723125036,0.102211304546321,-0.467401147476329,-0.694235961324853,-0.636250497371649,-0.40288009188023,0.603282495875526,0.371535859718319,-0.0392737085297955,0.399692051406267,-0.64397871427961],
'V':[1.08024450405846,-0.739024518031187,-0.692079691324131,-0.222827303436436,-0.90131786736301,-0.546691950851291,-0.366331153023816,-0.376202610162528,0.12621892874109,-0.727253983548856,-1.17842969002682,-0.824747879125707,-0.545816027189203,1.97953577277269],
'W':[0.810183378043843,-1.73386521538086,-1.64840799206293,-0.22355191406761,-1.08715660290178,-0.444598563499216,2.39011864175678,1.82987029966151,2.06450246963108,-1.64909325086336,-0.494621359256903,-0.693835517359722,-0.666153419010445,-0.73253616531935],
'Y':[0.260058862088147,-0.454784318788423,-1.09474423900363,4.2485160771289,-0.789814626039751,-0.469992350201001,1.25269883796256,1.19058591027243,-1.74595342759205,-1.43112140440476,0.645059192026285,0.35346337676816,-0.0816575158786997,0.750801139596297],
}


def ext_prop_enc(sequence):
    temp_list = []
    for letter in sequence:
        temp_list.append(letter)
    
    new_list = []
    for i,letter in enumerate(temp_list):
        new_list.append(extended_prop_aa[temp_list[i]])
    return np.array(new_list).reshape(-1)


def ESM2(seqs,OUTPUT_CSV_FILE,BATCH_SIZE=32):
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    # model.cuda() # comment this back in if you have access to GPU
    model.eval()  # disables dropout for deterministic results
        
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    def call_esm_model(data_chunk):
        # torch.cuda.empty_cache() # Comment back in if you have GPU
        batch_labels, _batch_strs, batch_tokens = batch_converter(data_chunk)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        # Extract per-residue representations (on CPU)
        # batch_tokens.cuda() # If you have GPU you'll need to send tokens to cuda
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        aa_reps = results["representations"][33].detach().cpu().numpy()
        # Generate per-sequence representations via averaging
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        mean_reps = []
        for i, tokens_len in enumerate(batch_lens):
            mean_reps.append(aa_reps[i, 1 : tokens_len - 1].mean(0))
        return batch_labels, mean_reps, aa_reps

    seq_col_name = "Sequence"
    rep_col_name = "ESM2 Representation"

    if os.path.exists(OUTPUT_CSV_FILE):
        df = pd.read_csv(OUTPUT_CSV_FILE, index_col=0)
    else:
        df = pd.DataFrame(seqs, columns=[seq_col_name])
        df.set_index(seq_col_name, inplace=True)
        df[rep_col_name] = np.nan

    esm2_input, mean_reps = None, None
    for chunk in tqdm(
        chunks(df, BATCH_SIZE),
        total=np.ceil(len(seqs) / BATCH_SIZE),
    ):
        if not (chunk[rep_col_name].isnull().values.any()):
            # Skip rows which we already know the ESM2 representations for
            continue
        esm2_input = [(seq, seq) for seq in chunk.index.to_numpy()]
        batch_labels, _mean_reps, aa_reps = call_esm_model(esm2_input)

        mask = df.index.isin(batch_labels) & df.columns.isin([rep_col_name])
        df.loc[mask, rep_col_name] = [
            json.dumps(
                [a.item() for a in aa_rep.reshape(aa_rep.shape[0] * aa_rep.shape[1])]
            )
            for aa_rep in aa_reps
        ]
        df.to_csv(OUTPUT_CSV_FILE)
        
    df[rep_col_name] = df[rep_col_name].apply(lambda x: np.array(json.loads(x)))

    return np.vstack(df[rep_col_name])

            
            

