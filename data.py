import re
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import numpy as np
import pandas as pd




def read_fasta(filepath):
    records = {}
    header = None
    seq_chunks = []

    with open(filepath,"r",encoding = "utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records[header] = "".join(seq_chunks).upper()
                header = line[1:].split()[0]
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if header is not None:
            records[header] = "".join(seq_chunks).upper()
    return records


def find_cds_positions(all_fa, cds_fa):
    all_records = read_fasta(all_fa)
    cds_records = read_fasta(cds_fa)

    rows = []

    for gene, cds_seq in cds_records.items():
        full_seq = all_records.get(gene, None)

        if full_seq is None:
            rows.append({
                "gene": gene,
                "cds_start_nt": None,
                "cds_end_nt": None,
                "status": "missing_in_all_fa"
            })
            continue

        matches = []
        start = 0
        while True:
            pos = full_seq.find(cds_seq, start)
            if pos == -1:
                break
            matches.append(pos)
            start = pos + 1

        if len(matches) == 1:
            cds_start = matches[0]
            cds_end = cds_start + len(cds_seq) - 1
            rows.append({
                "gene": gene,
                "cds_start_nt": cds_start+1,   # 1-based
                "cds_end_nt": cds_end+1,       # 1-based inclusive
                "status": "found"
            })
        elif len(matches) == 0:
            rows.append({
                "gene": gene,
                "cds_start_nt": None,
                "cds_end_nt": None,
                "status": "not_found"
            })
        else:
            rows.append({
                "gene": gene,
                "cds_start_nt": None,
                "cds_end_nt": None,
                "status": "multiple_matches"
            })

    return pd.DataFrame(rows)


def find_gene_cds_position(all_fa, cds_fa, gene):
    df_fa = find_cds_positions(all_fa,cds_fa)
    df_gene = df_fa.loc[df_fa["gene"]==gene]

    if df_gene.empty:
        raise ValueError(f"{gene} not found in CDS position table.")
    
    row = df_gene.iloc[0]
    if row["status"] != "found":
        raise ValueError(f"{gene} CDS position status = {row['status']}")

    cds_start_nt = int(row["cds_start_nt"])
    cds_end_nt = int(row["cds_end_nt"])
    return cds_start_nt,cds_end_nt





@dataclass
class TripletData:
    """
    Container for triplet-level observations built from one gene and one frame.

    Attributes
    ----------
    gene_name : str
        Target gene name, e.g. "RPS20".
    frame_offset : int
        Offset used when building triplets. This is your triplet-start offset
        in the sorted kmer table, not the biological reading frame label itself.
    kmer_df : pd.DataFrame
        Kmer-level dataframe after preprocessing.
    triplet_df : pd.DataFrame
        Triplet-level dataframe.
    X : np.ndarray
        Array of shape (T, 3), each row is [x1, x2, x3].
    N : np.ndarray
        Array of shape (T,), total count of each triplet.
    has_ATG_triplet : np.ndarray
        Array of shape (T,), whether the triplet contains ATG in any of its 3 kmers.
    has_STOP_triplet : np.ndarray
        Array of shape (T,), whether the triplet contains STOP in any of its 3 kmers.
    N_asin: np.ndarray
        Signal normalisation
    range_score: np.ndarray
    T : int
        Number of triplets.
    """
    gene_name: str
    frame_offset: int
    kmer_df: pd.DataFrame
    triplet_df: pd.DataFrame
    X: np.ndarray
    N: np.ndarray
    has_ATG_triplet: np.ndarray
    has_STOP_triplet: np.ndarray
    range_score:np.ndarray
    region_type:np.ndarray
    T: int

    @staticmethod
    def load_kmer_table(path: str, sep: str = "\t") -> pd.DataFrame:
        """
        Load the raw kmer count table.

        Parameters
        ----------
        path : str
            Path to the KmersFromContigsQuerySum-like file.
        sep : str
            File separator, default is tab.

        Returns
        -------
        pd.DataFrame
        """
        df = pd.read_csv(path, sep=sep)

        # Keep your original column cleanup:
        # e.g. SRRxxx_1 -> SRRxxx
        df = df.rename(
            columns=lambda x: x.split("_")[0] if x.startswith(("SRR", "ERR")) else x
        )
        return df

    @staticmethod
    def subset_gene(df: pd.DataFrame, gene_name: str, id_col: str = "id") -> pd.DataFrame:
        """
        Select rows belonging to one gene.

        Parameters
        ----------
        df : pd.DataFrame
            Raw dataframe.
        gene_name : str
            Gene prefix, e.g. "RPS20".
        id_col : str
            Column containing identifiers like 'RPS20_kmer_123'.

        Returns
        -------
        pd.DataFrame
        """
        out = df[df[id_col].astype(str).str.startswith(gene_name)].copy()
        out["gene_name"] = gene_name
        if out.empty:
            raise ValueError(f"No rows found for gene '{gene_name}' in column '{id_col}'.")
        return out

    @staticmethod
    def add_k_idx_and_frame(
        df: pd.DataFrame,
        gene_name: str,
        id_col: str = "id",
        k_idx_col: str = "k_idx",
        frame_col: str = "frame",
        offset_nt: int = 0,
    ) -> pd.DataFrame:
        """
        Extract kmer index from id and compute frame.

        Parameters
        ----------
        df : pd.DataFrame
            Gene-specific dataframe.
        gene_name : str
            Gene prefix used in the id pattern.
        id_col : str
            Column like 'RPS20_kmer_123'.
        k_idx_col : str
            Output column for kmer index.
        frame_col : str
            Output column for frame label (0,1,2).
        offset_nt : int
            Optional nucleotide offset. If you want the exact same behavior as your
            earlier code, leave this as 0 and use frame = (k_idx - 1) % 3.

            If later you want to account for a P-site or other shift, you can change it.

        Returns
        -------
        pd.DataFrame
        """
        out = df.copy()

        pattern = rf"{re.escape(gene_name)}_kmer_(\d+)"
        extracted = out[id_col].astype(str).str.extract(pattern)[0]

        if extracted.isna().any():
            bad_rows = out.loc[extracted.isna(), id_col].tolist()[:5]
            raise ValueError(
                f"Could not extract k_idx from some '{id_col}' values. Examples: {bad_rows}"
            )
        out[k_idx_col] = extracted.astype(int)
        out = out.sort_values(k_idx_col).reset_index(drop=True)

        # Same logic as your current script by default:
        out[frame_col] = ((out[k_idx_col] - 1 + offset_nt) % 3).astype(int)
        return out


    @staticmethod
    def add_region_type_to_kmers(df, cds_start_nt, cds_end_nt, offset_nt, k_idx_col="k_idx"):
        """
        transcript_nt_pos = k_idx + offset_nt
        """
        df = df.copy()
        df["transcript_nt_pos"] = df[k_idx_col] + offset_nt

        def classify(nt_pos):
            if nt_pos < cds_start_nt:
                return "UTR5"
            elif nt_pos > cds_end_nt:
                return "UTR3"
            else:
                return "CDS"

        df["region_type"] = df["transcript_nt_pos"].apply(classify)
        return df


    @staticmethod
    def add_has_atg(
        df: pd.DataFrame,
        seq_col: str = "tag",
        out_col: str = "has_ATG",
        motif: str = "ATG",
        offset = 12
    ) -> pd.DataFrame:
        """
        Add a binary column indicating whether the kmer sequence contains ATG.

        Parameters
        ----------
        df : pd.DataFrame
            Kmer dataframe.
        seq_col : str
            Column containing the kmer sequence/tag.
        out_col : str
            Output binary column.
        motif : str
            Sequence motif to search for.

        Returns
        -------
        pd.DataFrame
        """
        out = df.copy()
        if seq_col not in out.columns:
            raise ValueError(f"Column '{seq_col}' not found in dataframe.")
        
        out[out_col] = out[seq_col].astype(str).str.slice(offset, offset+3).str.contains(motif, na=False).astype(int)
        return out
    

    @staticmethod
    def add_start_type(df: pd.DataFrame, seq_col: str="tag", offset: int = 12) -> pd.DataFrame:
        """
        Classify start codons into two groups: 
        Binary columns indicating whether the kmer sequence contains a canonical or non-canonical start codon.
        1:Candidate; 0:Not candidate.

        - Canonical: ATG. 
        - Non-canonical: CTG, GTG, TTG, ACG, AAG, AGG, ATA, ATT, ATC
        
        Both columns are 0 when the codon is not a candicate start codon
        """
        out_df = df.copy()
        if seq_col not in out_df.columns:
            raise ValueError(f"Column '{seq_col}' not found in dataframe.")
        
        non_canonical_codon_start = {"CTG", "GTG", "TTG", "ACG", "AAG", "AGG", "ATA", "ATT", "ATC"}

        out_df["start_codon"] = out_df[seq_col].astype(str).str.slice(offset, offset+3)
        out_df["has_canonical_start"] = (out_df["start_codon"] == "ATG").astype(int)
        out_df["has_non_canonical_start"] = out_df["start_codon"].isin(non_canonical_codon_start).astype(int)

        return out_df
    

    @staticmethod
    def add_has_stop(
        df: pd.DataFrame,
        seq_col: str = "tag",
        out_col: str = "has_STOP",
        motif: list = ["TAA", "TGA", "TAG"],
        offset: int = 12
        ) -> pd.DataFrame:
            """
            Add a binary column indicating whether the kmer sequence contains codon stop.

            Parameters
            ----------
            df : pd.DataFrame
                Kmer dataframe.
            seq_col : str
                Column containing the kmer sequence/tag.
            out_col : str
                Output binary column.
            motif : str
                Sequence motif to search for.

            Returns
            -------
            pd.DataFrame
            """
            out = df.copy()
            if seq_col not in out.columns:
                raise ValueError(f"Column '{seq_col}' not found in dataframe.")
            pattern = "|".join(motif)
            out[out_col] = out[seq_col].astype(str).str.slice(offset,offset+3).str.contains(pattern, na=False).astype(int)
            return out

    @staticmethod
    def build_sliding_triplets(
        df: pd.DataFrame,
        signal_col: str = "sum",
        offset: int = 0,
        k_idx_col: str = "k_idx",
        frame_col: str = "frame",
        has_canonical_start_col: str = "has_canonical_start",
        has_non_canonical_start_col: str = "has_non_canonical_start",
        stop_col: str = "has_STOP"
    ) -> pd.DataFrame:
        required_cols = [signal_col, k_idx_col, frame_col, has_canonical_start_col, has_non_canonical_start_col, stop_col, "gene_name"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns for triplet building: {missing}")

        df = df.sort_values(k_idx_col).reset_index(drop=True)

        signal = df[signal_col].to_numpy()
        k_idx = df[k_idx_col].to_numpy()
        frame = df[frame_col].to_numpy()
        has_canonical_start = df[has_canonical_start_col].to_numpy()
        has_non_canonical_start = df[has_non_canonical_start_col].to_numpy()
        has_STOP = df[stop_col].to_numpy()
        gene_name = str(df["gene_name"].iloc[0])
        #region_type = df["region_type"].to_numpy()

        triplets = []
        for i in range(offset, len(df) - 2, 3):
            x1, x2, x3 = signal[i], signal[i + 1], signal[i + 2]
            canonical_start_1 = has_canonical_start[i]
            non_canonical_start_1 = has_non_canonical_start[i]
            stop1 = has_STOP[i]
            #region_1 = region_type[i]

            triplets.append({
                "gene_name": gene_name,
                "triplet_idx": len(triplets),
                #"row_start": int(i),
                #"row_mid": int(i + 1),
                #"row_end": int(i + 2),
                "start_kidx": int(k_idx[i]),
                "mid_kidx": int(k_idx[i + 1]),
                "end_kidx": int(k_idx[i + 2]),
                "frame_start": int(frame[i]),
                #"frame_mid": int(frame[i + 1]),
                #"frame_end": int(frame[i + 2]),
                "x1": float(x1),
                "x2": float(x2),
                "x3": float(x3),
                "N": float(x1 + x2 + x3),
                "has_canonical_start": int(canonical_start_1),
                "has_non_canonical_start": int(non_canonical_start_1),
                "has_STOP_triplet": int(stop1),
                #"region_type": str(region_1)
            })

        return pd.DataFrame(triplets)



    @staticmethod
    def add_N_asin(triplet_df: pd.DataFrame) -> pd.DataFrame:
        """        
        N_norm = N / max(N within that gene)
        N_asin = asin(sqrt(N_norm))
        """
        required_cols = {"gene_name", "N"}
        missing = required_cols - set(triplet_df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        df = triplet_df.copy()

        # Ensure numeric
        df["N"] = pd.to_numeric(df["N"], errors="coerce")

        # Per-gene maximum N
        gene_max = df.groupby("gene_name")["N"].transform("max")

        # Avoid division by zero / NaN
        df["N_norm"] = np.where(
            (gene_max > 0) & np.isfinite(gene_max),
            df["N"] / gene_max,
            0.0
        )

        # Numerical safety: clamp to [0, 1]
        df["N_norm"] = np.clip(df["N_norm"], 0.0, 1.0)

        # Arcsine sqrt transform
        df["N_asin"] = np.arcsin(np.sqrt(df["N_norm"]))

        return df


    @staticmethod
    def add_x1_asin(triplet_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add x1_asin column.
        x1_asin = asin(sqrt(x1 / max(x1 within that gene)))
        """
        df = triplet_df.copy()
        df["x1"] = pd.to_numeric(df["x1"], errors="coerce")
        x1_max = df.groupby("gene_name")["x1"].transform("max")
        df["x1_norm"] = np.where(
            (x1_max > 0) & np.isfinite(x1_max),
            df["x1"] / x1_max,
            0.0
        )

        df["x1_norm"] = np.clip(df["x1_norm"], 0.0, 1.0)
        df["x1_asin"] = np.arcsin(np.sqrt(df["x1_norm"]))

        return df
    

    @staticmethod
    def add_range_score(triplet_df: pd.DataFrame,signal_col: str="N", out_col: str="range_score", up_codons: int=10,
                        down_codons:int = 10, downstream_len: int = 10, eps=1e-12)->pd.DataFrame:
        """
        Add triplet-level range score.

        The score is defined as:
            range_score(i) =
                sum of signal from codon i to i + downstream_len - 1
                -----------------------------------------------------
                sum of signal from codon i - up_codons to i + down_codons

        This is computed within one gene and one triplet frame only, which matches
        the current design of TripletData.from_gene().

        Parameters
        ----------
        triplet_df : pd.DataFrame
            Triplet-level dataframe.
        signal_col : str
            Column used as triplet signal, default is "N".
        out_col : str
            Output column name.
        up_codons : int
            Number of upstream codons included in denominator.
        down_codons : int
            Number of downstream codons included in denominator.
        downstream_len : int
            Number of downstream codons included in numerator, including current codon.
        strict_window : bool
            If True, require a full window on both sides; otherwise use truncated
            windows near boundaries.

        Returns
        -------
        pd.DataFrame
            Triplet dataframe with added range_score column.
        """
        out = triplet_df.copy()

        if signal_col not in out.columns:
            raise ValueError(f"Column '{signal_col}' not found in triplet_df ❗ ")
        
        signal = out[signal_col].to_numpy(dtype=float)
        n = len(signal)

        scores = np.full(n, np.nan, dtype=float)
        numerators = np.full(n, np.nan, dtype=float)
        denominators = np.full(n, np.nan, dtype=float)

        for i in range(n):
                left_avail = min(up_codons, i)
                right_avail = min(down_codons, n - 1 - i)

                w = min(left_avail, right_avail)
                denom_start = i - w
                denom_end = i + w

                numer_end = min(i + downstream_len - 1, i + w)

                denom = signal[denom_start:denom_end + 1].sum()
                numer = signal[i:numer_end + 1].sum()

                numerators[i] = numer
                denominators[i] = denom
                if denom > 0:
                    scores[i] = numer/(denom + eps)
        
        out["range_numerator"] = numerators
        out["range_denominator"] = denominators
        out[out_col] = scores
        return out


    @staticmethod
    def add_x1_range_score(triplet_df: pd.DataFrame) -> pd.DataFrame:
        return TripletData.add_range_score(
            triplet_df,
            signal_col="x1",
            out_col="x1_range_score",
            up_codons=10,
            down_codons=10,
            downstream_len=10,
        )




    @classmethod
    def from_gene(
        cls,
        table_path: str,
        gene_name: str,
        signal_col: str = "sum",
        sep: str = "\t",
        id_col: str = "id",
        seq_col: str = "tag",
        offset_nt: int = 0,
        triplet_offset: int = 0,
    ) -> "TripletData":
        """
        Full pipeline:
        load raw table -> subset one gene -> add k_idx/frame -> add has_ATG -> add has_STOP -> build triplets -> add range_score

        Parameters
        ----------
        table_path : str
            Path to the raw kmer table.
        gene_name : str
            Target gene, e.g. "RPS20".
        signal_col : str
            Count/signal column to use.
        sep : str
            File separator.
        id_col : str
            Identifier column.
        seq_col : str
            Sequence/tag column used to detect ATG/STOP.
        offset_nt : int
            Optional nucleotide offset used in frame computation.
        triplet_offset : int
            Row offset used to build triplets.

        Returns
        -------
        TripletData
        """
        df = cls.load_kmer_table(table_path, sep=sep)
        df = cls.subset_gene(df, gene_name=gene_name, id_col=id_col)
        df = cls.add_k_idx_and_frame(
            df,
            gene_name=gene_name,
            id_col=id_col,
            offset_nt=offset_nt,
        )
        df = cls.add_has_atg(df, seq_col=seq_col, out_col="has_ATG")
        df = cls.add_has_stop(df, seq_col=seq_col, out_col="has_STOP")
        triplet_df = cls.build_sliding_triplets(
            df,
            signal_col=signal_col,
            offset=triplet_offset,
        )
        triplet_df = cls.add_N_asin(triplet_df)
        triplet_df = cls.add_range_score(
            triplet_df,
            signal_col="N",
            out_col="range_score",
            up_codons=10,
            down_codons=10,
            downstream_len=10,
        )

        X = triplet_df[["x1", "x2", "x3"]].to_numpy()
        N = triplet_df["N"].to_numpy()
        has_ATG_triplet = triplet_df["has_ATG_triplet"].to_numpy(dtype=int)
        has_STOP_triplet = triplet_df["has_STOP_triplet"].to_numpy(dtype=int)
        region_type = triplet_df["region_type"].to_numpy(dtype=str)
        N_asin = triplet_df["N_asin"].to_numpy(dtype=int)
        range_score = triplet_df["range_score"].to_numpy(dtype=float)
        T = len(triplet_df)

        return cls(
            gene_name=gene_name,
            frame_offset=triplet_offset,
            kmer_df=df,
            triplet_df=triplet_df,
            X=X,
            N=N,
            has_ATG_triplet=has_ATG_triplet,
            has_STOP_triplet=has_STOP_triplet,
            region_type=region_type,
            N_asin = N_asin,
            range_score = range_score,
            T=T,
        )

    def summary(self) -> str:
        """
        Return a short textual summary.
        """
        return (
            f"TripletData(gene={self.gene_name}, "
            f"n_kmers={len(self.kmer_df)}, "
            f"n_triplets={self.T}, "
            f"frame_offset={self.frame_offset})"
        )

    def to_numpy(self) -> np.ndarray:
        """
        Return triplet counts array of shape (T, 3).
        """
        return self.X