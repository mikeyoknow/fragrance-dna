"""
Fragrance DNA - Graph Construction & Recommendation Pipeline
EECS4414 | Matin Pakfetrat

Inputs (from data/processed/):
    - perfumes_cleaned.csv   : perfume metadata
    - perfume_notes.csv      : perfume-note edges with layer (top/middle/base)
    - perfume_accords.csv    : perfume-accord mappings

Outputs:
    - outputs/bipartite_edges.csv
    - outputs/note_cooccurrence_edges.csv
    - outputs/perfume_similarity_edges.csv
    - outputs/graph_stats.txt
    - outputs/graph_stats.csv
    - outputs/top_central_nodes.csv
    - outputs/communities.csv
    - outputs/recommendations.csv
    - outputs/top_k_recommendations.csv
"""

import gc
import os
from collections import defaultdict
from itertools import combinations

import community as community_louvain
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = "data/processed"
OUTPUT_DIR = "outputs"

SIM_THRESHOLD = 0.30
TOP_K = 5

LAYER_WEIGHTS = {"top": 1.0, "middle": 1.2, "base": 1.5}

NOTE_EDGE_WEIGHT = 0.5
ACCORD_EDGE_WEIGHT = 0.2
WEIGHTED_OVERLAP_WEIGHT = 0.3

MAX_EDGES_PER_PERFUME = 125
PRESELECT_K = 400
APPROX_BETWEENNESS_K = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)


def safe_mean(values):
    return float(np.mean(values)) if len(values) > 0 else 0.0


def jaccard_from_counts(shared, count_a, count_b):
    union = count_a + count_b - shared
    return float(shared / union) if union > 0 else 0.0


def cosine_from_counts(shared, count_a, count_b):
    denom = np.sqrt(count_a * count_b)
    return float(shared / denom) if denom > 0 else 0.0


# ── 1. Load Data ──────────────────────────────────────────────────────────────
print("Loading data...")
perfumes = pd.read_csv(f"{DATA_DIR}/perfumes_cleaned.csv")
notes_df = pd.read_csv(f"{DATA_DIR}/perfume_notes.csv")
accords_df = pd.read_csv(f"{DATA_DIR}/perfume_accords.csv")

perfumes["perfume_id"] = perfumes["perfume_id"].astype(str)
notes_df["perfume_id"] = notes_df["perfume_id"].astype(str)
accords_df["perfume_id"] = accords_df["perfume_id"].astype(str)

print(f"  Perfumes : {len(perfumes):,}")
print(f"  Note rows: {len(notes_df):,}")
print(f"  Accord rows: {len(accords_df):,}")


# ── 2. Perfume-Note Bipartite Graph G_B ───────────────────────────────────────
print("\nBuilding bipartite graph G_B...")

G_B = nx.Graph()

for _, row in perfumes.iterrows():
    pid = row["perfume_id"]
    G_B.add_node(
        f"P:{pid}",
        bipartite=0,
        node_type="perfume",
        label=row.get("perfume", pid),
        brand=row.get("brand", ""),
        rating=row.get("rating_value", None),
    )

for _, row in notes_df.iterrows():
    note_id = f"N:{str(row['note']).strip().lower()}"
    perf_id = f"P:{row['perfume_id']}"
    layer = str(row.get("note_layer", "base")).lower().strip()

    if not G_B.has_node(note_id):
        G_B.add_node(note_id, bipartite=1, node_type="note", label=str(row["note"]).strip().lower())

    weight = LAYER_WEIGHTS.get(layer, 1.0)
    G_B.add_edge(perf_id, note_id, layer=layer, weight=weight)

bipartite_edges = [
    {"perfume_node": u, "note_node": v, "layer": d["layer"], "weight": d["weight"]}
    for u, v, d in G_B.edges(data=True)
]
pd.DataFrame(bipartite_edges).to_csv(f"{OUTPUT_DIR}/bipartite_edges.csv", index=False)

perfume_nodes = [n for n, d in G_B.nodes(data=True) if d["node_type"] == "perfume"]
note_nodes = [n for n, d in G_B.nodes(data=True) if d["node_type"] == "note"]

print(f"  Perfume nodes : {len(perfume_nodes):,}")
print(f"  Note nodes    : {len(note_nodes):,}")
print(f"  Edges         : {G_B.number_of_edges():,}")


# ── 3. Note Co-occurrence Graph G_N ───────────────────────────────────────────
print("\nBuilding note co-occurrence graph G_N...")

note_to_perfumes = defaultdict(set)
perfume_to_notes = defaultdict(set)

for _, row in notes_df.iterrows():
    note = str(row["note"]).strip().lower()
    pid = row["perfume_id"]
    if note:
        note_to_perfumes[note].add(pid)
        perfume_to_notes[pid].add(note)

G_N = nx.Graph()
G_N.add_nodes_from(note_to_perfumes.keys())

cooccurrence = defaultdict(int)
for notes_in_perfume in perfume_to_notes.values():
    for n1, n2 in combinations(sorted(notes_in_perfume), 2):
        cooccurrence[(n1, n2)] += 1

for (n1, n2), co_count in cooccurrence.items():
    union = len(note_to_perfumes[n1] | note_to_perfumes[n2])
    jaccard = co_count / union if union > 0 else 0.0
    if jaccard > 0:
        G_N.add_edge(n1, n2, weight=jaccard, co_count=co_count)

cooc_edges = [
    {"note_1": u, "note_2": v, "jaccard": d["weight"], "co_count": d["co_count"]}
    for u, v, d in G_N.edges(data=True)
]
pd.DataFrame(cooc_edges).to_csv(f"{OUTPUT_DIR}/note_cooccurrence_edges.csv", index=False)

print(f"  Note nodes : {G_N.number_of_nodes():,}")
print(f"  Edges      : {G_N.number_of_edges():,}")
print(f"  Density    : {nx.density(G_N):.6f}")

components_N = list(nx.connected_components(G_N))
degrees_N = [d for _, d in G_N.degree()]
print(f"  Connected components : {len(components_N)}")
print(f"  Mean degree          : {safe_mean(degrees_N):.2f}")


# ── 4. Perfume Similarity Graph G_P ───────────────────────────────────────────
print("\nBuilding perfume similarity graph G_P...")
print("  Scoring local candidates row-by-row...")

notes_work = notes_df.copy()
notes_work["note_norm"] = notes_work["note"].astype(str).str.strip().str.lower()
notes_work["layer_norm"] = notes_work["note_layer"].fillna("base").astype(str).str.strip().str.lower()

accords_work = accords_df.copy()
accords_work["accord_norm"] = accords_work["accord"].astype(str).str.strip().str.lower()

notes_work = notes_work[
    (notes_work["perfume_id"].notna()) &
    (notes_work["note_norm"] != "")
].copy()

accords_work = accords_work[
    (accords_work["perfume_id"].notna()) &
    (accords_work["accord_norm"] != "")
].copy()

notes_work = notes_work.drop_duplicates(subset=["perfume_id", "note_norm", "layer_norm"])
accords_work = accords_work.drop_duplicates(subset=["perfume_id", "accord_norm"])

perfume_to_accords = defaultdict(set)
for _, row in accords_work.iterrows():
    perfume_to_accords[row["perfume_id"]].add(row["accord_norm"])

perfume_to_note_wts = defaultdict(dict)
for _, row in notes_work.iterrows():
    pid = row["perfume_id"]
    note = row["note_norm"]
    layer = row["layer_norm"] if row["layer_norm"] else "base"
    w = LAYER_WEIGHTS.get(layer, 1.0)
    perfume_to_note_wts[pid][note] = max(perfume_to_note_wts[pid].get(note, 0.0), w)

all_perfume_ids = list(perfumes["perfume_id"])

G_P = nx.Graph()
G_P.add_nodes_from(all_perfume_ids)

for _, row in perfumes.iterrows():
    pid = row["perfume_id"]
    G_P.nodes[pid]["label"] = row.get("perfume", pid)
    G_P.nodes[pid]["brand"] = row.get("brand", "")
    G_P.nodes[pid]["rating"] = row.get("rating_value", None)
    G_P.nodes[pid]["votes"] = row.get("rating_count", 0)

print("Preparing sparse matrices...")

perfume_index = {pid: idx for idx, pid in enumerate(all_perfume_ids)}
index_to_perfume = {idx: pid for pid, idx in perfume_index.items()}

note_vocab = sorted(notes_work["note_norm"].unique().tolist())
accord_vocab = sorted(accords_work["accord_norm"].unique().tolist())

note_index = {note: idx for idx, note in enumerate(note_vocab)}
accord_index = {acc: idx for idx, acc in enumerate(accord_vocab)}

valid_perfume_ids = set(all_perfume_ids)
notes_work = notes_work[notes_work["perfume_id"].isin(valid_perfume_ids)].copy()
accords_work = accords_work[accords_work["perfume_id"].isin(valid_perfume_ids)].copy()

note_rows = notes_work["perfume_id"].map(perfume_index).to_numpy(dtype=np.int32)
note_cols = notes_work["note_norm"].map(note_index).to_numpy(dtype=np.int32)
note_data = np.ones(len(notes_work), dtype=np.float32)

N_bin = sparse.csr_matrix(
    (note_data, (note_rows, note_cols)),
    shape=(len(all_perfume_ids), len(note_vocab)),
    dtype=np.float32,
)

acc_rows = accords_work["perfume_id"].map(perfume_index).to_numpy(dtype=np.int32)
acc_cols = accords_work["accord_norm"].map(accord_index).to_numpy(dtype=np.int32)
acc_data = np.ones(len(accords_work), dtype=np.float32)

A_bin = sparse.csr_matrix(
    (acc_data, (acc_rows, acc_cols)),
    shape=(len(all_perfume_ids), len(accord_vocab)),
    dtype=np.float32,
)

note_counts = np.asarray(N_bin.sum(axis=1)).ravel().astype(np.float32)
acc_counts = np.asarray(A_bin.sum(axis=1)).ravel().astype(np.float32)

print(f"  Perfume x Note Matrix   : {N_bin.shape}, nnz={N_bin.nnz:,}")
print(f"  Perfume x Accord Matrix : {A_bin.shape}, nnz={A_bin.nnz:,}")

print("Computing shared-note / shared-accord candidates...")

shared_note_matrix = (N_bin @ N_bin.T).tocsr()
shared_acc_matrix = (A_bin @ A_bin.T).tocsr()


def weighted_note_overlap(pid_a, pid_b):
    notes_a = perfume_to_note_wts.get(pid_a, {})
    notes_b = perfume_to_note_wts.get(pid_b, {})

    if not notes_a or not notes_b:
        return 0.0

    shared_notes = set(notes_a) & set(notes_b)
    if not shared_notes:
        return 0.0

    total_a = sum(notes_a.values())
    total_b = sum(notes_b.values())

    overlap_a = sum(notes_a[n] for n in shared_notes) / total_a if total_a > 0 else 0.0
    overlap_b = sum(notes_b[n] for n in shared_notes) / total_b if total_b > 0 else 0.0

    return 0.5 * (overlap_a + overlap_b)


def accord_aware_similarity(pid_a, pid_b, note_jaccard, accord_weight=0.3):
    accords_a = perfume_to_accords.get(pid_a, set())
    accords_b = perfume_to_accords.get(pid_b, set())

    union = accords_a | accords_b
    acc_jacc = len(accords_a & accords_b) / len(union) if union else 0.0
    return (1 - accord_weight) * note_jaccard + accord_weight * acc_jacc


sim_records = []
n_perfumes = len(all_perfume_ids)

for i in range(n_perfumes):
    if i % 1000 == 0:
        print(f"  Processed {i:,}/{n_perfumes:,} perfumes")

    pid_i = index_to_perfume[i]
    count_i_notes = float(note_counts[i])
    count_i_accs = float(acc_counts[i])

    row_notes = shared_note_matrix.getrow(i)
    row_accs = shared_acc_matrix.getrow(i)

    note_map = dict(zip(row_notes.indices.tolist(), row_notes.data.tolist()))
    acc_map = dict(zip(row_accs.indices.tolist(), row_accs.data.tolist()))

    candidate_indices = np.union1d(row_notes.indices, row_accs.indices)
    candidate_indices = candidate_indices[candidate_indices > i]

    if candidate_indices.size == 0:
        continue

    provisional = []

    for j in candidate_indices:
        pid_j = index_to_perfume[int(j)]

        shared_notes = float(note_map.get(int(j), 0.0))
        shared_accs = float(acc_map.get(int(j), 0.0))

        if shared_notes <= 0 and shared_accs <= 0:
            continue

        count_j_notes = float(note_counts[int(j)])
        count_j_accs = float(acc_counts[int(j)])

        note_jacc = jaccard_from_counts(shared_notes, count_i_notes, count_j_notes)
        acc_jacc = jaccard_from_counts(shared_accs, count_i_accs, count_j_accs)

        provisional_score = (0.7 * note_jacc) + (0.3 * acc_jacc)
        if provisional_score < SIM_THRESHOLD:
            continue

        cosine = cosine_from_counts(shared_notes, count_i_notes, count_j_notes)
        weighted = weighted_note_overlap(pid_i, pid_j)
        accord = accord_aware_similarity(pid_i, pid_j, note_jacc)

        combined_score = (
            NOTE_EDGE_WEIGHT * note_jacc
            + ACCORD_EDGE_WEIGHT * acc_jacc
            + WEIGHTED_OVERLAP_WEIGHT * weighted
        )

        if combined_score >= SIM_THRESHOLD:
            provisional.append(
                (
                    combined_score,
                    pid_j,
                    {
                        "jaccard": float(note_jacc),
                        "cosine": float(cosine),
                        "weighted_overlap": float(weighted),
                        "accord_jaccard": float(acc_jacc),
                        "accord_aware": float(accord),
                        "combined_score": float(combined_score),
                        "shared_notes": int(shared_notes),
                        "shared_accords": int(shared_accs),
                    },
                )
            )

    if not provisional:
        continue

    provisional.sort(key=lambda x: x[0], reverse=True)
    provisional = provisional[:PRESELECT_K]

    scored = []
    for _, pid_j, attrs in provisional:
        scored.append((attrs["combined_score"], pid_j, attrs))

    scored.sort(key=lambda x: x[0], reverse=True)
    scored = scored[:MAX_EDGES_PER_PERFUME]

    for _, pid_j, attrs in scored:
        if not G_P.has_edge(pid_i, pid_j):
            G_P.add_edge(
                pid_i,
                pid_j,
                weight=attrs["combined_score"],
                **attrs,
            )
            sim_records.append(
                {
                    "perfume_a": pid_i,
                    "perfume_b": pid_j,
                    "jaccard": round(attrs["jaccard"], 4),
                    "cosine": round(attrs["cosine"], 4),
                    "weighted_overlap": round(attrs["weighted_overlap"], 4),
                    "accord_jaccard": round(attrs["accord_jaccard"], 4),
                    "accord_aware": round(attrs["accord_aware"], 4),
                    "combined_score": round(attrs["combined_score"], 4),
                    "shared_notes": attrs["shared_notes"],
                    "shared_accords": attrs["shared_accords"],
                }
            )

    del row_notes, row_accs, note_map, acc_map, candidate_indices, provisional, scored

pd.DataFrame(sim_records).to_csv(f"{OUTPUT_DIR}/perfume_similarity_edges.csv", index=False)

print(f"  Perfume nodes : {G_P.number_of_nodes():,}")
print(f"  Edges (τ={SIM_THRESHOLD}) : {G_P.number_of_edges():,}")
print(f"  Density : {nx.density(G_P):.6f}")

components_P = list(nx.connected_components(G_P))
degrees_P = [d for _, d in G_P.degree()]
print(f"  Connected components : {len(components_P)}")
print(f"  Mean degree          : {safe_mean(degrees_P):.2f}")

del shared_note_matrix, shared_acc_matrix
gc.collect()


# ── 5. Centrality on G_N ──────────────────────────────────────────────────────
print("\nComputing centrality on G_N...")

degree_cent_N = nx.degree_centrality(G_N)

if APPROX_BETWEENNESS_K is not None and APPROX_BETWEENNESS_K < G_N.number_of_nodes():
    between_cent_N = nx.betweenness_centrality(G_N, normalized=True, k=APPROX_BETWEENNESS_K, seed=42)
else:
    between_cent_N = nx.betweenness_centrality(G_N, normalized=True)

pagerank_N = nx.pagerank(G_N, weight="weight")

top_degree = sorted(degree_cent_N.items(), key=lambda x: x[1], reverse=True)[:10]
top_between = sorted(between_cent_N.items(), key=lambda x: x[1], reverse=True)[:10]
top_pagerank = sorted(pagerank_N.items(), key=lambda x: x[1], reverse=True)[:10]

print("  Top-10 notes by degree centrality:")
for note, val in top_degree:
    print(f"    {note:<30} {val:.4f}")

print("  Top-10 notes by betweenness centrality:")
for note, val in top_between:
    print(f"    {note:<30} {val:.6f}")

print("  Top-10 notes by PageRank:")
for note, val in top_pagerank:
    print(f"    {note:<30} {val:.6f}")


# ── 6. Top-k Recommendations via Personalized PageRank ────────────────────────
print("\nGenerating top-k recommendations...")


def recommend_ppr(query_perfume_id, G, k=TOP_K):
    """
    Personalized PageRank seeded on query_perfume_id.
    Returns top-k similar perfumes that have a direct edge to the query.
    """
    if query_perfume_id not in G:
        return []

    personalization = {n: 0.0 for n in G.nodes()}
    personalization[query_perfume_id] = 1.0

    try:
        ppr = nx.pagerank(G, alpha=0.85, personalization=personalization, weight="weight")
    except nx.PowerIterationFailedConvergence:
        ppr = nx.pagerank(
            G,
            alpha=0.85,
            personalization=personalization,
            weight="weight",
            max_iter=200,
        )

    ranked = sorted(
        [
            (pid, score)
            for pid, score in ppr.items()
            if pid != query_perfume_id and G.has_edge(query_perfume_id, pid)
        ],
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked[:k]


def composite_score(pid, ppr_score, query_id, G, weight_ppr=0.5, weight_edge=0.3, weight_rating=0.2):
    """Blend PPR score, direct edge weight, and rating popularity."""
    if not G.has_edge(query_id, pid):
        return 0.0

    edge_w = G[query_id][pid]["jaccard"]
    rating = G.nodes[pid].get("rating", 0) or 0
    votes = G.nodes[pid].get("votes", 0) or 0

    norm_rating = min(float(rating), 5.0) / 5.0
    norm_votes = np.log1p(float(votes)) / np.log1p(30000)
    popularity = 0.5 * norm_rating + 0.5 * norm_votes

    return weight_ppr * ppr_score + weight_edge * edge_w + weight_rating * popularity


sample_queries = list(all_perfume_ids[:5])
rec_records = []

for qid in sample_queries:
    recs = recommend_ppr(qid, G_P, k=TOP_K)
    q_label = G_P.nodes[qid].get("label", qid)
    q_brand = G_P.nodes[qid].get("brand", "")

    for rank, (pid, ppr_score) in enumerate(recs, 1):
        comp = composite_score(pid, ppr_score, qid, G_P)
        rec_records.append(
            {
                "query_id": qid,
                "query_perfume": q_label,
                "query_brand": q_brand,
                "rank": rank,
                "rec_id": pid,
                "rec_perfume": G_P.nodes[pid].get("label", pid),
                "rec_brand": G_P.nodes[pid].get("brand", ""),
                "ppr_score": round(ppr_score, 6),
                "composite_score": round(comp, 6),
                "jaccard": round(G_P[qid][pid]["jaccard"], 4) if G_P.has_edge(qid, pid) else 0,
                "shared_notes": int(G_P[qid][pid]["shared_notes"]) if G_P.has_edge(qid, pid) else 0,
            }
        )

rec_df = pd.DataFrame(rec_records)
rec_df.to_csv(f"{OUTPUT_DIR}/top_k_recommendations.csv", index=False)
print(f"  Saved {len(rec_df)} recommendation rows for {len(sample_queries)} query perfumes.")


# ── 7. Summary Stats Report ────────────────────────────────────────────────────
stats_lines = [
    "=" * 60,
    "FRAGRANCE DNA — GRAPH STATISTICS REPORT",
    "=" * 60,
    "",
    "── G_B: Perfume-Note Bipartite Graph ──",
    f"  Perfume nodes : {len(perfume_nodes):,}",
    f"  Note nodes    : {len(note_nodes):,}",
    f"  Total edges   : {G_B.number_of_edges():,}",
    f"  Density       : {nx.density(G_B):.6f}",
    "",
    "── G_N: Note Co-occurrence Graph ──",
    f"  Nodes (notes) : {G_N.number_of_nodes():,}",
    f"  Edges         : {G_N.number_of_edges():,}",
    f"  Density       : {nx.density(G_N):.6f}",
    f"  Conn. components : {len(components_N)}",
    f"  Largest component: {max(len(c) for c in components_N)} nodes",
    f"  Mean degree   : {safe_mean(degrees_N):.2f}",
    f"  Max degree    : {max(degrees_N) if degrees_N else 0}",
    "",
    "── G_P: Perfume Similarity Graph ──",
    f"  Threshold τ   : {SIM_THRESHOLD}",
    f"  Nodes         : {G_P.number_of_nodes():,}",
    f"  Edges         : {G_P.number_of_edges():,}",
    f"  Density       : {nx.density(G_P):.6f}",
    f"  Conn. components : {len(components_P)}",
    f"  Largest component: {max(len(c) for c in components_P)} nodes",
    f"  Mean degree   : {safe_mean(degrees_P):.2f}",
    f"  Max degree    : {max(degrees_P) if degrees_P else 0}",
    "",
    "── Top-10 Notes by Degree Centrality ──",
    *[f"  {note:<30} {val:.4f}" for note, val in top_degree],
    "",
    "── Top-10 Notes by Betweenness Centrality ──",
    *[f"  {note:<30} {val:.6f}" for note, val in top_between],
    "",
    "── Top-10 Notes by PageRank ──",
    *[f"  {note:<30} {val:.6f}" for note, val in top_pagerank],
    "",
    "=" * 60,
]

stats_text = "\n".join(stats_lines)
print("\n" + stats_text)

with open(f"{OUTPUT_DIR}/graph_stats.txt", "w", encoding="utf-8") as f:
    f.write(stats_text)


# ── 8. Community Detection ────────────────────────────────────────────────────
print("\nRunning community detection...")

partition_N = community_louvain.best_partition(G_N, weight="weight", random_state=42)
modularity_N = community_louvain.modularity(partition_N, G_N, weight="weight")

note_community_records = []
comm_to_notes = defaultdict(list)
for note, comm_id in partition_N.items():
    comm_to_notes[comm_id].append(note)

for comm_id, members in sorted(comm_to_notes.items(), key=lambda x: -len(x[1])):
    subgraph = G_N.subgraph(members)
    local_degree = dict(subgraph.degree(weight="weight"))
    top_notes = sorted(local_degree, key=local_degree.get, reverse=True)[:5]

    note_community_records.append(
        {
            "graph": "G_N",
            "community_id": comm_id,
            "community_size": len(members),
            "representative_nodes": " | ".join(top_notes),
            "top_accords": "",
            "modularity": round(modularity_N, 4),
        }
    )

largest_cc_P = max(nx.connected_components(G_P), key=len)
G_P_lcc = G_P.subgraph(largest_cc_P).copy()

partition_P = community_louvain.best_partition(G_P_lcc, weight="weight", random_state=42)
modularity_P = community_louvain.modularity(partition_P, G_P_lcc, weight="weight")

perf_community_records = []
comm_to_perfumes = defaultdict(list)
for pid, comm_id in partition_P.items():
    comm_to_perfumes[comm_id].append(pid)

rating_lookup = perfumes.set_index("perfume_id")["rating_value"].to_dict()
label_lookup = perfumes.set_index("perfume_id")["perfume"].to_dict()
brand_lookup = perfumes.set_index("perfume_id")["brand"].to_dict()

for comm_id, members in sorted(comm_to_perfumes.items(), key=lambda x: -len(x[1])):
    sorted_members = sorted(
        members,
        key=lambda p: float(rating_lookup.get(p, 0) or 0),
        reverse=True,
    )[:5]
    rep_labels = [f"{label_lookup.get(p, p)} ({brand_lookup.get(p, '')})" for p in sorted_members]

    accord_counter = defaultdict(int)
    for p in members:
        for acc in perfume_to_accords.get(p, []):
            accord_counter[acc] += 1
    top_accords = sorted(accord_counter, key=accord_counter.get, reverse=True)[:3]

    perf_community_records.append(
        {
            "graph": "G_P",
            "community_id": comm_id,
            "community_size": len(members),
            "representative_nodes": " | ".join(rep_labels),
            "top_accords": " | ".join(top_accords),
            "modularity": round(modularity_P, 4),
        }
    )

communities_df = pd.DataFrame(note_community_records + perf_community_records)
communities_df.to_csv(f"{OUTPUT_DIR}/communities.csv", index=False)
print(f"  Saved communities.csv ({len(communities_df)} rows)")


# ── 9. Export graph_stats.csv ────────────────────────────────────────────────
print("\nExporting graph_stats.csv...")

cc_N = nx.average_clustering(G_N, weight="weight")
cc_P = nx.average_clustering(
    G_P_lcc.subgraph(list(G_P_lcc.nodes())[:5000]),
    weight="weight",
)

graph_stats_rows = [
    {
        "graph": "G_B (bipartite)",
        "nodes": G_B.number_of_nodes(),
        "edges": G_B.number_of_edges(),
        "density": round(nx.density(G_B), 6),
        "avg_degree": round(safe_mean([d for _, d in G_B.degree()]), 2),
        "connected_components": nx.number_connected_components(G_B),
        "largest_component_nodes": max(len(c) for c in nx.connected_components(G_B)),
        "avg_clustering_coeff": "N/A (bipartite)",
    },
    {
        "graph": "G_N (note co-occurrence)",
        "nodes": G_N.number_of_nodes(),
        "edges": G_N.number_of_edges(),
        "density": round(nx.density(G_N), 6),
        "avg_degree": round(safe_mean(degrees_N), 2),
        "connected_components": len(components_N),
        "largest_component_nodes": max(len(c) for c in components_N),
        "avg_clustering_coeff": round(cc_N, 4),
    },
    {
        "graph": f"G_P (perfume similarity, τ={SIM_THRESHOLD})",
        "nodes": G_P.number_of_nodes(),
        "edges": G_P.number_of_edges(),
        "density": round(nx.density(G_P), 6),
        "avg_degree": round(safe_mean(degrees_P), 2),
        "connected_components": len(components_P),
        "largest_component_nodes": max(len(c) for c in components_P),
        "avg_clustering_coeff": round(cc_P, 4),
    },
]

pd.DataFrame(graph_stats_rows).to_csv(f"{OUTPUT_DIR}/graph_stats.csv", index=False)
print("  Saved graph_stats.csv")


# ── 10. Export top_central_nodes.csv ─────────────────────────────────────────
print("\nExporting top_central_nodes.csv...")

TOP_N = 20
central_records = []

top_deg_N = sorted(degree_cent_N.items(), key=lambda x: x[1], reverse=True)[:TOP_N]
top_btw_N = sorted(between_cent_N.items(), key=lambda x: x[1], reverse=True)[:TOP_N]
top_pr_N = sorted(pagerank_N.items(), key=lambda x: x[1], reverse=True)[:TOP_N]

for note in set(n for n, _ in top_deg_N + top_btw_N + top_pr_N):
    central_records.append(
        {
            "graph": "G_N",
            "node": note,
            "node_type": "note",
            "degree_centrality": round(degree_cent_N.get(note, 0), 6),
            "betweenness_centrality": round(between_cent_N.get(note, 0), 6),
            "pagerank": round(pagerank_N.get(note, 0), 6),
        }
    )

degree_cent_P = nx.degree_centrality(G_P)
pagerank_P = nx.pagerank(G_P, weight="weight")

top_deg_P = sorted(degree_cent_P.items(), key=lambda x: x[1], reverse=True)[:TOP_N]
top_pr_P = sorted(pagerank_P.items(), key=lambda x: x[1], reverse=True)[:TOP_N]

for pid in set(p for p, _ in top_deg_P + top_pr_P):
    central_records.append(
        {
            "graph": "G_P",
            "node": label_lookup.get(pid, pid),
            "node_type": "perfume",
            "degree_centrality": round(degree_cent_P.get(pid, 0), 6),
            "betweenness_centrality": "",
            "pagerank": round(pagerank_P.get(pid, 0), 6),
        }
    )

central_df = pd.DataFrame(central_records)
central_df.sort_values(["graph", "degree_centrality"], ascending=[True, False], inplace=True)
central_df.to_csv(f"{OUTPUT_DIR}/top_central_nodes.csv", index=False)
print(f"  Saved top_central_nodes.csv ({len(central_df)} rows)")


# ── 11. Export recommendations.csv ───────────────────────────────────────────
print("\nExporting recommendations.csv...")

SAMPLE_QUERIES = 20
sample_queries_final = list(all_perfume_ids[:SAMPLE_QUERIES])
final_rec_records = []

for qid in sample_queries_final:
    recs = recommend_ppr(qid, G_P, k=TOP_K)
    q_label = label_lookup.get(qid, qid)
    q_brand = brand_lookup.get(qid, "")

    for rank, (pid, ppr_score) in enumerate(recs, 1):
        comp = composite_score(pid, ppr_score, qid, G_P)
        edge_data = G_P[qid][pid] if G_P.has_edge(qid, pid) else {}
        final_rec_records.append(
            {
                "query_id": qid,
                "query_perfume": q_label,
                "query_brand": q_brand,
                "rank": rank,
                "rec_id": pid,
                "rec_perfume": label_lookup.get(pid, pid),
                "rec_brand": brand_lookup.get(pid, ""),
                "similarity_score": round(edge_data.get("combined_score", 0), 4),
                "note_jaccard": round(edge_data.get("jaccard", 0), 4),
                "accord_jaccard": round(edge_data.get("accord_jaccard", 0), 4),
                "weighted_overlap": round(edge_data.get("weighted_overlap", 0), 4),
                "shared_notes": int(edge_data.get("shared_notes", 0)),
                "shared_accords": int(edge_data.get("shared_accords", 0)),
                "ppr_score": round(ppr_score, 6),
                "composite_score": round(comp, 6),
            }
        )

final_rec_df = pd.DataFrame(final_rec_records)
final_rec_df.to_csv(f"{OUTPUT_DIR}/recommendations.csv", index=False)
print(f"  Saved recommendations.csv ({len(final_rec_df)} rows, {SAMPLE_QUERIES} query perfumes)")


print(f"\nAll outputs saved to '{OUTPUT_DIR}/'")
print("Done.")