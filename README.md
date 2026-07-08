# Fragrance DNA

Graph-based analysis of perfume scent profiles, note communities, and fragrance recommendation.

## Overview

Fragrance DNA is a network analysis project that models perfumes, scent notes, and fragrance relationships as graphs. The goal is to study how perfumes are connected through shared scent profiles, identify structurally important notes, detect note communities, and build an explainable fragrance recommendation approach.

This project was completed for **EECS 4414: Social Networks** at York University.

## Key Features

- Constructed graph representations of perfumes and fragrance notes
- Analyzed perfume similarity using graph-based methods
- Identified central scent notes using network centrality measures
- Detected scent-note communities using community detection
- Built a recommendation pipeline using graph-based similarity and propagation methods
- Visualized fragrance relationships, note clusters, and network structure

## Project Structure

```text
fragrance-dna/
├── data/              # Dataset files and processed data
├── notebooks/         # Jupyter notebooks for analysis and experimentation
├── src/               # Source code and reusable scripts
├── figures/           # Graphs, plots, and visualizations
├── reports/           # Proposal, progress report, and final report
├── presentation/      # Final presentation slides
├── README.md
├── LICENSE
└── .gitignore
