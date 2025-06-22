# Quantum Chaos in Billiards

**Implementation and reproduction of results from "Quantum chaos in billiards" research paper**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Physics](https://img.shields.io/badge/field-quantum%20physics-brightgreen.svg)](https://en.wikipedia.org/wiki/Quantum_chaos)

## 📖 Overview

This project explores the fascinating transition from classical chaos to quantum mechanics in billiard systems, implementing and reproducing the key results from Bäcker's seminal work on [**"Quantum chaos in billiards"**](https://www.physik.tu-dresden.de/~baecker/papers/qc.pdf).

**Key Question**: *How do quantum mechanical systems exhibit signatures of classical chaos despite their linear evolution governed by the Schrödinger equation?*

We investigate cardioid billiards defined by `ρ(φ) = 1 + ε cos(φ)`, where the parameter `ε` controls the transition from integrable (circular, ε=0) to fully chaotic (cardioid, ε=1) dynamics.

## 🎯 Objectives

- **Classical Analysis**: Compute trajectories, Poincaré sections, and Lyapunov exponents
- **Quantum Analysis**: Calculate eigenvalues and eigenfunctions using the Boundary Integral Method (BIM)
- **Statistical Comparison**: Analyze level spacing distributions and compare with Random Matrix Theory predictions
- **Ergodicity Studies**: Investigate the quantum-classical correspondence in chaotic systems

## 🔬 Methodology

### Classical Billiards
- **Trajectory computation** with elastic reflections using Brent's root-finding algorithm
- **Lyapunov exponent calculation** via trajectory perturbation techniques
- **Poincaré section analysis** for phase space visualization

### Quantum Billiards
- **Boundary Integral Method (BIM)** for eigenvalue computation
- **Helmholtz equation** solution with Dirichlet boundary conditions
- **Green's function approach** using Hankel functions
- **Eigenfunction reconstruction** throughout the billiard domain

## 📊 Key Results

### Transition to Chaos
- **Classical**: Gradual transition with critical value ε ≈ 0.5 (convexity threshold)
- **Quantum**: Sharp transition at ε ≈ 0.01 (breakdown of exact integrability)

### Level Spacing Statistics
| System Type | Distribution | R² Fit | Physical Interpretation |
|-------------|--------------|--------|------------------------|
| Circle (ε=0) | Poisson | 0.996 | Integrable, no level repulsion |
| Cardioid (ε=1) | GOE (Wigner) | 0.894 | Chaotic, strong level repulsion |

### Validation Tests
- **Weyl's Law verification**: R² = 0.997 (excellent agreement with asymptotic eigenvalue density)
- **Spectral statistics**: Smooth crossover from Poisson to GOE as ε increases
- **Ergodicity analysis**: Transition from localized to uniformly distributed eigenstates

## 🔧 Technical Implementation

### Dependencies
```bash
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.0
