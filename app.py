"""
AKT-Scan AI
A High-Throughput Machine Learning Platform for SMILES-Based Bioactivity Screening
and Drug-Likeness Profiling Targeting AKT1

Expected files in the same directory as this app:
    - LightGBM.pkl
    - selected_feature_indices.npy

Run:
    streamlit run app.py
"""

from __future__ import annotations

import base64
import html
import hashlib
import io
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    from supabase import create_client
except Exception:
    create_client = None

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen, Descriptors, Draw, Lipinski, QED, rdMolDescriptors
try:
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
except Exception:
    FilterCatalog = None
    FilterCatalogParams = None


# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AKT-Scan AI",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
APP_NAME = "AKT-Scan AI"
APP_TITLE = (
    "A High-Throughput Machine Learning Platform for SMILES-Based Bioactivity "
    "Screening and Drug-Likeness Profiling Targeting AKT1"
)
DEVELOPERS = "Sheikh Sunzid Ahmed and M. Oliur Rahman"
AFFILIATION = (
    "Plant Taxonomy and Ethnobotany Laboratory · Department of Botany · "
    "University of Dhaka"
)

MODEL_FILE = "LightGBM.pkl"
FEATURE_INDEX_FILE = "selected_feature_indices.npy"
MAX_BATCH_MOLECULES = 3000
FINGERPRINT_BITS = 2048
FINGERPRINT_RADIUS = 2
DEFAULT_THRESHOLD = 0.50
MOLECULE_IMAGE_BACKGROUND_RGB = (245, 247, 242)
TOP_SIMILARITY_MATCHES = 3
ACTIVE_DATABASE_TABLE = "predicted_active_compounds"
AD_IN_DOMAIN_THRESHOLD = 0.45
AD_BORDERLINE_THRESHOLD = 0.25
HIGH_CONFIDENCE_MARGIN = 0.25
MODERATE_CONFIDENCE_MARGIN = 0.10

CV_METRICS = {
    "Accuracy": 0.9044,
    "F1 Score": 0.8613,
    "AUC": 0.9506,
    "MCC": 0.7894,
    "Sensitivity": 0.8908,
    "Specificity": 0.9111,
}

EXTERNAL_METRICS = {
    "Accuracy": 0.7400,
    "F1 Score": 0.6579,
    "AUC": 0.8186,
    "MCC": 0.5472,
    "Sensitivity": 0.5000,
    "Specificity": 0.9800,
}

REFERENCE_INHIBITORS = [
    {
        "name": "Vevorisertib",
        "type": "AKT inhibitor",
        "reference_note": "Model-consistent AKT inhibitor scaffold",
        "smiles": "CC(=O)N(C)C1CCN(CC1)C2=CC=CC(=C2)C3=NC4=C(C=C3)N=C(N4C5=CC=C(C=C5)C6(CCC6)N)C7=C(N=CC=C7)N",
    },
    {
        "name": "GSK690693",
        "type": "Pan-AKT inhibitor",
        "reference_note": "Model-consistent AKT inhibitor scaffold",
        "smiles": "CCN1C2=C(C(=NC=C2OC[C@H]3CCCNC3)C#CC(C)(C)O)N=C1C4=NON=C4N",
    },
    {
        "name": "Ipatasertib",
        "type": "AKT inhibitor",
        "reference_note": "Model-consistent AKT inhibitor scaffold",
        "smiles": "C[C@@H]1C[C@H](C2=C1C(=NC=N2)N3CCN(CC3)C(=O)[C@H](CNC(C)C)C4=CC=C(C=C4)Cl)O",
    },
    {
        "name": "Uprosertib",
        "type": "AKT inhibitor",
        "reference_note": "Model-consistent AKT inhibitor scaffold",
        "smiles": "CN1C(=C(C=N1)Cl)C2=C(OC(=C2)C(=O)N[C@@H](CC3=CC(=C(C=C3)F)F)CN)Cl",
    },
    {
        "name": "AT7867",
        "type": "AKT inhibitor",
        "reference_note": "Model-consistent AKT inhibitor scaffold",
        "smiles": "C1CNCCC1(C2=CC=C(C=C2)C3=CNN=C3)C4=CC=C(C=C4)Cl",
    },
]


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --akt-bg: #f5f7f2;
            --akt-panel: #ffffff;
            --akt-green: #1f4d3a;
            --akt-green-2: #2f6f53;
            --akt-sage: #dfe8d8;
            --akt-gold: #b78a3b;
            --akt-ink: #1d2b25;
            --akt-muted: #63766e;
            --akt-red: #8b2f2f;
        }

        .stApp {
            background: linear-gradient(180deg, #f7f9f3 0%, #eef4ec 100%);
            color: var(--akt-ink);
        }

        .hero {
            padding: 1.25rem 1.55rem;
            border-radius: 28px;
            background: linear-gradient(135deg, #fbfaf2 0%, #eef6ec 58%, #d8c289 100%);
            color: #17392d;
            border: 1px solid rgba(31, 77, 58, 0.18);
            box-shadow: 0 10px 24px rgba(22, 57, 45, 0.16);
            margin-bottom: 0.65rem;
        }

        .hero h1 {
            font-size: 2.35rem;
            line-height: 1.05;
            margin: 0 0 0.35rem 0;
            letter-spacing: -0.04em;
            color: #0b2f26;
            text-shadow: none;
        }

        .hero p {
            font-size: 0.98rem;
            max-width: 1100px;
            color: #36564a;
            font-weight: 500;
            margin-bottom: 0;
        }

        .section-card {
            background: linear-gradient(135deg, #edf5ea 0%, #fffdf4 100%);
            border: 1px solid rgba(31, 77, 58, 0.22);
            border-left: 7px solid #1f4d3a;
            border-radius: 18px;
            padding: 0.9rem 1.05rem;
            box-shadow: 0 7px 16px rgba(31, 77, 58, 0.08);
            margin-bottom: 0.65rem;
        }

        .section-card h3 {
            color: #12382b;
            margin-top: 0;
            margin-bottom: 0.45rem;
        }

        .similarity-card {
            background: linear-gradient(135deg, #fffdf4 0%, #edf5ea 100%);
            border: 1px solid rgba(183, 138, 59, 0.28);
            border-left: 7px solid #b78a3b;
            border-radius: 18px;
            padding: 0.95rem 1.1rem;
            box-shadow: 0 7px 16px rgba(31, 77, 58, 0.08);
            margin-bottom: 0.8rem;
        }

        .similarity-card h3 {
            color: #4b3718;
            margin-top: 0;
            margin-bottom: 0.45rem;
        }

        .performance-card {
            background: linear-gradient(135deg, #eef6ec 0%, #fff9e8 100%);
            border-left-color: #b78a3b;
        }

        .metric-card {
            background: #ffffff;
            border: 1px solid rgba(31, 77, 58, 0.10);
            border-radius: 20px;
            padding: 1rem;
            box-shadow: 0 8px 18px rgba(31, 77, 58, 0.07);
            text-align: center;
        }

        .metric-card .value {
            font-size: 1.65rem;
            font-weight: 800;
            color: var(--akt-green);
        }

        .metric-card .label {
            font-size: 0.85rem;
            color: var(--akt-muted);
            letter-spacing: 0.02em;
        }

        .prediction-active {
            background: linear-gradient(135deg, #e4f5ea, #ffffff);
            border-left: 8px solid #2f7d4f;
            border-radius: 20px;
            padding: 1.2rem;
        }

        .prediction-inactive {
            background: linear-gradient(135deg, #f9e9e5, #ffffff);
            border-left: 8px solid #9c3f32;
            border-radius: 20px;
            padding: 1.2rem;
        }

        .footer {
            margin-top: 2.5rem;
            padding: 1.25rem 1rem;
            border-top: 1px solid rgba(31, 77, 58, 0.15);
            color: #50645b;
            font-size: 0.92rem;
            text-align: center;
        }

        div[data-testid="stMetricValue"] {
            color: #1f4d3a;
        }

        .stProgress > div > div > div > div {
            background-image: linear-gradient(90deg, #1f4d3a, #b78a3b);
        }

        .small-note {
            color: #63766e;
            font-size: 0.9rem;
        }

        .centered-molecule-image {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            width: 100%;
            margin: 0.25rem 0 1rem 0;
        }

        .centered-molecule-image img {
            max-width: 100%;
            height: auto;
            border-radius: 18px;
            background: #f5f7f2;
        }

        .centered-molecule-caption {
            color: #63766e;
            font-size: 0.9rem;
            text-align: center;
            margin-top: 0.55rem;
        }
        .stButton > button,
        .stDownloadButton > button {
            background: linear-gradient(90deg, #1f4d3a, #2f6f53) !important;
            color: #ffffff !important;
            border: 1px solid rgba(31, 77, 58, 0.35) !important;
            border-radius: 14px !important;
            font-weight: 700 !important;
            box-shadow: 0 6px 14px rgba(31, 77, 58, 0.16) !important;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            background: linear-gradient(90deg, #17392d, #285f48) !important;
            color: #ffffff !important;
            border: 1px solid rgba(183, 138, 59, 0.65) !important;
        }

        .stButton > button:focus,
        .stDownloadButton > button:focus {
            color: #ffffff !important;
            border: 1px solid rgba(183, 138, 59, 0.85) !important;
            box-shadow: 0 0 0 0.18rem rgba(183, 138, 59, 0.25) !important;
        }

        .static-table {
            width: 100%;
            border-collapse: collapse;
            background: #ffffff;
            border-radius: 14px;
            overflow: hidden;
            box-shadow: 0 6px 14px rgba(31, 77, 58, 0.06);
            margin-bottom: 0.9rem;
        }

        .static-table th {
            background: #f3f6f1;
            color: #4f6259;
            text-align: left;
            padding: 0.7rem 0.75rem;
            font-size: 0.88rem;
            border-bottom: 1px solid rgba(31, 77, 58, 0.12);
        }

        .static-table td {
            padding: 0.68rem 0.75rem;
            border-bottom: 1px solid rgba(31, 77, 58, 0.08);
            color: #1d2b25;
            font-size: 0.9rem;
            vertical-align: top;
        }

        .static-table tr:last-child td {
            border-bottom: none;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Data structures and loading
# -----------------------------------------------------------------------------
@dataclass
class PredictionResult:
    smiles: str
    canonical_smiles: Optional[str]
    valid: bool
    prediction_label: Optional[str]
    predicted_class: Optional[int]
    probability_active: Optional[float]
    probability_inactive: Optional[float]
    error: Optional[str]
    descriptors: Dict[str, Optional[float]]
    similarity: Dict[str, Optional[object]]
    confidence: Dict[str, Optional[object]]
    applicability_domain: Dict[str, Optional[object]]
    structural_alerts: Dict[str, Optional[object]]


@st.cache_resource(show_spinner=False)
def load_model_and_features(
    model_path: str = MODEL_FILE,
    feature_path: str = FEATURE_INDEX_FILE,
):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file '{model_path}' was not found. Place it in the same folder as app.py."
        )
    if not os.path.exists(feature_path):
        raise FileNotFoundError(
            f"Feature index file '{feature_path}' was not found. Place it in the same folder as app.py."
        )

    model = joblib.load(model_path)
    selected_feature_indices = np.load(feature_path)
    selected_feature_indices = np.asarray(selected_feature_indices, dtype=int)

    if selected_feature_indices.ndim != 1:
        raise ValueError("selected_feature_indices.npy must contain a 1-dimensional array.")
    if selected_feature_indices.max(initial=0) >= FINGERPRINT_BITS:
        raise ValueError(
            f"Selected feature index exceeds Morgan fingerprint size of {FINGERPRINT_BITS} bits."
        )

    return model, selected_feature_indices


@st.cache_resource(show_spinner=False)
def load_reference_inhibitors() -> List[Dict[str, object]]:
    references = []
    for ref in REFERENCE_INHIBITORS:
        mol = smiles_to_mol(ref["smiles"])
        if mol is None:
            continue
        bitvect = mol_to_bitvect(mol)
        references.append(
            {
                **ref,
                "mol": mol,
                "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
                "fingerprint": bitvect,
            }
        )
    return references


@st.cache_resource(show_spinner=False)
def load_structural_alert_catalogs():
    catalogs = {}

    if FilterCatalog is None or FilterCatalogParams is None:
        return catalogs

    try:
        pains_params = FilterCatalogParams()
        pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
        pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
        pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
        catalogs["PAINS"] = FilterCatalog(pains_params)
    except Exception:
        pass

    try:
        brenk_params = FilterCatalogParams()
        brenk_params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
        catalogs["Brenk"] = FilterCatalog(brenk_params)
    except Exception:
        pass

    return catalogs


def confidence_template() -> Dict[str, Optional[object]]:
    return {
        "Prediction_Confidence_Margin": None,
        "Prediction_Confidence_Label": "Not available",
    }


def calculate_prediction_confidence(probability_active: Optional[float], threshold: float) -> Dict[str, Optional[object]]:
    if probability_active is None:
        return confidence_template()

    margin = abs(float(probability_active) - float(threshold))

    if margin >= HIGH_CONFIDENCE_MARGIN:
        label = "High confidence"
    elif margin >= MODERATE_CONFIDENCE_MARGIN:
        label = "Moderate confidence"
    else:
        label = "Low confidence / near threshold"

    return {
        "Prediction_Confidence_Margin": safe_float(margin),
        "Prediction_Confidence_Label": label,
    }


def applicability_domain_template() -> Dict[str, Optional[object]]:
    return {
        "Applicability_Domain_Status": "Not available",
        "Applicability_Domain_Reliability": "Not available",
        "Applicability_Domain_Score": None,
    }


def calculate_applicability_domain(similarity: Dict[str, Optional[object]]) -> Dict[str, Optional[object]]:
    score = similarity.get("Max_Tanimoto_Similarity")

    if score is None:
        return applicability_domain_template()

    score = float(score)

    if score >= AD_IN_DOMAIN_THRESHOLD:
        status = "Inside reference-scaffold domain"
        reliability = "Higher structural support"
    elif score >= AD_BORDERLINE_THRESHOLD:
        status = "Borderline reference-scaffold domain"
        reliability = "Moderate structural support"
    else:
        status = "Outside reference-scaffold domain"
        reliability = "Lower structural support"

    return {
        "Applicability_Domain_Status": status,
        "Applicability_Domain_Reliability": reliability,
        "Applicability_Domain_Score": safe_float(score),
    }


def structural_alert_template() -> Dict[str, Optional[object]]:
    return {
        "Structural_Alert_Flag": "Not available",
        "Structural_Alert_Count": None,
        "PAINS_Alert_Count": None,
        "Brenk_Alert_Count": None,
        "Structural_Alert_Summary": "Not available",
    }


def calculate_structural_alerts(mol: Optional[Chem.Mol]) -> Dict[str, Optional[object]]:
    if mol is None:
        return structural_alert_template()

    catalogs = load_structural_alert_catalogs()
    alert_labels = []

    for source_name, catalog in catalogs.items():
        try:
            matches = catalog.GetMatches(mol)
            for match in matches:
                alert_labels.append(f"{source_name}: {match.GetDescription()}")
        except Exception:
            continue

    unique_alerts = []
    seen = set()
    for label in alert_labels:
        if label not in seen:
            seen.add(label)
            unique_alerts.append(label)

    pains_count = sum(1 for alert in unique_alerts if alert.startswith("PAINS:"))
    brenk_count = sum(1 for alert in unique_alerts if alert.startswith("Brenk:"))

    if not catalogs:
        return {
            "Structural_Alert_Flag": "Not available",
            "Structural_Alert_Count": None,
            "PAINS_Alert_Count": None,
            "Brenk_Alert_Count": None,
            "Structural_Alert_Summary": "RDKit PAINS/Brenk catalogs unavailable",
        }

    if unique_alerts:
        summary = "; ".join(unique_alerts[:8])
        if len(unique_alerts) > 8:
            summary += f"; +{len(unique_alerts) - 8} more"
        flag = "Yes"
    else:
        summary = "No PAINS/Brenk alerts detected"
        flag = "No"

    return {
        "Structural_Alert_Flag": flag,
        "Structural_Alert_Count": len(unique_alerts),
        "PAINS_Alert_Count": pains_count,
        "Brenk_Alert_Count": brenk_count,
        "Structural_Alert_Summary": summary,
    }


# -----------------------------------------------------------------------------
# Chemistry utilities
# -----------------------------------------------------------------------------
def smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
    if smiles is None:
        return None
    smiles = str(smiles).strip()
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    return mol


def mol_to_bitvect(mol: Chem.Mol):
    return AllChem.GetMorganFingerprintAsBitVect(
        mol, FINGERPRINT_RADIUS, nBits=FINGERPRINT_BITS
    )


def mol_to_fingerprint_array(mol: Chem.Mol) -> np.ndarray:
    bitvect = mol_to_bitvect(mol)
    arr = np.zeros((FINGERPRINT_BITS,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bitvect, arr)
    return arr.astype(np.float32)


def selected_feature_matrix(mols: Iterable[Chem.Mol], selected_indices: np.ndarray) -> np.ndarray:
    fps = [mol_to_fingerprint_array(mol) for mol in mols]
    X = np.vstack(fps)
    return X[:, selected_indices]


def safe_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        value = float(value)
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    except Exception:
        return None


def calculate_adme_qed_descriptors(mol: Optional[Chem.Mol]) -> Dict[str, Optional[float]]:
    if mol is None:
        return descriptor_template()

    try:
        mol_wt = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        rot_bonds = Lipinski.NumRotatableBonds(mol)
        heavy_atoms = mol.GetNumHeavyAtoms()
        aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        aliphatic_rings = rdMolDescriptors.CalcNumAliphaticRings(mol)
        ring_count = rdMolDescriptors.CalcNumRings(mol)
        hetero_atoms = rdMolDescriptors.CalcNumHeteroatoms(mol)
        fraction_csp3 = rdMolDescriptors.CalcFractionCSP3(mol)
        formal_charge = Chem.GetFormalCharge(mol)
        molar_refractivity = Crippen.MolMR(mol)
        qed_score = QED.qed(mol)

        lipinski_violations = int(mol_wt > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10)
        ghose_violations = int(not (160 <= mol_wt <= 480)) + int(not (-0.4 <= logp <= 5.6)) + int(
            not (40 <= molar_refractivity <= 130)
        ) + int(not (20 <= heavy_atoms <= 70))
        veber_violations = int(rot_bonds > 10) + int(tpsa > 140)
        egan_violations = int(tpsa > 131.6) + int(logp > 5.88)
        muegge_violations = int(not (200 <= mol_wt <= 600)) + int(not (-2 <= logp <= 5)) + int(tpsa > 150) + int(
            ring_count > 7
        ) + int(heavy_atoms < 10) + int(hbd > 5) + int(hba > 10) + int(rot_bonds > 15)

        lead_like = int((250 <= mol_wt <= 350) and (logp <= 3.5) and (rot_bonds <= 7))
        bioavailability_score = approximate_bioavailability_score(mol_wt, logp, tpsa, hbd, hba, rot_bonds)

        return {
            "Molecular Weight": safe_float(mol_wt),
            "LogP (Crippen)": safe_float(logp),
            "TPSA": safe_float(tpsa),
            "H-bond Donors": safe_float(hbd),
            "H-bond Acceptors": safe_float(hba),
            "Rotatable Bonds": safe_float(rot_bonds),
            "Heavy Atoms": safe_float(heavy_atoms),
            "Heteroatoms": safe_float(hetero_atoms),
            "Aromatic Rings": safe_float(aromatic_rings),
            "Aliphatic Rings": safe_float(aliphatic_rings),
            "Total Rings": safe_float(ring_count),
            "Fraction Csp3": safe_float(fraction_csp3),
            "Formal Charge": safe_float(formal_charge),
            "Molar Refractivity": safe_float(molar_refractivity),
            "QED": safe_float(qed_score),
            "Lipinski Violations": safe_float(lipinski_violations),
            "Lipinski Ro5 Pass": "Yes" if lipinski_violations <= 1 else "No",
            "Veber Violations": safe_float(veber_violations),
            "Veber Pass": "Yes" if veber_violations == 0 else "No",
            "Ghose Violations": safe_float(ghose_violations),
            "Ghose Pass": "Yes" if ghose_violations == 0 else "No",
            "Egan Violations": safe_float(egan_violations),
            "Egan Pass": "Yes" if egan_violations == 0 else "No",
            "Muegge Violations": safe_float(muegge_violations),
            "Muegge Pass": "Yes" if muegge_violations == 0 else "No",
            "Lead-like Approximation": "Yes" if lead_like else "No",
            "Approx. Bioavailability Score": safe_float(bioavailability_score),
        }
    except Exception:
        return descriptor_template()


def descriptor_template() -> Dict[str, Optional[float]]:
    keys = [
        "Molecular Weight", "LogP (Crippen)", "TPSA",
        "H-bond Donors", "H-bond Acceptors", "Rotatable Bonds", "Heavy Atoms",
        "Heteroatoms", "Aromatic Rings", "Aliphatic Rings", "Total Rings",
        "Fraction Csp3", "Formal Charge", "Molar Refractivity", "QED",
        "Lipinski Violations", "Lipinski Ro5 Pass", "Veber Violations", "Veber Pass",
        "Ghose Violations", "Ghose Pass", "Egan Violations", "Egan Pass",
        "Muegge Violations", "Muegge Pass", "Lead-like Approximation",
        "Approx. Bioavailability Score",
    ]
    return {key: None for key in keys}


def approximate_bioavailability_score(
    mol_wt: float, logp: float, tpsa: float, hbd: int, hba: int, rot_bonds: int
) -> float:
    """
    A simple interpretive approximation for display only. It is not a replacement
    for experimental or specialized ADME prediction.
    """
    score = 0.55
    if mol_wt > 500 or logp > 5 or hbd > 5 or hba > 10:
        score -= 0.11
    if tpsa > 140 or rot_bonds > 10:
        score -= 0.11
    if 200 <= mol_wt <= 500 and -1 <= logp <= 5 and tpsa <= 120:
        score += 0.11
    return float(max(0.0, min(1.0, score)))


def interpret_similarity(score: Optional[float]) -> str:
    if score is None:
        return "Not available"
    if score >= 0.85:
        return "Very high scaffold similarity"
    if score >= 0.65:
        return "High scaffold similarity"
    if score >= 0.45:
        return "Moderate scaffold similarity"
    if score >= 0.25:
        return "Low scaffold similarity"
    return "Structurally distinct from references"


def calculate_scaffold_similarity(mol: Optional[Chem.Mol]) -> Dict[str, Optional[object]]:
    if mol is None:
        return similarity_template()

    references = load_reference_inhibitors()
    if not references:
        return similarity_template()

    query_fp = mol_to_bitvect(mol)
    matches = []

    for ref in references:
        score = float(DataStructs.TanimotoSimilarity(query_fp, ref["fingerprint"]))
        matches.append(
            {
                "Reference_Inhibitor": ref["name"],
                "Reference_Type": ref["type"],
                "Reference_Note": ref["reference_note"],
                "Reference_SMILES": ref["smiles"],
                "Tanimoto_Similarity": score,
                "Similarity_Interpretation": interpret_similarity(score),
            }
        )

    matches = sorted(matches, key=lambda x: x["Tanimoto_Similarity"], reverse=True)
    top_matches = matches[:TOP_SIMILARITY_MATCHES]
    nearest = top_matches[0] if top_matches else None

    if nearest is None:
        return similarity_template()

    top_match_text = "; ".join(
        [
            f"{m['Reference_Inhibitor']} ({m['Tanimoto_Similarity']:.4f})"
            for m in top_matches
        ]
    )

    return {
        "Nearest_Reference_Inhibitor": nearest["Reference_Inhibitor"],
        "Nearest_Reference_Type": nearest["Reference_Type"],
        "Max_Tanimoto_Similarity": safe_float(nearest["Tanimoto_Similarity"]),
        "Similarity_Interpretation": nearest["Similarity_Interpretation"],
        "Top_3_Scaffold_Matches": top_match_text,
        "All_Matches": matches,
    }


def similarity_template() -> Dict[str, Optional[object]]:
    return {
        "Nearest_Reference_Inhibitor": None,
        "Nearest_Reference_Type": None,
        "Max_Tanimoto_Similarity": None,
        "Similarity_Interpretation": "Not available",
        "Top_3_Scaffold_Matches": None,
        "All_Matches": [],
    }


def predict_molecules(
    smiles_list: List[str],
    model,
    selected_indices: np.ndarray,
    threshold: float,
    progress_bar=None,
    progress_text=None,
) -> List[PredictionResult]:
    results: List[PredictionResult] = []
    valid_mols: List[Chem.Mol] = []
    valid_positions: List[int] = []

    for i, smi in enumerate(smiles_list):
        mol = smiles_to_mol(smi)
        if mol is None:
            results.append(
                PredictionResult(
                    smiles=str(smi),
                    canonical_smiles=None,
                    valid=False,
                    prediction_label=None,
                    predicted_class=None,
                    probability_active=None,
                    probability_inactive=None,
                    error="Invalid or empty SMILES",
                    descriptors=descriptor_template(),
                    similarity=similarity_template(),
                    confidence=confidence_template(),
                    applicability_domain=applicability_domain_template(),
                    structural_alerts=structural_alert_template(),
                )
            )
        else:
            canonical = Chem.MolToSmiles(mol, canonical=True)
            similarity = calculate_scaffold_similarity(mol)
            results.append(
                PredictionResult(
                    smiles=str(smi),
                    canonical_smiles=canonical,
                    valid=True,
                    prediction_label=None,
                    predicted_class=None,
                    probability_active=None,
                    probability_inactive=None,
                    error=None,
                    descriptors=calculate_adme_qed_descriptors(mol),
                    similarity=similarity,
                    confidence=confidence_template(),
                    applicability_domain=calculate_applicability_domain(similarity),
                    structural_alerts=calculate_structural_alerts(mol),
                )
            )
            valid_mols.append(mol)
            valid_positions.append(i)

    total_valid = len(valid_mols)
    if total_valid == 0:
        return results

    chunk_size = 128
    completed = 0
    for start in range(0, total_valid, chunk_size):
        end = min(start + chunk_size, total_valid)
        mol_chunk = valid_mols[start:end]
        X = selected_feature_matrix(mol_chunk, selected_indices)

        probabilities_active = get_active_probabilities(model, X)
        if probabilities_active is None:
            preds = model.predict(X)
            probabilities_active = np.asarray(preds, dtype=float)
        else:
            preds = (probabilities_active >= threshold).astype(int)

        for local_idx, pos in enumerate(valid_positions[start:end]):
            p_active = float(probabilities_active[local_idx])
            pred_class = int(preds[local_idx])
            pred_class = int(p_active >= threshold)
            results[pos].predicted_class = pred_class
            results[pos].prediction_label = "Active" if pred_class == 1 else "Inactive"
            results[pos].probability_active = p_active
            results[pos].probability_inactive = 1.0 - p_active
            results[pos].confidence = calculate_prediction_confidence(p_active, threshold)

        completed = end
        if progress_bar is not None and progress_text is not None:
            pct = completed / total_valid
            progress_bar.progress(pct)
            progress_text.markdown(
                f"**Predicted {completed:,} of {total_valid:,} valid molecules** · "
                f"{pct * 100:,.2f}% complete"
            )
            time.sleep(0.015)

    return results


def get_active_probabilities(model, X: np.ndarray) -> Optional[np.ndarray]:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            if hasattr(model, "classes_"):
                classes = list(model.classes_)
                if 1 in classes:
                    return np.asarray(proba[:, classes.index(1)], dtype=float)
            return np.asarray(proba[:, 1], dtype=float)
        if proba.ndim == 1:
            return np.asarray(proba, dtype=float)
    return None


def results_to_dataframe(results: List[PredictionResult]) -> pd.DataFrame:
    records = []
    for input_index, r in enumerate(results, start=1):
        base = {
            "Input_Order": input_index,
            "Input_SMILES": r.smiles,
            "Canonical_SMILES": r.canonical_smiles,
            "Valid_SMILES": r.valid,
            "Prediction": r.prediction_label,
            "Predicted_Label": r.predicted_class,
            "Probability_Active": r.probability_active,
            "Probability_Inactive": r.probability_inactive,
            "Prediction_Confidence_Margin": r.confidence.get("Prediction_Confidence_Margin"),
            "Prediction_Confidence_Label": r.confidence.get("Prediction_Confidence_Label"),
            "Applicability_Domain_Status": r.applicability_domain.get("Applicability_Domain_Status"),
            "Applicability_Domain_Reliability": r.applicability_domain.get("Applicability_Domain_Reliability"),
            "Applicability_Domain_Score": r.applicability_domain.get("Applicability_Domain_Score"),
            "Structural_Alert_Flag": r.structural_alerts.get("Structural_Alert_Flag"),
            "Structural_Alert_Count": r.structural_alerts.get("Structural_Alert_Count"),
            "PAINS_Alert_Count": r.structural_alerts.get("PAINS_Alert_Count"),
            "Brenk_Alert_Count": r.structural_alerts.get("Brenk_Alert_Count"),
            "Structural_Alert_Summary": r.structural_alerts.get("Structural_Alert_Summary"),
            "Nearest_Reference_Inhibitor": r.similarity.get("Nearest_Reference_Inhibitor"),
            "Nearest_Reference_Type": r.similarity.get("Nearest_Reference_Type"),
            "Max_Tanimoto_Similarity": r.similarity.get("Max_Tanimoto_Similarity"),
            "Similarity_Interpretation": r.similarity.get("Similarity_Interpretation"),
            "Top_3_Scaffold_Matches": r.similarity.get("Top_3_Scaffold_Matches"),
            "Error": r.error,
        }
        base.update(r.descriptors)
        records.append(base)

    df = pd.DataFrame(records)
    float_cols = df.select_dtypes(include=["float", "float64", "float32"]).columns
    if len(float_cols):
        df[float_cols] = df[float_cols].round(4)
    return df


def format_dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    display_df = df.copy()
    for col in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[col]):
            display_df[col] = display_df[col].map(lambda x: "NA" if pd.isna(x) else f"{float(x):.4f}")
    return display_df


def mol_image_base64(mol: Chem.Mol, size: Tuple[int, int] = (460, 340)) -> str:
    image = Draw.MolToImage(mol, size=size)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def replace_white_background(image: Image.Image, background_rgb: Tuple[int, int, int]) -> Image.Image:
    """Replace RDKit's white canvas with the app background color."""
    rgba_image = image.convert("RGBA")
    arr = np.array(rgba_image)
    white_mask = (arr[:, :, 0] >= 248) & (arr[:, :, 1] >= 248) & (arr[:, :, 2] >= 248)
    arr[white_mask, 0] = background_rgb[0]
    arr[white_mask, 1] = background_rgb[1]
    arr[white_mask, 2] = background_rgb[2]
    arr[white_mask, 3] = 255
    return Image.fromarray(arr, mode="RGBA")


def mol_grid_png_bytes(
    mols: List[Chem.Mol],
    legends: List[str],
    mols_per_row: int = 2,
    sub_img_size: Tuple[int, int] = (420, 320),
) -> bytes:
    image = Draw.MolsToGridImage(
        mols,
        molsPerRow=mols_per_row,
        subImgSize=sub_img_size,
        legends=legends,
    )
    image = replace_white_background(image, MOLECULE_IMAGE_BACKGROUND_RGB)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def render_centered_png_image(image_bytes: bytes, caption: str) -> None:
    image64 = base64.b64encode(image_bytes).decode("utf-8")
    st.markdown(
        f"""
        <div class="centered-molecule-image">
            <img src="data:image/png;base64,{image64}" alt="{html.escape(caption)}" />
            <div class="centered-molecule-caption">{html.escape(caption)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def single_similarity_comparison_png(query_mol: Chem.Mol, result: PredictionResult) -> Optional[bytes]:
    all_matches = result.similarity.get("All_Matches", [])
    if not all_matches:
        return None

    nearest_name = all_matches[0]["Reference_Inhibitor"]
    references = load_reference_inhibitors()
    nearest_ref = next((r for r in references if r["name"] == nearest_name), None)
    if nearest_ref is None:
        return None

    score = result.similarity.get("Max_Tanimoto_Similarity")
    legends = [
        "Query compound",
        f"{nearest_ref['name']} | Tanimoto: {score:.4f}" if score is not None else nearest_ref["name"],
    ]
    return mol_grid_png_bytes([query_mol, nearest_ref["mol"]], legends, mols_per_row=2)






def get_uploaded_csv_dataframe(uploaded_file) -> Optional[pd.DataFrame]:
    if uploaded_file is None:
        return None

    signature = (
        uploaded_file.name,
        uploaded_file.size,
        getattr(uploaded_file, "type", None),
    )

    if (
        st.session_state.uploaded_csv_signature != signature
        or st.session_state.uploaded_csv_df is None
    ):
        uploaded_file.seek(0)
        st.session_state.uploaded_csv_df = pd.read_csv(uploaded_file)
        st.session_state.uploaded_csv_signature = signature

    return st.session_state.uploaded_csv_df


def render_static_csv_preview(df: pd.DataFrame, max_rows: int = 10) -> None:
    if df is None or df.empty:
        return

    preview_df = df.head(max_rows).copy()

    header_cells = ["<th></th>"] + [
        f"<th>{html.escape(str(col))}</th>" for col in preview_df.columns
    ]

    rows = []
    for idx, row in preview_df.iterrows():
        cells = [f"<td>{html.escape(str(idx))}</td>"]
        for col in preview_df.columns:
            value = row[col]
            display_value = "NA" if pd.isna(value) else str(value)
            cells.append(f"<td>{html.escape(display_value)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")

    table_html = f"""
    <div class="section-card">
        <h3>📄 Uploaded CSV Preview</h3>
        <table class="static-table">
            <thead>
                <tr>{''.join(header_cells)}</tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
        <p class="small-note">Showing first {min(len(df), max_rows)} row(s) only.</p>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)


def render_static_priority_table(df: pd.DataFrame) -> None:
    if df.empty:
        return

    columns = [
        "Input_Order",
        "Input_SMILES",
        "Prediction",
        "Probability_Active",
        "Prediction_Confidence_Label",
        "Applicability_Domain_Status",
        "Structural_Alert_Flag",
        "Nearest_Reference_Inhibitor",
        "Max_Tanimoto_Similarity",
        "Similarity_Interpretation",
        "QED",
        "Lipinski Ro5 Pass",
    ]

    existing_columns = [col for col in columns if col in df.columns]

    header_html = "".join(
        f"<th>{html.escape(str(col).replace('_', ' '))}</th>"
        for col in existing_columns
    )

    rows = []
    for _, row in df.iterrows():
        cells = []
        for col in existing_columns:
            value = row[col]
            if pd.isna(value):
                display_value = "NA"
            elif col in {"Probability_Active", "Max_Tanimoto_Similarity", "QED"}:
                display_value = f"{float(value):.4f}"
            else:
                display_value = str(value)

            cells.append(f"<td>{html.escape(display_value)}</td>")

        rows.append("<tr>" + "".join(cells) + "</tr>")

    table_html = f"""
    <table class="static-table">
        <thead>
            <tr>{header_html}</tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    """
    st.markdown(table_html, unsafe_allow_html=True)



def render_static_reliability_table(result: PredictionResult) -> None:
    rows = [
        ("Prediction Confidence", result.confidence.get("Prediction_Confidence_Label"), format_optional(result.confidence.get("Prediction_Confidence_Margin"))),
        ("Applicability Domain", result.applicability_domain.get("Applicability_Domain_Status"), result.applicability_domain.get("Applicability_Domain_Reliability")),
        ("Applicability Domain Score", "Max reference similarity", format_optional(result.applicability_domain.get("Applicability_Domain_Score"))),
        ("Structural Alert Flag", result.structural_alerts.get("Structural_Alert_Flag"), result.structural_alerts.get("Structural_Alert_Summary")),
        ("PAINS Alerts", format_optional(result.structural_alerts.get("PAINS_Alert_Count")), "PAINS structural filter count"),
        ("Brenk Alerts", format_optional(result.structural_alerts.get("Brenk_Alert_Count")), "Brenk structural filter count"),
    ]
    row_html = "".join(
        "<tr>"
        f"<td>{html.escape(str(feature))}</td>"
        f"<td>{html.escape(str(result_value))}</td>"
        f"<td>{html.escape(str(value))}</td>"
        "</tr>"
        for feature, result_value, value in rows
    )
    table_html = f"""
    <table class="static-table">
        <thead>
            <tr>
                <th>Feature</th>
                <th>Result</th>
                <th>Value / Note</th>
            </tr>
        </thead>
        <tbody>{row_html}</tbody>
    </table>
    """
    st.markdown(table_html, unsafe_allow_html=True)


def render_static_similarity_table(matches: List[Dict[str, object]]) -> None:
    if not matches:
        return

    rows = []
    for match in matches:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(match.get('Reference_Inhibitor', 'NA')))}</td>"
            f"<td>{html.escape(str(match.get('Reference_Type', 'NA')))}</td>"
            f"<td>{float(match.get('Tanimoto_Similarity', 0.0)):.4f}</td>"
            f"<td>{html.escape(str(match.get('Similarity_Interpretation', 'NA')))}</td>"
            "</tr>"
        )

    table_html = f"""
    <table class="static-table">
        <thead>
            <tr>
                <th>Reference Inhibitor</th>
                <th>Reference Type</th>
                <th>Tanimoto Similarity</th>
                <th>Similarity Interpretation</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    """
    st.markdown(table_html, unsafe_allow_html=True)



def render_dynamic_similarity_table(matches: List[Dict[str, object]]) -> None:
    if not matches:
        return

    match_df = pd.DataFrame(matches)
    if "Tanimoto_Similarity" in match_df.columns:
        match_df["Tanimoto_Similarity"] = match_df["Tanimoto_Similarity"].round(4)

    display_cols = [
        "Reference_Inhibitor",
        "Reference_Type",
        "Tanimoto_Similarity",
        "Similarity_Interpretation",
    ]
    display_cols = [col for col in display_cols if col in match_df.columns]

    table_height = max(175, min(248, 38 + 34 * (len(match_df) + 1)))
    st.dataframe(
        match_df[display_cols],
        use_container_width=True,
        hide_index=True,
        height=table_height,
    )


def render_dynamic_priority_table(df: pd.DataFrame) -> None:
    if df.empty:
        return

    columns = [
        "Input_Order",
        "Input_SMILES",
        "Prediction",
        "Probability_Active",
        "Prediction_Confidence_Label",
        "Applicability_Domain_Status",
        "Structural_Alert_Flag",
        "Nearest_Reference_Inhibitor",
        "Max_Tanimoto_Similarity",
        "Similarity_Interpretation",
        "QED",
        "Lipinski Ro5 Pass",
    ]

    existing_columns = [col for col in columns if col in df.columns]
    display_df = format_dataframe_for_display(df[existing_columns])

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=420,
    )



# -----------------------------------------------------------------------------
# Predicted active compound database utilities - Supabase + canonical SMILES identity
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_supabase_client():
    if create_client is None:
        raise ImportError(
            "The 'supabase' package is not installed. Install it with: pip install supabase"
        )

    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
    except Exception as exc:
        raise RuntimeError(
            "Supabase secrets are missing. Create .streamlit/secrets.toml with "
            "SUPABASE_URL and SUPABASE_KEY."
        ) from exc

    return create_client(url, key)


def canonical_smiles_to_compound_id(canonical_smiles: str) -> str:
    digest = hashlib.sha256(canonical_smiles.encode("utf-8")).hexdigest()[:12].upper()
    return f"AKT-ACT-{digest}"


def initialize_active_database() -> None:
    """
    Supabase table is created externally in the Supabase SQL Editor.

    This version uses the original table design:
    - compound_id: primary key
    - canonical_smiles: unique not null
    """
    return None


def database_record_from_result(
    result: PredictionResult,
    contributor_name: str,
    contributor_affiliation: str,
    contributor_email: str,
) -> Optional[Dict[str, object]]:
    if not result.valid or result.prediction_label != "Active" or not result.canonical_smiles:
        return None

    compound_id = canonical_smiles_to_compound_id(result.canonical_smiles)
    saved_at = pd.Timestamp.now(tz="UTC").isoformat()

    d = result.descriptors
    s = result.similarity
    c = result.confidence
    ad = result.applicability_domain
    alerts = result.structural_alerts

    contributor_name = (contributor_name or "").strip()
    contributor_affiliation = (contributor_affiliation or "").strip()
    contributor_email = (contributor_email or "").strip()

    return {
        "compound_id": compound_id,
        "canonical_smiles": result.canonical_smiles,
        "first_input_smiles": result.smiles,
        "prediction": result.prediction_label,
        "probability_active": result.probability_active,
        "probability_inactive": result.probability_inactive,
        "prediction_confidence_label": c.get("Prediction_Confidence_Label"),
        "prediction_confidence_margin": c.get("Prediction_Confidence_Margin"),
        "applicability_domain_status": ad.get("Applicability_Domain_Status"),
        "applicability_domain_reliability": ad.get("Applicability_Domain_Reliability"),
        "applicability_domain_score": ad.get("Applicability_Domain_Score"),
        "structural_alert_flag": alerts.get("Structural_Alert_Flag"),
        "structural_alert_count": alerts.get("Structural_Alert_Count"),
        "pains_alert_count": alerts.get("PAINS_Alert_Count"),
        "brenk_alert_count": alerts.get("Brenk_Alert_Count"),
        "structural_alert_summary": alerts.get("Structural_Alert_Summary"),
        "nearest_reference_inhibitor": s.get("Nearest_Reference_Inhibitor"),
        "nearest_reference_type": s.get("Nearest_Reference_Type"),
        "max_tanimoto_similarity": s.get("Max_Tanimoto_Similarity"),
        "similarity_interpretation": s.get("Similarity_Interpretation"),
        "top_3_scaffold_matches": s.get("Top_3_Scaffold_Matches"),
        "molecular_weight": d.get("Molecular Weight"),
        "logp_crippen": d.get("LogP (Crippen)"),
        "tpsa": d.get("TPSA"),
        "qed": d.get("QED"),
        "lipinski_ro5_pass": d.get("Lipinski Ro5 Pass"),
        "contributor_name": contributor_name or None,
        "contributor_affiliation": contributor_affiliation or None,
        "contributor_email": contributor_email or None,
        "first_saved_at": saved_at,
        "last_seen_at": saved_at,
        "run_count": 1,
    }


def save_active_results_to_database(
    results: List[PredictionResult],
    contributor_name: str = "",
    contributor_affiliation: str = "",
    contributor_email: str = "",
    save_contributor_info: bool = False,
) -> Dict[str, int]:
    if not save_contributor_info:
        contributor_name = ""
        contributor_affiliation = ""
        contributor_email = ""

    supabase = get_supabase_client()

    inserted = 0
    updated = 0
    skipped = 0
    failed = 0

    for result in results:
        record = database_record_from_result(
            result,
            contributor_name=contributor_name,
            contributor_affiliation=contributor_affiliation,
            contributor_email=contributor_email,
        )

        if record is None:
            skipped += 1
            continue

        try:
            existing_response = (
                supabase.table(ACTIVE_DATABASE_TABLE)
                .select("compound_id, run_count, probability_active")
                .eq("canonical_smiles", record["canonical_smiles"])
                .limit(1)
                .execute()
            )
            existing_data = existing_response.data or []

            if existing_data:
                existing = existing_data[0]
                old_probability = existing.get("probability_active")
                old_probability = float(old_probability) if old_probability is not None else -1.0
                new_probability = float(record["probability_active"] or -1.0)

                update_payload = {
                    "last_seen_at": record["last_seen_at"],
                    "run_count": int(existing.get("run_count") or 1) + 1,
                }

                # If contributor consent is provided, update/backfill contributor fields
                # for existing duplicate compounds as well.
                if save_contributor_info:
                    contributor_update_fields = {
                        "contributor_name": record.get("contributor_name"),
                        "contributor_affiliation": record.get("contributor_affiliation"),
                        "contributor_email": record.get("contributor_email"),
                    }
                    for field_name, field_value in contributor_update_fields.items():
                        if field_value is not None and str(field_value).strip():
                            update_payload[field_name] = str(field_value).strip()

                # Keep strongest prediction metadata if the same canonical compound appears again.
                if new_probability > old_probability:
                    update_payload.update(
                        {
                            "probability_active": record["probability_active"],
                            "probability_inactive": record["probability_inactive"],
                            "prediction_confidence_label": record["prediction_confidence_label"],
                            "prediction_confidence_margin": record["prediction_confidence_margin"],
                            "applicability_domain_status": record["applicability_domain_status"],
                            "applicability_domain_reliability": record["applicability_domain_reliability"],
                            "applicability_domain_score": record["applicability_domain_score"],
                            "structural_alert_flag": record["structural_alert_flag"],
                            "structural_alert_count": record["structural_alert_count"],
                            "pains_alert_count": record["pains_alert_count"],
                            "brenk_alert_count": record["brenk_alert_count"],
                            "structural_alert_summary": record["structural_alert_summary"],
                            "nearest_reference_inhibitor": record["nearest_reference_inhibitor"],
                            "nearest_reference_type": record["nearest_reference_type"],
                            "max_tanimoto_similarity": record["max_tanimoto_similarity"],
                            "similarity_interpretation": record["similarity_interpretation"],
                            "top_3_scaffold_matches": record["top_3_scaffold_matches"],
                            "molecular_weight": record["molecular_weight"],
                            "logp_crippen": record["logp_crippen"],
                            "tpsa": record["tpsa"],
                            "qed": record["qed"],
                            "lipinski_ro5_pass": record["lipinski_ro5_pass"],
                        }
                    )

                supabase.table(ACTIVE_DATABASE_TABLE).update(update_payload).eq(
                    "canonical_smiles", record["canonical_smiles"]
                ).execute()
                updated += 1

            else:
                supabase.table(ACTIVE_DATABASE_TABLE).insert(record).execute()
                inserted += 1

        except Exception as exc:
            failed += 1
            st.warning(f"Supabase save failed for one compound: {exc}")

    return {
        "inserted": inserted,
        "updated_existing": updated,
        "skipped_nonactive_or_invalid": skipped,
        "failed": failed,
    }


def load_active_database_dataframe() -> pd.DataFrame:
    supabase = get_supabase_client()

    all_rows = []
    page_size = 1000
    start = 0

    while True:
        response = (
            supabase.table(ACTIVE_DATABASE_TABLE)
            .select("*")
            .order("probability_active", desc=True)
            .range(start, start + page_size - 1)
            .execute()
        )

        rows = response.data or []
        all_rows.extend(rows)

        if len(rows) < page_size:
            break

        start += page_size

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    rename_map = {
        "compound_id": "Compound_ID",
        "canonical_smiles": "Canonical_SMILES",
        "prediction": "Prediction",
        "probability_active": "Probability_Active",
        "prediction_confidence_label": "Prediction_Confidence",
        "applicability_domain_status": "Applicability_Domain",
        "structural_alert_flag": "Structural_Alert_Flag",
        "nearest_reference_inhibitor": "Nearest_Reference_Inhibitor",
        "max_tanimoto_similarity": "Max_Tanimoto_Similarity",
        "qed": "QED",
        "lipinski_ro5_pass": "Lipinski_Ro5_Pass",
        "contributor_name": "Contributor_Name",
        "contributor_affiliation": "Contributor_Affiliation",
        "contributor_email": "Contributor_Email",
        "first_saved_at": "First_Saved_At",
        "last_seen_at": "Last_Seen_At",
        "run_count": "Run_Count",
    }

    df = df.rename(columns=rename_map)

    hidden_cols = ["First_Saved_At", "Last_Seen_At"]

    preferred_cols = [
        "Compound_ID",
        "Canonical_SMILES",
        "Prediction",
        "Probability_Active",
        "Prediction_Confidence",
        "Applicability_Domain",
        "Structural_Alert_Flag",
        "Nearest_Reference_Inhibitor",
        "Max_Tanimoto_Similarity",
        "QED",
        "Lipinski_Ro5_Pass",
        "Run_Count",
    ]

    contributor_cols = [
        "Contributor_Name",
        "Contributor_Affiliation",
        "Contributor_Email",
    ]

    existing_preferred_cols = [col for col in preferred_cols if col in df.columns]
    existing_contributor_cols = [col for col in contributor_cols if col in df.columns]
    other_cols = [
        col for col in df.columns
        if col not in existing_preferred_cols
        and col not in existing_contributor_cols
        and col not in hidden_cols
    ]

    return df[existing_preferred_cols + other_cols + existing_contributor_cols]


def render_contributor_inputs(context_key: str) -> Tuple[str, str, str, bool]:
    with st.expander("🗄️ Contributor information for predicted-active database", expanded=True):
        st.caption(
            "Optional. Fill these fields before running prediction. Details are saved only with predicted Active compounds "
            "and only if consent is checked. Please avoid commas in contributor fields; use underscores instead. "
        )

        contributor_name = st.text_input(
            "Contributor name",
            key=f"{context_key}_contributor_name",
            placeholder="Firstname_Middlename_Lastname",
            help="Avoid commas. Use spaces or underscores.",
        ).strip()

        contributor_affiliation = st.text_input(
            "Contributor affiliation",
            key=f"{context_key}_contributor_affiliation",
            placeholder="Department_University",
            help="Avoid commas. Recommended format: Department_University or Lab_Department_University.",
        ).strip()

        contributor_email = st.text_input(
            "Contributor email address",
            key=f"{context_key}_contributor_email",
            placeholder="name@example.com",
            help="Use a single email address only. Do not add comma-separated emails.",
        ).strip()

        save_contributor_info = st.checkbox(
            "I agree to save this contributor information with predicted Active compounds.",
            key=f"{context_key}_save_contributor_info",
        )

        validation_error = validate_contributor_inputs(
            contributor_name, contributor_affiliation, contributor_email, save_contributor_info
        )
        if validation_error:
            st.warning(validation_error)
        elif save_contributor_info:
            st.caption(
                f"Contributor record ready: {contributor_name} · "
                f"{contributor_affiliation} · {contributor_email}"
            )

    # Store comma-free values to avoid CSV/display parsing problems.
    contributor_name = contributor_name.replace(",", "_")
    contributor_affiliation = contributor_affiliation.replace(",", "_")
    contributor_email = contributor_email.replace(",", "_")

    return contributor_name, contributor_affiliation, contributor_email, save_contributor_info


def validate_contributor_inputs(
    contributor_name: str,
    contributor_affiliation: str,
    contributor_email: str,
    save_contributor_info: bool,
) -> Optional[str]:
    """Return a user-facing validation message, or None when valid.

    The contributor widgets are rendered inside a Streamlit form in both single
    and batch modes. That forces text_input values and the consent checkbox to
    be submitted atomically with the prediction button, avoiding the stale-value
    behavior where Streamlit shows "Press Enter to apply" and Python still sees
    old empty values.
    """
    if not save_contributor_info:
        return None

    contributor_name = (contributor_name or "").strip()
    contributor_affiliation = (contributor_affiliation or "").strip()
    contributor_email = (contributor_email or "").strip()

    missing_fields = []
    if not contributor_name:
        missing_fields.append("name")
    if not contributor_affiliation:
        missing_fields.append("affiliation")
    if not contributor_email:
        missing_fields.append("email address")
    if missing_fields:
        return (
            "Contributor consent is checked, but the following field(s) are empty: "
            + ", ".join(missing_fields)
            + ". Please fill them before running prediction."
        )

    comma_fields = []
    if "," in contributor_name:
        comma_fields.append("name")
    if "," in contributor_affiliation:
        comma_fields.append("affiliation")
    if "," in contributor_email:
        comma_fields.append("email address")
    if comma_fields:
        return (
            "Please remove commas from the following contributor field(s): "
            + ", ".join(comma_fields)
            + ". Use underscores instead, e.g., Department_of_Botany_University_of_Dhaka."
        )

    if "@" not in contributor_email or "." not in contributor_email:
        return "The contributor email address appears invalid. Please check it before running prediction."

    return None


def render_active_database_page() -> None:
    st.markdown("## 🗄️ Predicted Active Compound Database")
    st.write(
        "This Supabase repository stores unique compounds predicted as Active. Duplicate compounds are prevented using canonical SMILES."
    )

    try:
        db_df = load_active_database_dataframe()
    except Exception as exc:
        st.error(f"Could not load Supabase database: {exc}")
        st.info("Check your .streamlit/secrets.toml values, Supabase table, and RLS policies.")
        return

    if db_df.empty:
        st.info("No predicted Active compounds have been saved yet.")
        return

    total_compounds = len(db_df)
    total_contributors = int(db_df["Contributor_Email"].dropna().nunique()) if "Contributor_Email" in db_df.columns else 0
    mean_probability = db_df["Probability_Active"].mean() if "Probability_Active" in db_df.columns else np.nan
    mean_similarity = db_df["Max_Tanimoto_Similarity"].mean() if "Max_Tanimoto_Similarity" in db_df.columns else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💊 Unique Active Compounds", f"{total_compounds:,}")
    c2.metric("🗄️ Unique Contributor Emails", f"{total_contributors:,}")
    c3.metric("📈 Mean P(active)", format_optional(mean_probability))
    c4.metric("🔬 Mean Similarity", format_optional(mean_similarity))

    search_query = st.text_input(
        "🔎 Search database",
        placeholder="Search by compound ID, canonical SMILES, contributor, reference inhibitor...",
        key="active_database_search_query",
    )

    view_df = db_df.copy()
    if search_query.strip():
        query = search_query.strip().lower()
        mask = view_df.astype(str).apply(
            lambda row: row.str.lower().str.contains(query, regex=False).any(),
            axis=1,
        )
        view_df = view_df[mask]

    st.markdown("### 💾 Saved Predicted Active Compounds")
    st.dataframe(
        format_dataframe_for_display(view_df),
        use_container_width=True,
        hide_index=True,
        height=520,
    )

    csv_bytes = db_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Active Compound Database as CSV",
        data=csv_bytes,
        file_name="AKT_Scan_AI_predicted_active_database.csv",
        mime="text/csv",
        use_container_width=True,
        key="download_active_database_csv",
    )

    with st.expander("ℹ️ Database information"):
        st.write("Database backend: `Supabase PostgreSQL`")
        st.write("Unique compound rule: canonical SMILES → stable AKT-ACT ID.")
# -----------------------------------------------------------------------------
# UI components
# -----------------------------------------------------------------------------
def render_hero() -> None:
    st.markdown(
        f"""
        <div class="hero">
            <h1>{APP_NAME}</h1>
            <p>{APP_TITLE}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_intro() -> None:
    st.markdown(
        """
        <div class="section-card">
            <h3>🌿 About</h3>
            <p>
            <strong>AKT-Scan AI</strong> is a machine-learning platform for rapid
            SMILES-based virtual screening against <strong>AKT1 (RAC-alpha serine/threonine-protein kinase)</strong>,
            a key regulator in cancer signaling pathways. The platform integrates a trained LightGBM classifier
            with molecular fingerprinting and medicinal-chemistry profiling to support early-stage anticancer
            bioactivity prioritization, drug-likeness assessment, and high-throughput compound triage. In addition to predictive screening, AKT-Scan AI maintains a continuously 
            growing <strong>Predicted Active Compound Database (AKT1 Active Library)</strong>, 
            where compounds predicted as Active are stored as unique chemical entities. 
            This evolving repository enables accumulation of model-consistent 
            AKT1-focused chemical space across user submissions, supporting scaffold 
            exploration, prioritization consistency, and future data-driven refinement.
            </p>
            <p class="small-note">
            Predictions are computational estimates and should be interpreted as decision-support
            evidence. Experimental validation remains necessary before biological conclusions are made.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_similarity_method_note() -> None:
    st.markdown(
        """
        <div class="similarity-card">
            <h3>🔬 Scaffold Similarity Analysis</h3>
            <p>
            Scaffold similarity is computed against a curated set of five model-consistent AKT inhibitor
            references using Morgan fingerprint-based Tanimoto similarity. This supports structural
            interpretation of predicted compounds and does not confirm biological equivalence.
            </p>
            <p class="small-note">
            Reference set: Vevorisertib, GSK690693, Ipatasertib, Uprosertib, and AT7867.
            Higher similarity suggests closer structural resemblance to one of these reference inhibitor
            scaffolds; lower similarity may indicate a more structurally distinct compound.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )



def render_phase1_method_note() -> None:
    st.markdown(
        f"""
        <div class="section-card">
            <h3>🛡️ Reliability and Safety Screening</h3>
            <p>
            This screening module adds three interpretation layers: prediction confidence,
            reference-scaffold applicability domain, and PAINS/Brenk structural alert detection.
            These features help users judge whether a prediction is close to the decision boundary,
            structurally supported by the model-consistent reference inhibitors, and whether common
            medicinal-chemistry alert patterns are detected.
            </p>
            <p class="small-note">
            Applicability domain is estimated from the maximum Tanimoto similarity to the curated
            reference inhibitors. In-domain threshold: {AD_IN_DOMAIN_THRESHOLD:.2f}; borderline threshold:
            {AD_BORDERLINE_THRESHOLD:.2f}. Structural alerts are screening flags only and are not toxicity
            predictions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_active_library_method_note() -> None:
    st.markdown(
        """
        <div class="section-card">
            <h3>🗄️ Predicted Active Compound Database / AKT1 Active Library</h3>
            <p>
            The <strong>AKT1 Active Library</strong> is a continuously growing Supabase-backed repository
            of unique compounds predicted as Active by AKT-Scan AI. Each saved compound is assigned a stable
            AKT-ACT identifier, and duplicate entries are prevented using canonical SMILES-based matching.
            </p>
            <p>
            As more users screen molecules, the library can gradually become a curated resource of
            machine-learning-prioritized AKT1-focused chemical matter. This can support future hit
            prioritization, scaffold comparison, compound triage, contributor tracking, and downstream
            medicinal-chemistry analysis.
            </p>
            <p class="small-note">
            Future implementation may connect this predicted-active library with structure-based drug design
            workflows, including docking, pharmacophore modeling, molecular dynamics prioritization,
            scaffold clustering, and lead-optimization campaigns. All entries remain computational predictions
            and require experimental validation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(model_ready: bool, selected_indices: Optional[np.ndarray]) -> float:
    with st.sidebar:
        st.markdown("### 🎛️ AKT-Scan AI Controls")
        st.caption("🎚️ Bioactivity threshold and model configuration")
        threshold = st.slider(
            "Active classification threshold",
            min_value=0.05,
            max_value=0.95,
            value=DEFAULT_THRESHOLD,
            step=0.01,
            help="Compounds with Probability Active greater than or equal to this value are classified as Active.",
        )

        st.markdown("---")
        st.markdown("### ⚙️ Prediction Model Status")
        if model_ready:
            st.success("✅ Model and selected features loaded")
            st.write(f"**Model file:** `{MODEL_FILE}`")
            st.write(f"**Selected features:** `{len(selected_indices):,}`")
            st.write(f"**Fingerprint:** Morgan radius {FINGERPRINT_RADIUS}, {FINGERPRINT_BITS} bits")
        else:
            st.error("❌ Model files not loaded")
            st.write(f"Required: `{MODEL_FILE}` and `{FEATURE_INDEX_FILE}`")

        st.markdown("---")
        st.markdown("### 🔬 AKT Inhibitor Reference Library")
        st.write(f"Reference inhibitors: **{len(REFERENCE_INHIBITORS)}**")
        st.caption("Tanimoto similarity supports scaffold comparison and reference-domain estimation.")

        st.markdown("---")
        st.markdown("### 📦 Screening Batch Limits")
        st.write(f"Maximum molecules per batch: **{MAX_BATCH_MOLECULES:,}**")
        st.caption("Input can be pasted manually or uploaded as a CSV with a header named SMILES.")

    return threshold


def render_performance_section() -> None:
    st.markdown(
        """
        <div class="section-card performance-card">
            <h3>📊 ML Performance and Model Information</h3>
            <p>
            The deployed classifier is a LightGBM model trained using binary molecular fingerprints.
            The positive class is interpreted as <strong>Active</strong> and the negative class as
            <strong>Inactive</strong>. The app applies the saved feature index array to reproduce the
            selected fingerprint feature space before inference.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_cv, tab_external = st.tabs(
        ["🧪 Five-fold Cross Validation", "🌍 External Validation"]
    )
    with tab_cv:
        render_metric_grid(CV_METRICS)
    with tab_external:
        render_metric_grid(EXTERNAL_METRICS)


def render_top_info_tabs() -> None:
    about_tab, performance_tab, similarity_tab, phase1_tab, library_tab = st.tabs(
        [
            "🌿 About",
            "📊 ML Performance and Model Information",
            "🔬 Scaffold Similarity Analysis",
            "🛡️ Reliability and Safety Screening",
            "🗄️ AKT1 Active Library",
        ]
    )

    with about_tab:
        render_intro()

    with performance_tab:
        render_performance_section()

    with similarity_tab:
        render_similarity_method_note()

    with phase1_tab:
        render_phase1_method_note()

    with library_tab:
        render_active_library_method_note()


def render_metric_grid(metrics: Dict[str, float]) -> None:
    cols = st.columns(3)
    for i, (name, value) in enumerate(metrics.items()):
        with cols[i % 3]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="value">{value:.4f}</div>
                    <div class="label">{name}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )



def render_single_phase1_section(result: PredictionResult) -> None:
    st.markdown("### 🛡️ Prediction Reliability and Safety Alerts")
    render_static_reliability_table(result)
    st.caption(
        "Confidence is based on distance from the selected threshold. Applicability domain is reference-scaffold based. Structural alerts are rule-based screening flags, not toxicity predictions."
    )


def render_single_similarity_section(result: PredictionResult, mol: Chem.Mol) -> None:
    st.markdown("### 🔬 Scaffold Similarity to AKT Inhibitor References")

    sim = result.similarity
    nearest_ref = sim.get("Nearest_Reference_Inhibitor") or "NA"
    max_similarity = format_optional(sim.get("Max_Tanimoto_Similarity"))
    interpretation = sim.get("Similarity_Interpretation") or "NA"

    st.markdown(
        f"""
        <div class="similarity-card">
            <h3>🎯 Nearest Scaffold Match</h3>
            <p><strong>Nearest reference inhibitor:</strong> {nearest_ref}</p>
            <p><strong>Maximum Tanimoto similarity:</strong> {max_similarity}</p>
            <p><strong>Interpretation:</strong> {interpretation}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    matches = sim.get("All_Matches", [])
    if matches:
        render_dynamic_similarity_table(matches)

        comparison_png = single_similarity_comparison_png(mol, result)
        if comparison_png:
            render_centered_png_image(
                comparison_png,
                "Query compound vs nearest model-consistent AKT inhibitor reference",
            )
            st.download_button(
                label="⬇️ Download Query vs Nearest Reference Image",
                data=comparison_png,
                file_name="AKT_Scan_AI_query_vs_nearest_reference.png",
                mime="image/png",
                use_container_width=True,
                key="download_single_similarity_image",
            )

    st.caption(
        "Similarity values are for structural interpretation only. They should not be interpreted as experimental activity confirmation."
    )

def render_single_mode(model, selected_indices: np.ndarray, threshold: float) -> None:
    st.markdown("## 🎗️ Single Molecule Drug-Activity Prediction")
    st.write("Enter one SMILES string to predict AKT1 bioactivity, compute molecular properties, and compare scaffold similarity.")

    example_smiles = "CC(=O)Oc1ccccc1C(=O)O"
    with st.form("single_prediction_form", clear_on_submit=False):
        smiles = st.text_input(
            "SMILES",
            value="",
            placeholder=f"Example: {example_smiles}",
            help="Paste a valid canonical or non-canonical SMILES string.",
        )

        contributor_name, contributor_affiliation, contributor_email, save_contributor_info = render_contributor_inputs("single")

        predict_button = st.form_submit_button(
            "🚀 Predict Single Molecule",
            type="primary",
            use_container_width=True,
        )

    if predict_button:
        if not smiles.strip():
            st.warning("Please enter a SMILES string.")
            st.session_state.single_prediction_result = None
            st.session_state.single_prediction_smiles = None
            return

        contributor_error = validate_contributor_inputs(
            contributor_name, contributor_affiliation, contributor_email, save_contributor_info
        )
        if contributor_error:
            st.error(contributor_error)
            return

        with st.spinner("Generating fingerprint, predicting bioactivity, calculating molecular properties, and comparing reference scaffolds..."):
            results = predict_molecules([smiles], model, selected_indices, threshold)
        result = results[0]

        if not result.valid:
            st.error(f"Invalid SMILES: {result.error}")
            st.session_state.single_prediction_result = None
            st.session_state.single_prediction_smiles = None
            return

        save_summary = save_active_results_to_database(
            results,
            contributor_name=contributor_name,
            contributor_affiliation=contributor_affiliation,
            contributor_email=contributor_email,
            save_contributor_info=save_contributor_info,
        )
        if save_summary["inserted"] > 0:
            st.success(f"Saved {save_summary['inserted']} new predicted Active compound to the database.")
        elif save_summary["updated_existing"] > 0:
            st.info("This predicted Active compound already exists in the database; run count was updated.")

        # Store the successful single-mode result so Streamlit reruns caused by
        # download_button clicks do not clear the output area.
        st.session_state.single_prediction_result = result
        st.session_state.single_prediction_smiles = smiles

    # Streamlit reruns the script when a download_button is clicked. Render the
    # latest stored single result outside the predict_button block so the results
    # remain visible after downloading the image.
    result = st.session_state.get("single_prediction_result")
    stored_smiles = st.session_state.get("single_prediction_smiles")

    if result is None or not stored_smiles:
        return

    mol = smiles_to_mol(stored_smiles)
    if mol is None:
        return

    left, right = st.columns([0.92, 1.08])

    with left:
        st.markdown("### 🧫 2D Chemical Structure")
        img64 = mol_image_base64(mol)
        st.markdown(
            f"""
            <div class="section-card" style="text-align:center;">
                <img src="data:image/png;base64,{img64}" style="max-width:100%; border-radius:18px;" />
                <p class="small-note"><strong>Canonical SMILES:</strong><br>{result.canonical_smiles}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        active = result.prediction_label == "Active"
        card_class = "prediction-active" if active else "prediction-inactive"
        label_icon = "✅" if active else "⛔"
        st.markdown(
            f"""
            <div class="{card_class}">
                <h3>{label_icon} Predicted Bioactivity: {result.prediction_label}</h3>
                <p><strong>Probability Active:</strong> {result.probability_active:.4f}</p>
                <p><strong>Probability Inactive:</strong> {result.probability_inactive:.4f}</p>
                <p><strong>Confidence:</strong> {result.confidence.get("Prediction_Confidence_Label")} ({format_optional(result.confidence.get("Prediction_Confidence_Margin"))})</p>
                <p><strong>Applicability Domain:</strong> {result.applicability_domain.get("Applicability_Domain_Status")}</p>
                <p><strong>Structural Alerts:</strong> {result.structural_alerts.get("Structural_Alert_Flag")}</p>
                <p><strong>Nearest Reference Scaffold:</strong> {result.similarity.get("Nearest_Reference_Inhibitor")}</p>
                <p><strong>Max Tanimoto Similarity:</strong> {format_optional(result.similarity.get("Max_Tanimoto_Similarity"))}</p>
                <p class="small-note">Decision threshold: {threshold:.4f}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("### Key Drug-Likeness Summary")
        d = result.descriptors
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("💊 QED", format_optional(d.get("QED")))
        k2.metric("🧪 LogP", format_optional(d.get("LogP (Crippen)")))
        k3.metric("📏 Lipinski", format_optional(d.get("Lipinski Ro5 Pass")))
        k4.metric("⚖️ MW", format_optional(d.get("Molecular Weight")))

    st.markdown("### 🧪 ADME, QED, and Medicinal Chemistry Profile")
    desc_df = pd.DataFrame(
        [{"Property": k, "Value": format_optional(v)} for k, v in result.descriptors.items()]
    )
    st.dataframe(desc_df, use_container_width=True, hide_index=True)

    render_single_phase1_section(result)
    render_single_similarity_section(result, mol)


def render_batch_mode(model, selected_indices: np.ndarray, threshold: float) -> None:
    st.markdown("## 📦 Batch Drug-Activity Screening")
    st.write(
        f"Paste SMILES or upload a CSV file. A maximum of {MAX_BATCH_MOLECULES:,} molecules is allowed per batch."
    )
    st.info("For CSV upload, include a column header exactly named **SMILES**. Without a header, the first molecule may be interpreted as the header and skipped.")

    input_tab, upload_tab = st.tabs(["✍️ Paste SMILES", "📄 Upload CSV"])
    smiles_list: List[str] = []
    source_label = None

    with input_tab:
        pasted = st.text_area(
            "Enter one SMILES per line",
            height=230,
            placeholder="CCO\nCC(=O)Oc1ccccc1C(=O)O\nC1=CC=CC=C1",
            key="batch_pasted_smiles",
        )
        if pasted.strip():
            smiles_list = [line.strip() for line in pasted.splitlines() if line.strip()]
            source_label = "pasted input"

    with upload_tab:
        st.warning("CSV files should include a header row with a column named **SMILES**. Example: first row = SMILES, then one molecule per row.")
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="batch_csv_uploader")
        if uploaded_file is not None:
            try:
                upload_df = get_uploaded_csv_dataframe(uploaded_file)
                if upload_df is not None:
                    st.markdown("### 📄 Uploaded CSV Preview")
                    st.dataframe(
                        upload_df.head(10),
                        use_container_width=True,
                        height=360,
                        hide_index=False,
                    )
                    st.caption(f"Showing first {min(len(upload_df), 10)} row(s) only.")

                    possible_smiles_cols = [c for c in upload_df.columns if c.lower() in {"smiles", "canonical_smiles", "smile"}]
                    default_idx = upload_df.columns.get_loc(possible_smiles_cols[0]) if possible_smiles_cols else 0
                    smiles_col = st.selectbox(
                        "Select the SMILES column",
                        options=list(upload_df.columns),
                        index=int(default_idx),
                        key="batch_smiles_column",
                    )
                    if smiles_col:
                        smiles_list = upload_df[smiles_col].dropna().astype(str).str.strip().tolist()
                        smiles_list = [s for s in smiles_list if s]
                        source_label = f"CSV column '{smiles_col}'"
            except Exception as exc:
                st.error(f"Could not read uploaded CSV: {exc}")

    st.info(
        f"Current batch size from {source_label or 'input'}: **{len(smiles_list):,}** molecule(s)."
    )

    with st.form("batch_prediction_form", clear_on_submit=False):
        contributor_name, contributor_affiliation, contributor_email, save_contributor_info = render_contributor_inputs("batch")

        col_run, col_clear = st.columns([3, 1])
        with col_run:
            run_batch = st.form_submit_button(
                "🚀 Run Batch Prediction",
                type="primary",
                use_container_width=True,
            )
        with col_clear:
            clear_outputs = st.form_submit_button(
                "🧹 Clear Results",
                use_container_width=True,
            )

    if clear_outputs:
        clear_batch_session_outputs()
        st.success("Batch results cleared.")
        return

    if run_batch:
        if not smiles_list:
            st.warning("Please provide SMILES by pasting them or uploading a CSV file.")
            return
        if len(smiles_list) > MAX_BATCH_MOLECULES:
            st.error(
                f"Batch size is {len(smiles_list):,}. Please limit input to {MAX_BATCH_MOLECULES:,} molecules."
            )
            return

        contributor_error = validate_contributor_inputs(
            contributor_name, contributor_affiliation, contributor_email, save_contributor_info
        )
        if contributor_error:
            st.error(contributor_error)
            return

        st.markdown("### ⏳ Batch Prediction Progress")
        progress_bar = st.progress(0.0)
        progress_text = st.empty()
        status_box = st.empty()

        start_time = time.time()
        with st.spinner("Running high-throughput bioactivity, ADME/QED, and scaffold similarity analysis..."):
            results = predict_molecules(
                smiles_list,
                model,
                selected_indices,
                threshold,
                progress_bar=progress_bar,
                progress_text=progress_text,
            )
        elapsed = time.time() - start_time
        progress_bar.progress(1.0)
        progress_text.markdown(
            f"**Predicted {len([r for r in results if r.valid]):,} valid molecules out of {len(results):,} submitted molecules** · 100.00% complete"
        )
        status_box.success(f"Batch prediction completed in {elapsed:.4f} seconds.")

        df = results_to_dataframe(results)
        db_save_summary = save_active_results_to_database(
            results,
            contributor_name=contributor_name,
            contributor_affiliation=contributor_affiliation,
            contributor_email=contributor_email,
            save_contributor_info=save_contributor_info,
        )
        st.session_state.last_database_save_summary = db_save_summary

        st.session_state.batch_results_df = df
        st.session_state.batch_elapsed = elapsed
        st.session_state.batch_source_label = source_label
        st.session_state.batch_completed = True

    if st.session_state.batch_completed and st.session_state.batch_results_df is not None:
        df = st.session_state.batch_results_df
        valid_count = int(df["Valid_SMILES"].sum())
        invalid_count = len(df) - valid_count
        active_count = int((df["Prediction"] == "Active").sum())
        inactive_count = int((df["Prediction"] == "Inactive").sum())

        valid_df = df[df["Valid_SMILES"] == True].copy()
        avg_similarity = valid_df["Max_Tanimoto_Similarity"].mean() if not valid_df.empty else np.nan
        top_similarity = valid_df["Max_Tanimoto_Similarity"].max() if not valid_df.empty else np.nan

        if st.session_state.batch_elapsed is not None:
            st.success(f"Batch prediction completed in {st.session_state.batch_elapsed:.4f} seconds.")

        if st.session_state.last_database_save_summary:
            dbs = st.session_state.last_database_save_summary
            st.info(
                f"Database update: {dbs['inserted']} new Active compound(s) saved; "
                f"{dbs['updated_existing']} duplicate/existing Active compound(s) updated; "
                f"{dbs['skipped_nonactive_or_invalid']} non-active or invalid compound(s) skipped; "
                f"{dbs.get('failed', 0)} failed save(s)."
            )

        low_conf_count = int((df["Prediction_Confidence_Label"] == "Low confidence / near threshold").sum()) if "Prediction_Confidence_Label" in df.columns else 0
        out_domain_count = int((df["Applicability_Domain_Status"] == "Outside reference-scaffold domain").sum()) if "Applicability_Domain_Status" in df.columns else 0
        alert_count = int((df["Structural_Alert_Flag"] == "Yes").sum()) if "Structural_Alert_Flag" in df.columns else 0

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("📥 Submitted", f"{len(df):,}")
        c2.metric("✅ Valid", f"{valid_count:,}")
        c3.metric("🟢 Predicted Active", f"{active_count:,}")
        c4.metric("⚠️ Low Confidence", f"{low_conf_count:,}")
        c5.metric("🧭 Outside Domain", f"{out_domain_count:,}")
        c6.metric("🚩 With Alerts", f"{alert_count:,}")

        st.markdown("### 📊 Batch Screening Results")
        preferred_cols = [
            "Input_Order", "Input_SMILES", "Canonical_SMILES", "Valid_SMILES", "Prediction",
            "Probability_Active", "Probability_Inactive",
            "Prediction_Confidence_Label", "Prediction_Confidence_Margin",
            "Applicability_Domain_Status", "Applicability_Domain_Reliability", "Applicability_Domain_Score",
            "Structural_Alert_Flag", "Structural_Alert_Count", "PAINS_Alert_Count", "Brenk_Alert_Count",
            "Structural_Alert_Summary",
            "Nearest_Reference_Inhibitor", "Nearest_Reference_Type",
            "Max_Tanimoto_Similarity", "Similarity_Interpretation",
            "Top_3_Scaffold_Matches", "QED", "LogP (Crippen)",
            "Molecular Weight", "TPSA", "Lipinski Ro5 Pass", "Error",
        ]
        remaining_cols = [c for c in df.columns if c not in preferred_cols]
        display_cols = [c for c in preferred_cols if c in df.columns] + remaining_cols
        st.dataframe(format_dataframe_for_display(df[display_cols]), use_container_width=True, hide_index=True, height=420)


        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Enriched Prediction Results as CSV",
            data=csv_bytes,
            file_name="AKT_Scan_AI_enriched_batch_predictions.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_batch_csv_button",
        )


def format_optional(value) -> str:
    if value is None:
        return "NA"
    if isinstance(value, str):
        return value
    try:
        value = float(value)
        if np.isnan(value) or np.isinf(value):
            return "NA"
        return f"{value:.4f}"
    except Exception:
        return str(value)


def render_footer() -> None:
    st.markdown(
        f"""
        <div class="footer">
            <strong>Developed by {DEVELOPERS}</strong><br>
            {AFFILIATION}
        </div>
        """,
        unsafe_allow_html=True,
    )



def initialize_session_state() -> None:
    defaults = {
        "batch_results_df": None,
        "batch_elapsed": None,
        "batch_source_label": None,
        "batch_completed": False,
        "uploaded_csv_df": None,
        "uploaded_csv_signature": None,
        "last_database_save_summary": None,
        "single_prediction_result": None,
        "single_prediction_smiles": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_batch_session_outputs() -> None:
    st.session_state.batch_results_df = None
    st.session_state.batch_elapsed = None
    st.session_state.batch_source_label = None
    st.session_state.batch_completed = False


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    initialize_session_state()
    initialize_active_database()
    inject_css()
    render_hero()
    render_top_info_tabs()

    model = None
    selected_indices = None
    model_ready = False
    model_error = None

    try:
        model, selected_indices = load_model_and_features()
        model_ready = True
    except Exception as exc:
        model_error = str(exc)

    threshold = render_sidebar(model_ready, selected_indices)

    if not model_ready:
        st.error(model_error)
        st.markdown(
            """
            Place the following files in the same folder as `app.py`, then restart the Streamlit app:

            - `LightGBM.pkl`
            - `selected_feature_indices.npy`
            """
        )
        render_performance_section()
        render_footer()
        return

    st.markdown("---")
    mode = st.radio(
        "🧭 Select Navigation",
        options=["🎗️ Single Molecule Drug-Activity Prediction", "📦 Batch Drug-Activity Screening", "🗄️ Predicted Active Database"],
        horizontal=True,
    )

    if mode == "🎗️ Single Molecule Drug-Activity Prediction":
        render_single_mode(model, selected_indices, threshold)
    elif mode == "📦 Batch Drug-Activity Screening":
        render_batch_mode(model, selected_indices, threshold)
    else:
        render_active_database_page()

    render_footer()


if __name__ == "__main__":
    main()
