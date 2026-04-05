from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from functools import lru_cache
import os

app = FastAPI(title="MedAssist API", version="1.0.0")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
DOCUMENTS_FOLDER = "documents"

KNOWLEDGE_BASE = """
=== TRANSPLANTATION RÉNALE — BASE DE CONNAISSANCES ===

--- INTRODUCTION ---
La transplantation rénale est le traitement de référence de l'insuffisance rénale chronique terminale (IRCT).
Elle offre une meilleure qualité de vie et une survie prolongée par rapport à la dialyse chronique.
En 2023, plus de 3 500 transplantations rénales ont été réalisées en France, dont environ 22% à partir de donneurs vivants.
L'IRCT touche environ 90 000 patients en France. La transplantation permet de normaliser la fonction rénale,
de contrôler l'hypertension artérielle, de corriger l'anémie et d'améliorer le métabolisme phospho-calcique.

--- INDICATIONS ---
La transplantation rénale est indiquée chez tout patient présentant une IRCT, définie par un DFG inférieur à 15 mL/min/1,73m2.
Principales étiologies : néphropathie diabétique (25%), néphropathie vasculaire/HTA (22%),
glomérulonéphrites chroniques (18%), polykystose rénale autosomique dominante (10%),
néphropathies héréditaires comme Alport (7%), autres/causes inconnues (18%).

--- CONTRE-INDICATIONS ---
Contre-indications absolues : néoplasie évolutive ou récente (moins de 2 ans selon le type),
infection active non contrôlée, insuffisance cardiaque sévère non compensée (FEVG < 25%),
espérance de vie inférieure à 2 ans, non-observance thérapeutique avérée.

--- BILAN PRÉ-GREFFE ---
Bilan biologique : groupe sanguin ABO, phénotype érythrocytaire, typage HLA classes I et II,
recherche d'anticorps anti-HLA (PRA), sérologies virales (VIH, VHB, VHC, CMV, EBV, HSV, HTLV),
NFS, ionogramme, bilan hépatique, lipidique et thyroïdien.
Bilan cardiovasculaire : ECG, échographie cardiaque transthoracique, épreuve d'effort ou scintigraphie myocardique.

--- IMMUNOSUPPRESSION ---
Le traitement immunosuppresseur repose sur une trithérapie :
1. Tacrolimus (FK506) : inhibe IL-2. Effets : néphrotoxicité, HTA, diabète.
   Taux cible J0-J30 : 8-12 ng/mL. Après 1 an : 5-8 ng/mL.
2. Ciclosporine A : inhibe IL-2. Effets : néphrotoxicité, hirsutisme.
3. Mycophénolate mofétil (MMF) : inhibe la synthèse des purines. Effets : troubles digestifs, leucopénie.
4. Sirolimus / Evérolimus : inhibe mTOR. Effets : dyslipidémie, retard de cicatrisation.
5. Prednisone : anti-inflammatoire. Effets : ostéoporose, diabète, HTA.
6. Basiliximab : anti-CD25 (induction), bien toléré.

--- REJET AIGU ---
Rejet aigu cellulaire : médié par les lymphocytes T.
Traitement : bolus méthylprednisolone 500mg IV x3j, puis SAT si grade III.
Rejet aigu humoral : médié par DSA (anticorps anti-HLA donneurs spécifiques).
Traitement : échanges plasmatiques (5-10 séances) + IgIV (2g/kg) + rituximab 375mg/m2.
Signes : élévation créatinine, oligurie, douleur greffon, fièvre.

--- REJET CHRONIQUE ---
Néphropathie chronique d'allogreffe : principale cause de perte du greffon à long terme.
Manifestations : dégradation progressive de la fonction rénale, protéinurie,
fibrose interstitielle et atrophie tubulaire à la biopsie (critères de Banff).

--- COMPLICATIONS INFECTIEUSES ---
CMV : surveillance PCR, traitement valganciclovir. Prophylaxie : valganciclovir 900mg/j x 3-6 mois.
BK virus : BKVN — réduction de l'immunosuppression.
Pneumocystose (PCP) : prévention par cotrimoxazole.
EBV : risque de PTLD (lymphome post-transplantation).

--- COMPLICATIONS MÉTABOLIQUES ---
NODAT (diabète post-transplantation) : 15-25% des receveurs.
HTA : 80% des patients. IEC ou ARA2 recommandés.
Dyslipidémie : statines recommandées.
Ostéoporose : calcium + vitamine D.

--- FACTEURS DE RISQUE DE REJET ---
Immunologiques : incompatibilités HLA, DSA positifs, PRA > 80%, cross-match positif, rejet antérieur.
Non immunologiques : ischémie froide > 24h, âge donneur > 60 ans, DGF (retard de reprise de fonction).
Receveur : non-observance, infections, HTA non contrôlée, tabagisme.

--- BIOMARQUEURS ET XAI ---
Principaux biomarqueurs prédictifs de rejet :
- Taux résiduel de tacrolimus : cible 8-12 ng/mL (précoce), 5-8 ng/mL (tardif).
- DSA MFI > 3000 : risque élevé de rejet humoral (Banff 2022).
- Créatinine J30 : hausse > 20% doit alerter.
- Protéinurie > 0.5 g/j : mauvais pronostic.
- dd-cfDNA (ADN libre du donneur) : biomarqueur non invasif émergent.
- PRA (Panel Reactive Antibodies) : sensibilisation immunologique.
L'XAI (Explainable AI) avec les valeurs SHAP permet de combiner ces biomarqueurs
pour prédire le risque de rejet avec une interprétabilité clinique.

--- CRITÈRES DE BANFF 2022 ---
Catégorie 1 : Normal ou modifications non spécifiques.
Catégorie 2 : Modifications médiées par les anticorps (AMR) — active ou chronique active.
Catégorie 3 : Lésions borderline (suspectes de TCMR).
Catégorie 4 : Rejet aigu/actif médié par lymphocytes T (TCMR).
Catégorie 5 : Fibrose interstitielle et atrophie tubulaire (IFTA).
Catégorie 6 : Autres changements non considérés comme rejet.

--- SUIVI POST-GREFFE ---
0-3 mois : hebdomadaire — créatinine, tacrolimus, NFS, ionogramme.
3-12 mois : mensuelle — créatinine, protéinurie, taux immunosuppresseurs.
Au-delà de 1 an : trimestrielle — créatinine, protéinurie, DSA, dépistage cancer.
Annuel : biopsie protocolaire, bilan cardiovasculaire complet.

--- RÉSULTATS ET SURVIE ---
Survie greffon à 1 an : >95% (donneur décédé), >98% (donneur vivant).
Survie greffon à 10 ans : ~60% (donneur décédé), ~75% (donneur vivant).
Demi-vie fonctionnelle : 12 ans (décédé), 20 ans (vivant).
Gain espérance de vie vs dialyse : 10-15 ans.
"""

# ── EMBEDDINGS CACHÉS — chargés une seule fois ────────────────────────────────
@lru_cache(maxsize=1)
def get_embeddings():
    print("📦 Chargement du modèle d'embeddings...")
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64}
    )

# ── VECTORSTORE CACHÉ — construit une seule fois ──────────────────────────────
@lru_cache(maxsize=1)
def get_vectorstore():
    print("🔧 Construction du vectorstore...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
    all_docs = [Document(page_content=KNOWLEDGE_BASE, metadata={"source": "base_interne"})]

    # Charger les fichiers locaux
    if os.path.exists(DOCUMENTS_FOLDER):
        for filename in os.listdir(DOCUMENTS_FOLDER):
            filepath = os.path.join(DOCUMENTS_FOLDER, filename)
            try:
                if filename.endswith(".pdf"):
                    from langchain_community.document_loaders import PyPDFLoader
                    docs = PyPDFLoader(filepath).load()
                elif filename.endswith(".txt"):
                    from langchain_community.document_loaders import TextLoader
                    docs = TextLoader(filepath, encoding="utf-8").load()
                else:
                    continue
                for d in docs:
                    d.metadata["source"] = filename
                all_docs.extend(docs)
                print(f"  ✅ {filename} chargé")
            except Exception as e:
                print(f"  ⚠️ {filename} : {e}")

    chunks = splitter.split_documents(all_docs)
    vs = FAISS.from_documents(chunks, get_embeddings())
    print(f"✅ Vectorstore prêt — {len(chunks)} chunks.")
    return vs

# ── CLIENT GROQ CACHÉ ─────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_groq_client():
    return Groq(api_key=GROQ_API_KEY)

# ── INIT AU DÉMARRAGE (warm-up) ───────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    print("🚀 Démarrage MedAssist API — warm-up...")
    get_embeddings()
    get_vectorstore()
    get_groq_client()
    print("✅ API prête et chaude !")

# ── MODÈLES ───────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    history: list = []
    model: str = "llama-3.3-70b-versatile"
    max_tokens: int = 1024

class ChatResponse(BaseModel):
    answer: str
    sources: list

# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "message": "MedAssist API opérationnelle"}

@app.get("/health")
def health():
    vs = get_vectorstore()
    return {"status": "healthy", "chunks_indexed": vs.index.ntotal}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        vs      = get_vectorstore()
        results = vs.similarity_search(req.question, k=4)
        context = "\n\n---\n\n".join(r.page_content for r in results)
        sources = list(set(r.metadata.get("source", "?") for r in results))

        system_prompt = f"""Tu es MedAssist, assistant médical expert spécialisé en transplantation rénale et XAI.
Réponds en français, de façon claire et concise. Base-toi sur le contexte fourni.
Si l'information est absente, utilise tes connaissances générales.

Contexte :
{context}"""

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(req.history[-6:])  # Garder seulement les 6 derniers messages
        messages.append({"role": "user", "content": req.question})

        response = get_groq_client().chat.completions.create(
            model=req.model,
            max_tokens=req.max_tokens,
            messages=messages,
            temperature=0.3,  # Plus déterministe = plus rapide
        )
        return ChatResponse(
            answer=response.choices[0].message.content,
            sources=sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))