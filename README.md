<img src="https://capsule-render.vercel.app/api?type=venom&height=300&color=gradient&customColorList=0,2,2,5,30&text=ANAND%20VISHWAKARMA&fontColor=ffffff&fontSize=60&fontAlign=50&fontAlignY=40&desc=AI%20Engineer%20%E2%80%A2%20GenAI%20Specialist%20%E2%80%A2%20Multi-Agent%20Architect&descFontColor=a0f0ff&descSize=18&descAlign=50&descAlignY=62&animation=twinkling&stroke=0a84ff&strokeWidth=1" width="100%"/>

<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=600&size=20&duration=3000&pause=1500&color=00D4FF&center=true&vCenter=true&width=860&lines=Building+Autonomous+AI+Systems+that+Think+%F0%9F%A7%A0;Engineering+LLMs+that+Act+%7C+Reason+%7C+Deliver+%E2%9A%A1;Turning+Complex+Problems+into+Elegant+AI+Solutions+%F0%9F%8E%AF;From+RAG+Pipelines+to+Multi-Agent+Orchestration+%F0%9F%A4%96" alt="Typing SVG" />

<br/>

[![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/anand-vishwakarma-32a13b293/)
[![Gmail](https://img.shields.io/badge/Gmail-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:anandvishwakarma21j@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Anand21J-V)
[![LeetCode](https://img.shields.io/badge/LeetCode-FFA116?style=for-the-badge&logo=leetcode&logoColor=black)](https://leetcode.com/anandvishwakarma)
[![Profile Views](https://komarev.com/ghpvc/?username=Anand21J-V&color=00d4ff&style=for-the-badge&label=PROFILE+VIEWS)](https://github.com/Anand21J-V)

<br/>

<!-- ✅ LIVE OPEN TO WORK BADGE — Change "YES" to "NO" and color to "red" when not available -->
[![Open To Work](https://img.shields.io/badge/OPEN%20TO%20WORK-YES-00d4ff?style=for-the-badge&logo=handshake&logoColor=white&labelColor=0d1117)](mailto:anandvishwakarma21j@gmail.com)
[![Hire Me](https://img.shields.io/badge/AI%20%2F%20GenAI%20%2F%20Data%20Engineering-Roles%20I'm%20Targeting-6366f1?style=for-the-badge&logo=target&logoColor=white&labelColor=0d1117)](mailto:anandvishwakarma21j@gmail.com)

</div>

<br/>

---

## ◈ IDENTITY MATRIX

```
╔══════════════════════════════════════════════════════════════════════════════╗
║  SYSTEM PROFILE :: ANAND VISHWAKARMA                                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Role         →  AI Engineer & GenAI Specialist                             ║
║  Focus        →  Multi-Agent Systems | LLMs | Data Engineering              ║
║  Institution  →  O.P. Jindal University, CSE  [CGPA: 9.38 / 10.0]          ║
║  Status       →  AI Engineering Intern @ Sentius Technologies  [ACTIVE 🟢]  ║
║  Certified    →  Oracle Cloud Infrastructure 2025 GenAI Professional        ║
║  Hackathons   →  4× Finalist  [SIH · GIET · HackVyuha · more]              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

<div align="center">

| 🤖 Agentic AI | 🔬 GenAI & RAG | 🗣️ NLP & ASR | 👁️ Computer Vision | ⚙️ MLOps |
|:-:|:-:|:-:|:-:|:-:|
| LangGraph · CrewAI | RAG · FAISS · LlamaIndex | Whisper · gTTS · BERT | OpenCV · Gemini Pro | MLflow · Docker · CI/CD |

</div>

---

## ◈ 💭 MY ENGINEERING PHILOSOPHY

> *How I think about building AI systems*

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│   "I don't build AI that just responds — I build AI that reasons, plans,        │
│    and acts. Every system I design starts with one question: what should         │
│    this agent do when things go wrong? Resilience is not a feature,             │
│    it's the foundation."                                                         │
│                                                                                 │
│    My approach is simple:                                                        │
│    → Break hard problems into agent-sized subtasks                              │
│    → Make every pipeline observable and debuggable from day one                 │
│    → Ship to production early, evaluate rigorously, iterate fast                │
│    → Prefer composable, modular systems over monolithic complexity              │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

</div>

---

## ◈ 🏗️ MULTI-AGENT ARCHITECTURE DEEP DIVE

> How I architect agentic AI systems — a real-world pattern I use across projects

```mermaid
graph TD
    U([👤 User Input]) --> O

    subgraph ORCHESTRATION ["🧠 Orchestration Layer  (LangGraph / OpenAI Agents SDK)"]
        O[Planner Agent] --> |Subtask 1| A
        O --> |Subtask 2| B
        O --> |Subtask 3| C
    end

    subgraph AGENTS ["⚡ Specialist Agents"]
        A[🔍 Research Agent\nRAG + Vector Search]
        B[💻 Coder Agent\nCode Generation]
        C[📊 Analyst Agent\nNL2SQL + Insights]
    end

    subgraph TOOLS ["🛠️ Tool Layer"]
        T1[(Vector DB\nFAISS / Pinecone)]
        T2[🌐 Web Search\nReal-time Data]
        T3[(SQL Database\nPostgres / MySQL)]
        T4[📁 File System\nDocs / CSV]
    end

    subgraph MEMORY ["🗃️ Memory & State"]
        M1[Short-term\nConversation Buffer]
        M2[Long-term\nVector Memory]
    end

    A <--> T1
    A <--> T2
    B <--> T4
    C <--> T3

    A & B & C <--> M1
    M1 <--> M2

    A --> R
    B --> R
    C --> R

    R[🔁 Reviewer Agent\nQuality + Hallucination Check] --> |Pass| OUT
    R --> |Fail — Retry| O

    OUT([✅ Final Output\nto User])

    style ORCHESTRATION fill:#0d1f3c,stroke:#00d4ff,stroke-width:2px,color:#ffffff
    style AGENTS fill:#0d2d1f,stroke:#00ff88,stroke-width:2px,color:#ffffff
    style TOOLS fill:#1f0d2d,stroke:#a78bfa,stroke-width:2px,color:#ffffff
    style MEMORY fill:#2d1f0d,stroke:#ffaa00,stroke-width:2px,color:#ffffff
    style U fill:#00d4ff,stroke:#00d4ff,color:#000000
    style OUT fill:#00ff88,stroke:#00ff88,color:#000000
    style R fill:#ff6b6b,stroke:#ff6b6b,color:#000000
```

---

## ◈ 🚀 IMPACT AT A GLANCE

<div align="center">

| ⚡ Metric | 📈 Result |
|:---------|:---------|
| 🎯 Inventory Error Reduction | **90%+** via AI-powered OCR & Computer Vision |
| 🔍 Search Retrieval Quality | **Top-10 real-time** clinical study retrieval with vector similarity |
| 🤖 Agents Shipped to Production | **5+ multi-agent systems** across 2 internships |
| 🏆 Competitive AI Hackathons | **4× National Finalist** (SIH, GIET, HackVyuha & more) |
| 📚 Academic Performance | **9.38 / 10.0 CGPA** — Top of cohort |
| 🌐 APIs Deployed | **Production-grade FastAPI** endpoints live on Render |
| 🧩 Frameworks Mastered | **10+ AI/ML frameworks** across GenAI, MLOps & NLP |

</div>

---

## ◈ 🗓️ CAREER TIMELINE

<div align="center">

```
  2022               2023                2025 (Jan–Jun)      2025 (Jul–Sep)        2026 →
   │                  │                      │                    │                   │
   ▼                  ▼                      ▼                    ▼                   ▼
┌─────────┐     ┌──────────┐          ┌───────────┐        ┌───────────┐       ┌───────────┐
│ B.Tech  │     │   SIH    │          │  Oracle   │        │  Inflera  │       │  Sentius  │
│  CSE    │     │Hackathon │          │  GenAI    │        │  Tech.    │       │  Tech.    │
│ O.P.    │     │Finalist  │          │  Cert.    │        │  Intern   │       │  Intern   │
│ Jindal  │     │ National │          │ ☁️ Cloud  │        │ LangGraph │       │ Pipelines │
│  CGPA:  │     │  Level   │          │   2025    │        │ Agents &  │       │ Semantic  │
│ 9.38/10 │     │          │          │           │        │  FinOps   │       │ Search AI │
└─────────┘     └──────────┘          └───────────┘        └───────────┘       └───────────┘
    🎓               🏆                    🥇                   💼                  💼 🟢
```

</div>

---

## ◈ 🔬 CURRENTLY IN THE LAB

<div align="center">

| 🚧 Building | 📖 Exploring | 🎯 Next Goal |
|:-----------:|:------------:|:------------:|
| Semantic Search with Custom Re-ranking | Agentic RAG & Self-healing Pipelines | Deploy End-to-End Multi-Agent Product |
| FastAPI Production Data Pipelines | LLM Evaluation Frameworks (DeepEval) | Contribute to Open-Source AI Tooling |
| Transformer-based Summarization Systems | Advanced Prompt Chaining Strategies | Oracle Cloud Advanced Certification |

</div>

---

## ◈ EXPERIENCE

<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=0,2,2,5,30&height=3&width=100%" width="100%"/>

### ▸ SENTIUS TECHNOLOGIES PVT. LTD. &nbsp;|&nbsp; `AI Engineering Intern` &nbsp;|&nbsp; 🟢 Jan 2026 — Present

> **Building production-grade AI pipelines and intelligent search systems**

| Module | What Was Built | Impact |
|--------|---------------|--------|
| 🚂 **Data Pipelines** | Large-scale bus · train · flight web scraping via FastAPI, deployed on Render | Production-ready scalable APIs |
| 🎬 **YouTube AI Summarizer** | Transformer-based video summarization with structured metadata to CSV | Downstream analytics-ready |
| 🔍 **Visual Semantic Search** | Custom re-ranking logic + real-time data ingestion | Outperforms embedding-only baselines |

`FastAPI` `Transformers` `Web Scraping` `Render` `Semantic Search` `Re-ranking` `Pipeline Orchestration`

<br/>

### ▸ INFLERA TECHNOLOGIES PTY LIMITED &nbsp;|&nbsp; `AI Engineering Intern` &nbsp;|&nbsp; ⚪ Jul 2025 — Sep 2025 &nbsp;·&nbsp; [Certificate](#)

> **Architected multi-agent systems and intelligent workflow automation**

| Module | What Was Built | Impact |
|--------|---------------|--------|
| 💰 **FinOps Assistant** | LangGraph cost analysis & anomaly detection agent | Automated financial insights |
| 🗄️ **Schema-Agent** | Intelligent DB schema analysis & optimization | Smarter SQL pipelines |
| 🔐 **BYOD Workflow** | Secure data ingestion with token-based access control | Enterprise-grade security |
| 📊 **Viz Agents** | Chart + breakdown + feature-flagged automation agents | Real-time visual reporting |

`LangGraph` `LangSmith` `NL2SQL` `Streamlit` `OpenAI API` `Groq API` `Prompt Engineering` `LLM Evaluation`

<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=0,2,2,5,30&height=3&width=100%" width="100%"/>

---

## ◈ PROJECT SHOWCASE

<table>
<tr>
<td width="50%" valign="top">

### 🤖 LLM Coding Agent Assistant
`Multi-Agent` `LLM Automation`&nbsp;&nbsp;[![GitHub](https://img.shields.io/badge/Code-121011?style=flat-square&logo=github&logoColor=white)](https://github.com/Anand21J-V)

> Autonomous dev system with 3 specialized agents in pipeline

```
🎯 Planner  →  Breaks task into subtasks
💻 Coder    →  Generates production files
🔍 Reviewer →  LLM-powered code review
⚡ Runner   →  Async orchestration layer
```

**Stack:** `OpenAI Agents SDK` `Gemini API` `Streamlit` `AsyncIO` `Pydantic`

</td>
<td width="50%" valign="top">

### 🏥 Medical Insights Platform
`NLP` `RAG` `Semantic Search`&nbsp;&nbsp;[![GitHub](https://img.shields.io/badge/Code-121011?style=flat-square&logo=github&logoColor=white)](https://github.com/Anand21J-V)

> Clinical trial search engine using vector embeddings + RAG

```
📚 Ingest   →  PDFMiner clinical data
🔤 Embed    →  MiniLM-L6-v2 vectors
🗃️ Index    →  FAISS similarity store
🔍 Retrieve →  Groq LLM answers
```

**Stack:** `LangChain` `FAISS` `Sentence-BERT` `Groq` `Streamlit`

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 🎙️ Multilingual Meeting Assistant
`ASR/TTS` `NLP` `Real-Time`&nbsp;&nbsp;[![GitHub](https://img.shields.io/badge/Code-121011?style=flat-square&logo=github&logoColor=white)](https://github.com/Anand21J-V)

> End-to-end speech → summary → audio pipeline for meetings

```
🎤 Whisper  →  High-accuracy ASR
📝 Groq LLM →  Multilingual summary
🔊 gTTS     →  Audio generation
💾 Flask    →  Session management
```

**Stack:** `Whisper` `Groq LLM` `gTTS` `Pydub` `Flask` `Streamlit`

</td>
<td width="50%" valign="top">

### 📦 Smart Product Scanning System
`Computer Vision` `GenAI`&nbsp;&nbsp;[![GitHub](https://img.shields.io/badge/Code-121011?style=flat-square&logo=github&logoColor=white)](https://github.com/Anand21J-V)

> Real-time AI inventory system — 90%+ fewer manual errors

```
📷 OpenCV   →  Live camera feed
🔍 Gemini   →  Intelligent OCR
💾 Postgres →  Inventory storage
☁️ Azure    →  Cloud deployment
```

**Stack:** `Gemini Pro` `OpenCV` `Flask` `PostgreSQL` `Azure`

</td>
</tr>
</table>

---

## ◈ TECHNICAL ARSENAL

<div align="center">

### ━━━ 🤖 Generative AI & Agentic Frameworks ━━━

![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-00D4FF?style=for-the-badge&logo=buffer&logoColor=black)
![CrewAI](https://img.shields.io/badge/CrewAI-6C63FF?style=for-the-badge&logo=rocket&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)
![Autogen](https://img.shields.io/badge/Autogen-000000?style=for-the-badge&logo=microsoft&logoColor=white)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-7F52FF?style=for-the-badge&logo=serverless&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-F55036?style=for-the-badge&logo=thunderstorm&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Agents SDK](https://img.shields.io/badge/Agents_SDK-412991?style=for-the-badge&logo=openai&logoColor=white)
![LangSmith](https://img.shields.io/badge/LangSmith-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![LangFlow](https://img.shields.io/badge/LangFlow-FF6B6B?style=for-the-badge&logo=datadog&logoColor=white)

### ━━━ 🧠 Machine Learning & Deep Learning ━━━

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![FAISS](https://img.shields.io/badge/FAISS_Meta-0467DF?style=for-the-badge&logo=meta&logoColor=white)
![Whisper](https://img.shields.io/badge/Whisper_OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)

### ━━━ 📊 Data Science & Analytics ━━━

![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=plotly&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

### ━━━ ⚙️ MLOps & Dev Tools ━━━

![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![MLFlow](https://img.shields.io/badge/MLFlow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![DagsHub](https://img.shields.io/badge/DagsHub-2D7FF9?style=for-the-badge&logo=git&logoColor=white)
![DeepEval](https://img.shields.io/badge/DeepEval-FF6B6B?style=for-the-badge&logo=checkmarx&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)

### ━━━ 🌐 Backend & Cloud ━━━

![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Render](https://img.shields.io/badge/Render-46E3B7?style=for-the-badge&logo=render&logoColor=black)
![Azure](https://img.shields.io/badge/Azure-0078D4?style=for-the-badge&logo=microsoftazure&logoColor=white)

### ━━━ 🗄️ Databases ━━━

![MySQL](https://img.shields.io/badge/MySQL-4479A1?style=for-the-badge&logo=mysql&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-336791?style=for-the-badge&logo=postgresql&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-47A248?style=for-the-badge&logo=mongodb&logoColor=white)
![Supabase](https://img.shields.io/badge/Supabase-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)
![VectorDB](https://img.shields.io/badge/Vector_DB-FF6B6B?style=for-the-badge&logo=pinecone&logoColor=white)

</div>

---

## ◈ SKILL PROFICIENCY

<div align="center">

**🐍 Python & Data Engineering**
![92%](https://geps.dev/progress/92?dangerColor=00d4ff&warningColor=00d4ff&successColor=00d4ff)

**🤖 Agentic AI & LLMs**
![90%](https://geps.dev/progress/90?dangerColor=00d4ff&warningColor=00d4ff&successColor=00d4ff)

**🗣️ NLP & Transformers**
![89%](https://geps.dev/progress/85?dangerColor=00d4ff&warningColor=00d4ff&successColor=00d4ff)

**🔬 RAG & Semantic Search**
![88%](https://geps.dev/progress/85?dangerColor=00d4ff&warningColor=00d4ff&successColor=00d4ff)

**🌐 Backend APIs (FastAPI / Flask)**
![85%](https://geps.dev/progress/82?dangerColor=00d4ff&warningColor=00d4ff&successColor=00d4ff)

**🧠 Machine Learning & Deep Learning**
![85%](https://geps.dev/progress/80?dangerColor=00d4ff&warningColor=00d4ff&successColor=00d4ff)

**⚙️ MLOps & Deployment**
![82%](https://geps.dev/progress/78?dangerColor=00d4ff&warningColor=00d4ff&successColor=00d4ff)

**👁️ Computer Vision & OCR**
![81%](https://geps.dev/progress/75?dangerColor=00d4ff&warningColor=00d4ff&successColor=00d4ff)

</div>

---

## ◈ GITHUB INTELLIGENCE

<div align="center">

<img src="https://github-readme-stats.vercel.app/api?username=Anand21J-V&show_icons=true&theme=tokyonight&hide_border=true&count_private=true&include_all_commits=true&bg_color=0d1117&title_color=00d4ff&icon_color=00d4ff&text_color=c9d1d9&border_radius=12" height="180"/>
&nbsp;&nbsp;
<img src="https://github-readme-stats.vercel.app/api/top-langs/?username=Anand21J-V&theme=tokyonight&hide_border=true&layout=compact&langs_count=8&bg_color=0d1117&title_color=00d4ff&text_color=c9d1d9&border_radius=12" height="180"/>

<br/>

<img src="https://github-readme-streak-stats.herokuapp.com/?user=Anand21J-V&theme=tokyonight&hide_border=true&background=0d1117&ring=00d4ff&fire=ff6b6b&currStreakLabel=00d4ff&sideLabels=c9d1d9&dates=8b949e&border_radius=12" width="70%"/>

<br/>

<img src="https://github-readme-activity-graph.vercel.app/graph?username=Anand21J-V&bg_color=0d1117&color=00d4ff&line=00d4ff&point=ffffff&area=true&area_color=00d4ff&hide_border=true&radius=12" width="95%"/>

<br/>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Anand21J-V/Anand21J-V/output/github-contribution-grid-snake-dark.svg" />
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/Anand21J-V/Anand21J-V/output/github-contribution-grid-snake.svg" />
  <img alt="github contribution grid snake animation" src="https://raw.githubusercontent.com/Anand21J-V/Anand21J-V/output/github-contribution-grid-snake.svg" width="95%"/>
</picture>

<br/>

<img src="https://github-profile-trophy.vercel.app/?username=Anand21J-V&theme=tokyonight&no-frame=true&no-bg=true&row=1&column=7&margin-w=12" width="100%"/>

</div>

---

## ◈ LEETCODE ARENA

<div align="center">

<img src="https://leetcard.jacoblin.cool/anandvishwakarma?theme=dark&font=JetBrains%20Mono&ext=heatmap&border=0&radius=12&bgcolor=0d1117&border_color=00d4ff" width="60%"/>

</div>

---

## ◈ ACHIEVEMENTS

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────┐
│  🥇  ORACLE CLOUD INFRASTRUCTURE 2025  ·  Certified GenAI Professional  │
│  🏆  SMART INDIA HACKATHON 2023        ·  National Finalist              │
│  🏆  GIET INNOVATION HACKATHON X 4.0  ·  Finalist  ·  2025              │
│  🏆  GEEKFORGEEKS HACKVYUVA 2025      ·  Finalist                        │
└─────────────────────────────────────────────────────────────────────────┘
```

</div>

---

## ◈ EDUCATION

<div align="center">

```
╔══════════════════════════════════════════════════════════╗
║   🎓  O.P. JINDAL UNIVERSITY  ·  Chhattisgarh, India    ║
║   B.Tech · Computer Science & Engineering                ║
║   July 2022 → Present          CGPA: 9.38 / 10.0  ✦     ║
║   Specialization: AI · Machine Learning · Data Science   ║
╚══════════════════════════════════════════════════════════╝
```

</div>

---

## ◈ LET'S BUILD SOMETHING EXTRAORDINARY

<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&size=16&duration=3500&pause=1000&color=00D4FF&center=true&vCenter=true&width=700&lines=Open+to+AI+Engineering+%7C+GenAI+%7C+Data+Engineering+roles;Let's+connect+and+build+the+future+together+%F0%9F%9A%80" />

<br/>

[![LinkedIn](https://img.shields.io/badge/Connect_on_LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/anand-vishwakarma-32a13b293/)
[![Email](https://img.shields.io/badge/Send_an_Email-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:anandvishwakarma21j@gmail.com)
[![GitHub](https://img.shields.io/badge/Follow_on_GitHub-121011?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Anand21J-V)
[![LeetCode](https://img.shields.io/badge/LeetCode_Profile-FFA116?style=for-the-badge&logo=leetcode&logoColor=black)](https://leetcode.com/anandvishwakarma)

**📞 +91 7011472391**

<br/>

<img src="https://quotes-github-readme.vercel.app/api?type=horizontal&theme=tokyonight" width="70%"/>

</div>

<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=0,2,2,5,30&height=120&section=footer&text=Built%20with%20Intelligence%20%C2%B7%20Passion%20%C2%B7%20Coffee&fontColor=a0f0ff&fontSize=16&fontAlign=50&fontAlignY=70&animation=twinkling" width="100%"/>
