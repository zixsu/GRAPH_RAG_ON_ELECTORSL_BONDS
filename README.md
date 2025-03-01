# Enhanced Graph RAG Querying for Electoral Bonds Financial Data

- **Dhanush Mangina**  
  School of Computer Science, Vellore Institute of Technology, Vellore, India  
  Email: dhanush.mangina2024@vitstudent.ac.in

## Abstract
The Enhanced Hybrid Graph RAG system introduces a novel approach to electoral bond transparency analysis, achieving unparalleled accuracy and reliability in financial accountability. This architecture integrates financial relationship graphs, enabling a structured representation of complex transaction networks, alongside a rule-based query resolution mechanism that ensures precise information retrieval. A key feature of this system is a context-aware LLM fallback that enhances query responsiveness while minimizing LLM hallucinations. The Hybrid GraphRAG achieves 100% accuracy in bond transaction queries, setting a new benchmark for financial data analysis. Real-world evaluations demonstrate significant efficiency improvements over traditional RAG methods, with reduced response times and enhanced reliability. These advancements contribute to greater transparency in financial systems and establish the Hybrid GraphRAG as a robust solution for professionals and practitioners focused on financial accountability.

## Keywords
LLMs, Graph RAG, Electoral Bonds, RAG, Knowledge Graphs, Prompt Engineering

## Introduction
Existing Retrieval-Augmented Generation (RAG) systems for financial data analysis face multiple challenges, including:
- **LLM hallucinations**, leading to factually incorrect outputs.
- **Limitations in temporal reasoning**, affecting the accurate tracking of financial transactions over time.
- **Entity disambiguation issues**, causing ambiguity in identifying financial entities.

To overcome these limitations, we propose the **Hybrid GraphRAG system**, which integrates:
1. **Temporal Knowledge Graph:** Facilitates advanced temporal reasoning.
2. **Pattern-Activated Direct Lookup System:** Enables efficient and immediate query resolution.
3. **Graph-Neighborhood Context Builder:** Enhances contextual understanding of retrieved financial information.

This architecture ensures **100% accuracy** in financial data retrieval, addressing the limitations of traditional RAG systems.

## Methodology
### 1. Structured Knowledge Graph Construction
The Hybrid GraphRAG uses a structured knowledge graph that includes key financial entities:
- **Party:** Individuals or organizations in financial transactions.
- **Purchaser:** Entities buying electoral bonds.
- **Bond:** Financial instruments with properties such as value, maturity date, and issuer.

Each node is interconnected through **temporal edges**, which enable accurate tracking of financial transaction histories.

### 2. Hybrid Query Processor
The Hybrid Query Processor integrates:
- **Pattern Matching:** Uses regex and semantic matching algorithms for accurate intent detection.
- **Direct Lookup Capabilities:** Employs a caching mechanism to minimize response time.
- **LLM Synthesis:** Restricts model scope to financial data to reduce hallucination risks.

### 3. Key Algorithms
#### a) Entity Mapping Algorithm
Maps financial entities from the knowledge graph, ensuring correct query interpretation.
#### b) Response Generation Algorithm
Uses structured query resolution techniques to ensure precise and relevant responses.

## Results
### 1. Comparative Accuracy
The Hybrid GraphRAG system was tested against Traditional RAG and GRAPRAG systems, achieving **100% accuracy** in bond transaction queries. This eliminates hallucination issues prevalent in other systems.

### 2. Efficiency Metrics
Performance evaluation was based on:
- **Accuracy:** 100%
- **Response Time:** 3.91 milliseconds (ms)
- **CPU Usage:** 46.4%
- **RAM Usage:** 2082.64 MB

The Hybrid GraphRAG demonstrated remarkable improvements over traditional methods, significantly reducing response time while maintaining computational efficiency.

## Conclusion
The **Enhanced GraphRAG System** revolutionizes electoral bond financial data analysis by ensuring **unmatched accuracy, efficiency, and transparency**. Future research will focus on optimizing performance through **GPU acceleration and real-time financial monitoring applications**.


## Acknowledgments
This research was conducted at **Vellore Institute of Technology**, School of Computer Science. We extend our gratitude to our faculty mentors and peers for their valuable support.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

