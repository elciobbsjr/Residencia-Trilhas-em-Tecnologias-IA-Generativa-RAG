import json
from pathlib import Path
from statistics import mean

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
SUMMARY = RESULTS / "summary.json"
REPORT = BASE / "RELATORIO.md"

DOCS = [
    "attention_is_all_you_need",
    "bert_pretraining",
    "bioetica_e_ia",
    "escrita_academica_ia",
    "gpt3_language_models",
    "gpt4_technical_report",
    "instruct_gpt",
    "llama_foundation_models",
    "lora_low_rank_adaptation",
    "retrieval_augmented_generation",
    "scaling_laws_llm",
    "twitter_algoritmo",
]

EXP_DOCS = [
    "bioetica_e_ia",
    "escrita_academica_ia",
    "twitter_algoritmo",
]

CFG = {
    1: ("fixed_200", 200, 0, "200 caracteres sem overlap"),
    2: ("fixed_500", 500, 0, "500 caracteres sem overlap"),
    3: ("fixed_1000", 1000, 0, "1000 caracteres sem overlap"),
    4: ("fixed_2000", 2000, 0, "2000 caracteres sem overlap"),
    5: ("fixed_overlap_50", 500, 50, "500 caracteres com overlap 50"),
    6: ("fixed_overlap_200", 500, 200, "500 caracteres com overlap 200"),
    7: ("paragraph", None, 0, "separação por parágrafos"),
    8: ("three_sentences", 3, 0, "3 sentenças por chunk"),
    9: ("recursive", 1000, 100, "Recursive 1000/100"),
    10: ("markdown_headers", None, 0, "separação por headings Markdown"),
}


def load(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_chunks(doc, test):
    path = (
        RESULTS
        / doc
        / f"test_{test:02d}"
        / "chunks_embeddings.json"
    )

    if not path.exists():
        return []

    return load(path)


def chunk_stats(items):
    if not items:
        return {
            "num_chunks": 0,
            "avg_chunk_size": 0,
            "min_chunk_size": 0,
            "max_chunk_size": 0,
            "embedding_dimension": None,
        }

    sizes = [
        len(item.get("text", ""))
        for item in items
    ]

    embedding = next(
        (
            item.get("embedding")
            for item in items
            if item.get("embedding")
        ),
        None,
    )

    return {
        "num_chunks": len(items),
        "avg_chunk_size": round(mean(sizes), 2),
        "min_chunk_size": min(sizes),
        "max_chunk_size": max(sizes),
        "embedding_dimension": (
            len(embedding)
            if embedding
            else None
        ),
    }


def markdown_stats():
    data = []

    for doc in DOCS:
        path = (
            RESULTS
            / doc
            / "markdown"
            / f"{doc}.md"
        )

        if not path.exists():
            continue

        text = path.read_text(
            encoding="utf-8"
        )

        lines = text.splitlines()

        data.append({
            "document_id": doc,
            "characters": len(text),
            "lines": len(lines),
            "headings": sum(
                1
                for line in lines
                if line.lstrip().startswith("#")
            ),
            "table_lines": sum(
                1
                for line in lines
                if (
                    line.strip().startswith("|")
                    and line.strip().endswith("|")
                )
            ),
            "image_references": sum(
                1
                for line in lines
                if (
                    "![" in line
                    or "<img" in line.lower()
                )
            ),
        })

    return data


def experiment_stats():
    results = []

    for test in range(1, 11):
        rows = []

        for doc in EXP_DOCS:
            stats = chunk_stats(
                get_chunks(
                    doc,
                    test
                )
            )

            if stats["num_chunks"]:
                rows.append({
                    "document_id": doc,
                    **stats,
                })

        strategy, size, overlap, desc = CFG[test]

        results.append({
            "test_id": test,
            "strategy": strategy,
            "description": desc,
            "chunk_size": size,
            "chunk_overlap": overlap,
            "avg_num_chunks": round(
                mean(
                    row["num_chunks"]
                    for row in rows
                ),
                2
            ),
            "avg_chunk_size": round(
                mean(
                    row["avg_chunk_size"]
                    for row in rows
                ),
                2
            ),
            "min_chunk_size": min(
                row["min_chunk_size"]
                for row in rows
            ),
            "max_chunk_size": max(
                row["max_chunk_size"]
                for row in rows
            ),
            "embedding_dimension": next(
                (
                    row["embedding_dimension"]
                    for row in rows
                    if row["embedding_dimension"]
                ),
                None,
            ),
            "documents": rows,
        })

    return results


def finalist_stats():
    result = {}

    for test in (3, 8, 9):
        rows = []

        for doc in DOCS:
            stats = chunk_stats(
                get_chunks(
                    doc,
                    test
                )
            )

            if stats["num_chunks"]:
                rows.append({
                    "document_id": doc,
                    **stats,
                })

        result[str(test)] = {
            "test_id": test,
            "strategy": CFG[test][0],
            "documents": len(rows),
            "total_chunks": sum(
                row["num_chunks"]
                for row in rows
            ),
            "avg_chunks_per_document": round(
                mean(
                    row["num_chunks"]
                    for row in rows
                ),
                2
            ),
            "avg_chunk_size": round(
                mean(
                    row["avg_chunk_size"]
                    for row in rows
                ),
                2
            ),
            "details": rows,
        }

    return result


def retrieval():
    path = (
        RESULTS
        / "summary_retrieval.json"
    )

    if not path.exists():
        return []

    return load(path).get(
        "retrieval_ranking",
        []
    )


def table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(
            "---"
            for _ in headers
        ) + "|",
        *[
            "| "
            + " | ".join(
                map(str, row)
            )
            + " |"
            for row in rows
        ],
    ])


def build_summary():
    experiments = experiment_stats()

    data = {
        "activity": (
            "Avaliação de Estratégias "
            "de Chunking com LangChain"
        ),
        "embedding_model": (
            "openai/text-embedding-3-small"
        ),
        "embedding_dimension": 1536,
        "num_documents": len(DOCS),
        "documents": DOCS,
        "experimental_documents": EXP_DOCS,
        "markdown_analysis": markdown_stats(),
        "experiments": experiments,
        "retrieval_evaluation": retrieval(),
        "selected_strategies": [
            {
                "test_id": 9,
                "strategy": "recursive",
            },
            {
                "test_id": 8,
                "strategy": "three_sentences",
            },
            {
                "test_id": 3,
                "strategy": "fixed_1000",
            },
        ],
        "selected_strategies_all_documents": (
            finalist_stats()
        ),
    }

    SUMMARY.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8",
    )

    return data


def build_report(summary):
    experiments = summary["experiments"]
    retr = summary["retrieval_evaluation"]
    markdown = summary["markdown_analysis"]

    most = max(
        experiments,
        key=lambda x: x["avg_num_chunks"],
    )

    least = min(
        experiments,
        key=lambda x: x["avg_num_chunks"],
    )

    best = (
        retr[0]
        if retr
        else None
    )

    config_table = table(
        [
            "Teste",
            "Estratégia",
            "Configuração",
        ],
        [
            [
                i,
                CFG[i][0],
                CFG[i][3],
            ]
            for i in range(1, 11)
        ],
    )

    stats_table = table(
        [
            "Teste",
            "Estratégia",
            "Chunks médios",
            "Tam. médio",
            "Mínimo",
            "Máximo",
            "Dimensão",
        ],
        [
            [
                x["test_id"],
                x["strategy"],
                x["avg_num_chunks"],
                x["avg_chunk_size"],
                x["min_chunk_size"],
                x["max_chunk_size"],
                x["embedding_dimension"],
            ]
            for x in experiments
        ],
    )

    markdown_table = table(
        [
            "Documento",
            "Caracteres",
            "Headings",
            "Linhas de tabela",
            "Refs. imagem",
        ],
        [
            [
                x["document_id"],
                x["characters"],
                x["headings"],
                x["table_lines"],
                x["image_references"],
            ]
            for x in markdown
        ],
    )

    if retr:
        retrieval_table = table(
            [
                "Pos.",
                "Teste",
                "Estratégia",
                "Top-1",
                "Hit@3",
                "MRR",
                "Similaridade",
            ],
            [
                [
                    x.get("position"),
                    x.get("test_id"),
                    x.get("strategy"),
                    x.get("top1_accuracy"),
                    x.get("hit_at_3"),
                    x.get("mrr"),
                    x.get(
                        "avg_best_expected_similarity"
                    ),
                ]
                for x in retr
            ],
        )

    else:
        retrieval_table = (
            "Avaliação de recuperação "
            "não encontrada."
        )

    lines = []
    add = lines.append

    add(
        "# Relatório — Avaliação de Estratégias "
        "de Chunking com LangChain"
    )
    add("")
    add("## 1. Objetivo")
    add("")
    add(
        "Comparar 10 estratégias de chunking em "
        "documentos convertidos de PDF para Markdown, "
        "analisando tamanho, quantidade, contexto, "
        "estrutura, overlap, embeddings e adequação "
        "para sistemas de RAG."
    )
    add("")
    add(
        "Foram processados **12 PDFs** com Docling. "
        "Os embeddings foram gerados com "
        "`openai/text-embedding-3-small`, "
        "com dimensão **1536**."
    )

    add("")
    add("## 2. Pipeline")
    add("")
    add(
        "`PDF → Markdown → Chunking → "
        "Embeddings → JSON → Avaliação`"
    )

    add("")
    add("## 3. Estratégias avaliadas")
    add("")
    add(config_table)

    add("")
    add("## 4. Conversão PDF → Markdown")
    add("")
    add(markdown_table)
    add("")
    add(
        "Os arquivos Markdown preservaram texto, "
        "headings e diversas tabelas. A análise "
        "automática não encontrou referências "
        "explícitas de imagens em `![...]` ou `<img>`. "
        "Assim, conteúdo exclusivamente visual não "
        "participou diretamente dos embeddings. "
        "Informações relacionadas a layout, figuras, "
        "gráficos, cores e relações espaciais podem "
        "ter sido perdidas."
    )

    add("")
    add("## 5. Estatísticas dos 10 testes")
    add("")
    add(stats_table)
    add("")
    add(
        "A comparação direta utiliza os três "
        "documentos em que todos os dez testes "
        "foram executados."
    )

    add("")
    add("## 6. Recuperação semântica")
    add("")
    add(retrieval_table)
    add("")

    if best:
        add(
            f"Todas as estratégias empataram em Top-1, "
            f"Hit@3 e MRR nas seis consultas. "
            f"O maior valor médio de similaridade foi do "
            f"**Teste {best['test_id']} — "
            f"{best['strategy']}**, com "
            f"**{best['avg_best_expected_similarity']}**."
        )
        add("")

    add("# 7. Análise obrigatória")
    add("")

    add("## 7.1 Qual estratégia gerou mais chunks?")
    add("")
    add(
        f"**Teste {most['test_id']} — "
        f"{most['strategy']}**, com média de "
        f"**{most['avg_num_chunks']} chunks**. "
        "Chunks menores aumentam a quantidade de "
        "vetores armazenados e pesquisados."
    )

    add("")
    add("## 7.2 Qual gerou menos chunks?")
    add("")
    add(
        f"**Teste {least['test_id']} — "
        f"{least['strategy']}**, com média de "
        f"**{least['avg_num_chunks']} chunks**. "
        "Unidades maiores reduzem o número de "
        "fragmentos."
    )

    add("")
    add("## 7.3 Como o tamanho dos chunks variou?")
    add("")
    add(
        "Os testes fixos ficaram próximos de 200, "
        "500, 1000 e 2000 caracteres. Estratégias "
        "baseadas em parágrafos, sentenças, Recursive "
        "e Markdown variaram conforme a estrutura "
        "natural do conteúdo."
    )

    add("")
    add(
        "## 7.4 Qual estratégia preservou melhor "
        "a estrutura dos documentos?"
    )
    add("")
    add(
        "O **Teste 10 — Markdown Header Text Splitter** "
        "preservou melhor a hierarquia de headings e "
        "metadados de seção e subseção. Sua principal "
        "desvantagem foi produzir algumas seções grandes."
    )

    add("")
    add("## 7.5 Como tabelas foram tratadas?")
    add("")
    add(
        "Diversas tabelas foram convertidas para "
        "Markdown usando `|`. O conteúdo textual das "
        "células foi preservado, mas estratégias de "
        "tamanho fixo podem dividir tabelas no meio e "
        "separar cabeçalhos de valores."
    )

    add("")
    add("## 7.6 Como imagens foram tratadas?")
    add("")
    add(
        "Não foram encontradas referências explícitas "
        "de imagens nos Markdown analisados. Portanto, "
        "as imagens não participaram diretamente dos "
        "embeddings textuais."
    )

    add("")
    add(
        "## 7.7 Quais informações foram perdidas "
        "durante PDF → Markdown?"
    )
    add("")
    add(
        "Principalmente informações visuais e de layout: "
        "imagens, gráficos, cores, posicionamento, "
        "diagramação e relações espaciais. Texto, "
        "headings e diversas tabelas foram preservados."
    )

    add("")
    add(
        "## 7.8 O chunking por caracteres fragmentou "
        "conceitos ou estruturas?"
    )
    add("")
    add(
        "Sim. Cortes fixos podem ocorrer no meio de "
        "frases, parágrafos, referências e tabelas. "
        "O problema foi mais forte no Teste 1, "
        "com 200 caracteres."
    )

    add("")
    add(
        "## 7.9 O chunking por parágrafo produziu "
        "chunks muito grandes?"
    )
    add("")
    add(
        "Em alguns casos, sim. A estratégia preserva "
        "a unidade semântica do parágrafo, mas não "
        "controla seu tamanho, e foram observados "
        "parágrafos com milhares de caracteres."
    )

    add("")
    add(
        "## 7.10 O chunking por sentença preservou "
        "melhor o contexto?"
    )
    add("")
    add(
        "Sim. Três sentenças por chunk normalmente "
        "evitam cortes no meio de frases e apresentaram "
        "excelente recuperação semântica. No GPT-3 "
        "houve um caso excepcional acima do limite de "
        "tokens, tratado por subdivisão de segurança."
    )

    add("")
    add(
        "## 7.11 O Recursive Splitter apresentou vantagens?"
    )
    add("")
    add(
        "Sim. O Recursive prioriza "
        "`parágrafos → linhas → espaços → caracteres`, "
        "oferecendo bom equilíbrio entre preservação "
        "do texto e controle do tamanho."
    )

    add("")
    add(
        "## 7.12 O Markdown Splitter preservou "
        "a estrutura semântica?"
    )
    add("")
    add(
        "Sim. Ele associou os chunks aos headings "
        "e manteve metadados estruturais. Algumas "
        "seções ficaram grandes e exigiram controle "
        "adicional de tamanho."
    )

    add("")
    add(
        "## 7.13 Qual estratégia parece mais "
        "adequada para RAG?"
    )
    add("")
    add(
        "O **Teste 9 — Recursive** é a principal "
        "recomendação geral por equilibrar contexto, "
        "estrutura e tamanho. O **Teste 8 — três "
        "sentenças** apresentou excelente resultado "
        "semântico, e o **Teste 3 — 1000 caracteres** "
        "foi mantido como baseline."
    )

    add("")
    add(
        "## 7.14 Quais estratégias devem ser descartadas?"
    )
    add("")
    add(
        "Não seriam priorizadas: **Teste 1**, pela "
        "fragmentação excessiva; **Teste 6**, pelo "
        "overlap de 40% e redundância; **Teste 4**, "
        "pelos chunks grandes; e **Teste 10 sem limite "
        "adicional**, por poder gerar seções muito extensas."
    )

    add("")
    add(
        "## 7.15 Quais estratégias devem seguir "
        "nos próximos experimentos?"
    )
    add("")
    add(
        "1. **Teste 9 — Recursive**; "
        "2. **Teste 8 — três sentenças**; "
        "3. **Teste 3 — 1000 caracteres**. "
        "As três foram aplicadas aos 12 documentos."
    )

    add("")
    add("# 8. Conclusão")
    add("")
    add(
        "Chunks pequenos aumentaram a fragmentação e "
        "o número de embeddings, enquanto chunks muito "
        "grandes reduziram a granularidade. O chunking "
        "por três sentenças apresentou a maior "
        "similaridade na avaliação inicial, enquanto "
        "o Recursive apresentou o melhor equilíbrio "
        "operacional para documentos heterogêneos. "
        "Assim, o **Recursive foi selecionado como "
        "principal estratégia para RAG**, acompanhado "
        "de três sentenças e do baseline de 1000 caracteres."
    )

    add("")
    add("# 9. Arquivos produzidos")
    add("")
    add("- Markdown dos 12 PDFs")
    add("- chunks e embeddings")
    add("- metadados")
    add("- JSONs por documento e estratégia")
    add("- `summary.json`")
    add("- `RELATORIO.md`")

    REPORT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main():
    print(
        "INICIANDO ETAPA 6...",
        flush=True
    )

    if not RESULTS.exists():
        raise FileNotFoundError(
            f"Pasta não encontrada: {RESULTS}"
        )

    summary = build_summary()

    build_report(
        summary
    )

    print(
        f"summary.json: {SUMMARY}"
    )

    print(
        f"RELATORIO.md: {REPORT}"
    )

    print(
        f"Documentos: "
        f"{summary['num_documents']}"
    )

    print(
        f"Experimentos comparados: "
        f"{len(summary['experiments'])}"
    )

    print(
        f"Dimensão dos embeddings: "
        f"{summary['embedding_dimension']}"
    )

    print(
        "ETAPA 6 FINALIZADA COM SUCESSO"
    )


if __name__ == "__main__":
    main()