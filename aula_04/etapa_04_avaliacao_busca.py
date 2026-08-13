import json
import os
from pathlib import Path
from statistics import mean

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# CAMINHOS
# ============================================================

PASTA_ATUAL = Path(__file__).resolve().parent
ROOT = PASTA_ATUAL.parent

PASTA_RESULTS = PASTA_ATUAL / "results"

ARQUIVO_SUMMARY_CHUNKING = (
    PASTA_RESULTS
    / "summary_chunking_preliminar.json"
)


# ============================================================
# .ENV
# ============================================================

load_dotenv(ROOT / ".env")

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)

EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "openai/text-embedding-3-small"
)

if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY não encontrada no .env."
    )


# ============================================================
# CLIENTE OPENROUTER
# ============================================================

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENAI_API_KEY
)


# ============================================================
# DOCUMENTOS
# ============================================================

DOCUMENTOS = [
    "bioetica_e_ia",
    "escrita_academica_ia",
    "twitter_algoritmo",
]

TOTAL_TESTES = 10
TOP_K = 3


# ============================================================
# QUERIES DE AVALIAÇÃO
# ============================================================
#
# Cada pergunta possui um documento que esperamos que
# seja recuperado pela busca semântica.
#
# Vamos pesquisar em TODOS os três documentos juntos.
# ============================================================

QUERIES = [
    {
        "query_id": "q01",
        "query": (
            "Como a opacidade dos algoritmos de inteligência "
            "artificial pode afetar a autonomia do paciente "
            "e o consentimento informado?"
        ),
        "expected_document": "bioetica_e_ia",
    },

    {
        "query_id": "q02",
        "query": (
            "Quais riscos éticos surgem quando sistemas de "
            "inteligência artificial na medicina reproduzem "
            "vieses presentes nos dados de treinamento?"
        ),
        "expected_document": "bioetica_e_ia",
    },

    {
        "query_id": "q03",
        "query": (
            "Para que serve o diário de bordo da IA durante "
            "o processo de escrita acadêmica?"
        ),
        "expected_document": "escrita_academica_ia",
    },

    {
        "query_id": "q04",
        "query": (
            "Por que o pesquisador deve realizar apropriação "
            "e reescrita autoral de textos produzidos com "
            "auxílio de inteligência artificial?"
        ),
        "expected_document": "escrita_academica_ia",
    },

    {
        "query_id": "q05",
        "query": (
            "Como os algoritmos de curadoria e moderação "
            "do Twitter ou X influenciam a formação "
            "do debate público?"
        ),
        "expected_document": "twitter_algoritmo",
    },

    {
        "query_id": "q06",
        "query": (
            "Quais consequências ocorreram após mudanças "
            "nas políticas de moderação do Twitter ou X?"
        ),
        "expected_document": "twitter_algoritmo",
    },
]


# ============================================================
# NOMES DOS TESTES
# ============================================================

ESTRATEGIAS = {
    1: "fixed_200",
    2: "fixed_500",
    3: "fixed_1000",
    4: "fixed_2000",
    5: "fixed_overlap_50",
    6: "fixed_overlap_200",
    7: "paragraph",
    8: "three_sentences",
    9: "recursive",
    10: "markdown_headers",
}


# ============================================================
# SIMILARIDADE DE COSSENO
# ============================================================

def similaridade_cosseno(
    vec1,
    vec2
):
    """
    Calcula a similaridade de cosseno entre dois vetores.
    """

    vec1 = np.asarray(
        vec1,
        dtype=np.float32
    )

    vec2 = np.asarray(
        vec2,
        dtype=np.float32
    )

    norma1 = np.linalg.norm(
        vec1
    )

    norma2 = np.linalg.norm(
        vec2
    )

    if norma1 == 0 or norma2 == 0:
        return 0.0

    return float(
        np.dot(vec1, vec2)
        / (norma1 * norma2)
    )


# ============================================================
# GERAR EMBEDDINGS DAS QUERIES
# ============================================================

def gerar_embeddings_queries():
    """
    Gera os embeddings das perguntas em uma única chamada.
    """

    print(
        "\nGerando embeddings das queries..."
    )

    textos = [
        item["query"]
        for item in QUERIES
    ]

    resposta = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=textos
    )

    embeddings = [
        item.embedding
        for item in resposta.data
    ]

    if len(embeddings) != len(QUERIES):

        raise RuntimeError(
            "Quantidade incorreta de embeddings de queries."
        )

    resultado = {}

    for query, embedding in zip(
        QUERIES,
        embeddings
    ):

        resultado[
            query["query_id"]
        ] = embedding

    print(
        f"Queries processadas: "
        f"{len(resultado)}"
    )

    print(
        f"Dimensão: "
        f"{len(embeddings[0])}"
    )

    return resultado


# ============================================================
# CARREGAR CHUNKS DE UM TESTE
# ============================================================

def carregar_corpus_teste(
    test_id
):
    """
    Para determinado teste, reúne os chunks dos
    três documentos em um único corpus.
    """

    corpus = []

    for document_id in DOCUMENTOS:

        caminho = (
            PASTA_RESULTS
            / document_id
            / f"test_{test_id:02d}"
            / "chunks_embeddings.json"
        )

        if not caminho.exists():

            raise FileNotFoundError(
                f"Arquivo não encontrado:\n{caminho}"
            )

        with caminho.open(
            "r",
            encoding="utf-8"
        ) as arquivo:

            chunks = json.load(
                arquivo
            )

        for chunk in chunks:

            if not chunk.get(
                "embedding"
            ):
                raise ValueError(
                    f"Chunk sem embedding: "
                    f"{chunk.get('chunk_id')}"
                )

            corpus.append(
                chunk
            )

    return corpus


# ============================================================
# PREPARAR MATRIZ DOS EMBEDDINGS
# ============================================================

def preparar_matriz_embeddings(
    corpus
):
    """
    Converte os embeddings dos chunks em uma matriz NumPy
    e normaliza os vetores.

    Isso torna a busca mais rápida.
    """

    matriz = np.array(
        [
            chunk["embedding"]
            for chunk in corpus
        ],
        dtype=np.float32
    )

    normas = np.linalg.norm(
        matriz,
        axis=1,
        keepdims=True
    )

    normas = np.maximum(
        normas,
        1e-12
    )

    matriz_normalizada = (
        matriz / normas
    )

    return matriz_normalizada


# ============================================================
# BUSCA SEMÂNTICA
# ============================================================

def buscar(
    query_embedding,
    corpus,
    matriz_normalizada,
    top_k=TOP_K
):
    """
    Realiza a busca semântica utilizando similaridade
    de cosseno.
    """

    query_vector = np.asarray(
        query_embedding,
        dtype=np.float32
    )

    norma_query = np.linalg.norm(
        query_vector
    )

    if norma_query == 0:
        raise ValueError(
            "Embedding da query possui norma zero."
        )

    query_vector = (
        query_vector
        / norma_query
    )

    # Como os vetores estão normalizados,
    # produto escalar = similaridade de cosseno.
    scores = (
        matriz_normalizada
        @ query_vector
    )

    ranking = np.argsort(
        scores
    )[::-1]

    top_indices = ranking[
        :top_k
    ]

    resultados = []

    for posicao, indice in enumerate(
        top_indices,
        start=1
    ):

        chunk = corpus[
            int(indice)
        ]

        texto = chunk[
            "text"
        ]

        resultados.append({
            "rank":
                posicao,

            "score":
                round(
                    float(
                        scores[indice]
                    ),
                    6
                ),

            "chunk_id":
                chunk[
                    "chunk_id"
                ],

            "document_id":
                chunk[
                    "document_id"
                ],

            "text":
                texto,

            "metadata":
                chunk.get(
                    "metadata",
                    {}
                ),
        })

    return (
        resultados,
        ranking,
        scores
    )


# ============================================================
# AVALIAR UMA QUERY
# ============================================================

def avaliar_query(
    query,
    query_embedding,
    corpus,
    matriz_normalizada
):
    """
    Executa a busca e calcula métricas simples de recuperação.
    """

    resultados, ranking, scores = (
        buscar(
            query_embedding,
            corpus,
            matriz_normalizada,
            TOP_K
        )
    )

    expected_document = query[
        "expected_document"
    ]

    # --------------------------------------------------------
    # TOP 1
    # --------------------------------------------------------

    top1_document = (
        resultados[0][
            "document_id"
        ]
    )

    top1_correct = (
        top1_document
        == expected_document
    )

    # --------------------------------------------------------
    # HIT@3
    # --------------------------------------------------------

    hit_at_3 = any(
        resultado[
            "document_id"
        ]
        == expected_document

        for resultado
        in resultados
    )

    # --------------------------------------------------------
    # RECIPROCAL RANK
    # --------------------------------------------------------

    first_relevant_rank = None

    for posicao, indice in enumerate(
        ranking,
        start=1
    ):

        chunk = corpus[
            int(indice)
        ]

        if (
            chunk["document_id"]
            == expected_document
        ):

            first_relevant_rank = (
                posicao
            )

            break

    if first_relevant_rank:

        reciprocal_rank = (
            1.0
            / first_relevant_rank
        )

    else:

        reciprocal_rank = 0.0

    # --------------------------------------------------------
    # MELHOR SCORE DO DOCUMENTO ESPERADO
    # --------------------------------------------------------

    scores_documento = []

    for indice, chunk in enumerate(
        corpus
    ):

        if (
            chunk["document_id"]
            == expected_document
        ):

            scores_documento.append(
                float(
                    scores[indice]
                )
            )

    best_expected_score = max(
        scores_documento
    )

    return {
        "query_id":
            query["query_id"],

        "query":
            query["query"],

        "expected_document":
            expected_document,

        "top1_document":
            top1_document,

        "top1_correct":
            top1_correct,

        "hit_at_3":
            hit_at_3,

        "first_relevant_rank":
            first_relevant_rank,

        "reciprocal_rank":
            round(
                reciprocal_rank,
                6
            ),

        "top1_similarity":
            resultados[0][
                "score"
            ],

        "best_expected_similarity":
            round(
                best_expected_score,
                6
            ),

        "top3":
            resultados,
    }


# ============================================================
# ESTATÍSTICAS DE CHUNKING
# ============================================================

def carregar_estatisticas_chunking():
    """
    Recupera as estatísticas produzidas na Etapa 2.
    """

    if not ARQUIVO_SUMMARY_CHUNKING.exists():

        return {}

    with ARQUIVO_SUMMARY_CHUNKING.open(
        "r",
        encoding="utf-8"
    ) as arquivo:

        dados = json.load(
            arquivo
        )

    agrupado = {}

    for item in dados[
        "results"
    ]:

        test_id = item[
            "test_id"
        ]

        if test_id not in agrupado:

            agrupado[
                test_id
            ] = []

        agrupado[
            test_id
        ].append(
            item
        )

    estatisticas = {}

    for test_id, itens in (
        agrupado.items()
    ):

        estatisticas[
            test_id
        ] = {
            "avg_num_chunks": round(
                mean(
                    item[
                        "num_chunks"
                    ]
                    for item in itens
                ),
                2
            ),

            "avg_chunk_size": round(
                mean(
                    item[
                        "avg_chunk_size"
                    ]
                    for item in itens
                ),
                2
            ),

            "avg_tokens": round(
                mean(
                    item[
                        "avg_tokens"
                    ]
                    for item in itens
                ),
                2
            ),
        }

    return estatisticas


# ============================================================
# EXECUTAR AVALIAÇÃO
# ============================================================

def executar_avaliacao():

    print("\n")
    print("=" * 70)
    print(
        "ATIVIDADE 04 - ETAPA 4"
    )
    print(
        "AVALIAÇÃO DA RECUPERAÇÃO SEMÂNTICA"
    )
    print("=" * 70)

    print(
        f"\nModelo: "
        f"{EMBEDDING_MODEL}"
    )

    print(
        f"Queries: "
        f"{len(QUERIES)}"
    )

    print(
        f"Estratégias: "
        f"{TOTAL_TESTES}"
    )

    # ========================================================
    # EMBEDDINGS DAS QUERIES
    # ========================================================

    embeddings_queries = (
        gerar_embeddings_queries()
    )

    estatisticas_chunking = (
        carregar_estatisticas_chunking()
    )

    resultados_testes = []

    # ========================================================
    # TESTES 1 A 10
    # ========================================================

    for test_id in range(
        1,
        TOTAL_TESTES + 1
    ):

        print("\n")
        print("#" * 70)

        print(
            f"TESTE {test_id:02d} "
            f"- {ESTRATEGIAS[test_id]}"
        )

        print("#" * 70)

        # ----------------------------------------------------
        # CORPUS
        # ----------------------------------------------------

        corpus = carregar_corpus_teste(
            test_id
        )

        print(
            f"Chunks no corpus: "
            f"{len(corpus)}"
        )

        matriz = (
            preparar_matriz_embeddings(
                corpus
            )
        )

        resultados_queries = []

        # ----------------------------------------------------
        # QUERIES
        # ----------------------------------------------------

        for query in QUERIES:

            resultado = avaliar_query(
                query,
                embeddings_queries[
                    query["query_id"]
                ],
                corpus,
                matriz
            )

            resultados_queries.append(
                resultado
            )

            status = (
                "OK"
                if resultado[
                    "top1_correct"
                ]
                else "ERRO"
            )

            print(
                f"\n{query['query_id']} "
                f"| {status}"
            )

            print(
                f"Esperado: "
                f"{query['expected_document']}"
            )

            print(
                f"TOP 1: "
                f"{resultado['top1_document']}"
            )

            print(
                f"Score: "
                f"{resultado['top1_similarity']}"
            )

            print(
                f"Hit@3: "
                f"{resultado['hit_at_3']}"
            )

            print(
                f"Primeiro relevante: "
                f"{resultado['first_relevant_rank']}"
            )

        # ====================================================
        # MÉTRICAS AGREGADAS
        # ====================================================

        top1_accuracy = mean(
            1 if item[
                "top1_correct"
            ]
            else 0

            for item
            in resultados_queries
        )

        hit_at_3 = mean(
            1 if item[
                "hit_at_3"
            ]
            else 0

            for item
            in resultados_queries
        )

        mrr = mean(
            item[
                "reciprocal_rank"
            ]

            for item
            in resultados_queries
        )

        avg_top1_similarity = mean(
            item[
                "top1_similarity"
            ]

            for item
            in resultados_queries
        )

        avg_best_expected_similarity = mean(
            item[
                "best_expected_similarity"
            ]

            for item
            in resultados_queries
        )

        resultado_teste = {
            "test_id":
                test_id,

            "strategy":
                ESTRATEGIAS[
                    test_id
                ],

            "num_queries":
                len(QUERIES),

            "top1_accuracy":
                round(
                    top1_accuracy,
                    4
                ),

            "hit_at_3":
                round(
                    hit_at_3,
                    4
                ),

            "mrr":
                round(
                    mrr,
                    4
                ),

            "avg_top1_similarity":
                round(
                    avg_top1_similarity,
                    4
                ),

            "avg_best_expected_similarity":
                round(
                    avg_best_expected_similarity,
                    4
                ),

            "chunking_statistics":
                estatisticas_chunking.get(
                    test_id,
                    {}
                ),

            "queries":
                resultados_queries,
        }

        resultados_testes.append(
            resultado_teste
        )

    # ========================================================
    # RANKING PRELIMINAR
    # ========================================================
    #
    # Ordenação:
    #
    # 1 - precisão TOP 1
    # 2 - Hit@3
    # 3 - MRR
    # 4 - similaridade do melhor trecho esperado
    #
    # NÃO representa ainda a decisão final de melhor
    # estratégia para RAG.
    # ========================================================

    ranking = sorted(
        resultados_testes,
        key=lambda item: (
            item[
                "top1_accuracy"
            ],
            item[
                "hit_at_3"
            ],
            item[
                "mrr"
            ],
            item[
                "avg_best_expected_similarity"
            ],
        ),
        reverse=True
    )

    for posicao, item in enumerate(
        ranking,
        start=1
    ):

        item[
            "retrieval_rank"
        ] = posicao

    # ========================================================
    # SALVAR
    # ========================================================

    resultado_final = {
        "embedding_model":
            EMBEDDING_MODEL,

        "documents":
            DOCUMENTOS,

        "num_queries":
            len(QUERIES),

        "queries":
            QUERIES,

        "results":
            resultados_testes,

        "retrieval_ranking": [
            {
                "position":
                    posicao,

                "test_id":
                    item[
                        "test_id"
                    ],

                "strategy":
                    item[
                        "strategy"
                    ],

                "top1_accuracy":
                    item[
                        "top1_accuracy"
                    ],

                "hit_at_3":
                    item[
                        "hit_at_3"
                    ],

                "mrr":
                    item[
                        "mrr"
                    ],

                "avg_top1_similarity":
                    item[
                        "avg_top1_similarity"
                    ],

                "avg_best_expected_similarity":
                    item[
                        "avg_best_expected_similarity"
                    ],
            }

            for posicao, item
            in enumerate(
                ranking,
                start=1
            )
        ]
    }

    caminho_saida = (
        PASTA_RESULTS
        / "summary_retrieval.json"
    )

    with caminho_saida.open(
        "w",
        encoding="utf-8"
    ) as arquivo:

        json.dump(
            resultado_final,
            arquivo,
            ensure_ascii=False,
            indent=2
        )

    # ========================================================
    # TABELA FINAL
    # ========================================================

    print("\n")
    print("=" * 100)
    print(
        "COMPARAÇÃO DAS 10 ESTRATÉGIAS"
    )
    print("=" * 100)

    print(
        f"{'POS':<6}"
        f"{'TESTE':<8}"
        f"{'ESTRATÉGIA':<24}"
        f"{'TOP1':<10}"
        f"{'HIT@3':<10}"
        f"{'MRR':<10}"
        f"{'SIM':<10}"
        f"{'CHUNKS':<10}"
        f"{'TAM.MÉDIO':<12}"
    )

    for posicao, item in enumerate(
        ranking,
        start=1
    ):

        estatisticas = (
            item[
                "chunking_statistics"
            ]
        )

        print(
            f"{posicao:<6}"
            f"{item['test_id']:<8}"
            f"{item['strategy']:<24}"
            f"{item['top1_accuracy']:<10}"
            f"{item['hit_at_3']:<10}"
            f"{item['mrr']:<10}"
            f"{item['avg_best_expected_similarity']:<10}"
            f"{estatisticas.get('avg_num_chunks', '-'):<10}"
            f"{estatisticas.get('avg_chunk_size', '-'):<12}"
        )

    print("\n")
    print(
        f"Resultado salvo em:\n"
        f"{caminho_saida}"
    )

    print("\n")
    print("=" * 70)
    print(
        "ETAPA 4 FINALIZADA COM SUCESSO"
    )
    print("=" * 70)


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    executar_avaliacao()