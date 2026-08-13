import json
import os
import time
from pathlib import Path
from statistics import mean

import nltk
import tiktoken

from dotenv import load_dotenv
from openai import OpenAI

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    NLTKTextSplitter,
)


# ============================================================
# CAMINHOS
# ============================================================

PASTA_ATUAL = Path(__file__).resolve().parent
ROOT = PASTA_ATUAL.parent

PASTA_RESULTS = PASTA_ATUAL / "results"


# ============================================================
# CONFIGURAÇÃO OPENROUTER
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


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENAI_API_KEY
)


# ============================================================
# TOKENIZER
# ============================================================

TOKENIZER = tiktoken.get_encoding(
    "cl100k_base"
)


# ============================================================
# 12 DOCUMENTOS
# ============================================================

DOCUMENTOS = [
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


# ============================================================
# ESTRATÉGIAS FINALISTAS
# ============================================================

ESTRATEGIAS = [
    {
        "test_id": 3,
        "strategy": "fixed_1000",
        "description": "1000 caracteres sem overlap",
    },
    {
        "test_id": 8,
        "strategy": "three_sentences",
        "description": "3 sentenças por chunk",
    },
    {
        "test_id": 9,
        "strategy": "recursive",
        "description": "Recursive 1000 caracteres com overlap 100",
    },
]


# ============================================================
# CONFIGURAÇÃO EMBEDDINGS
# ============================================================

BATCH_SIZE = 64
PAUSA_BATCH = 0.2

# Margem abaixo do limite de 8192 tokens por entrada.
MAX_TOKENS_CHUNK = 8000

# Limite seguro para a soma de tokens em uma requisição.
MAX_TOKENS_BATCH = 250000


# ============================================================
# NLTK
# ============================================================

def preparar_nltk():

    recursos = [
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
    ]

    for pacote, caminho in recursos:

        try:
            nltk.data.find(caminho)

        except LookupError:
            nltk.download(
                pacote,
                quiet=False
            )


# ============================================================
# TOKENS
# ============================================================

def contar_tokens(texto):
    """
    Conta tokens tratando sequências como <|endofprompt|>
    como texto normal.
    """

    return len(
        TOKENIZER.encode(
            texto,
            disallowed_special=()
        )
    )


# ============================================================
# FALLBACK PARA CHUNKS GRANDES
# ============================================================

def dividir_texto_por_limite_tokens(
    texto,
    max_tokens=MAX_TOKENS_CHUNK
):
    """
    Divide recursivamente textos que ultrapassam o limite
    máximo de tokens aceito pelo modelo.
    """

    quantidade_tokens = contar_tokens(
        texto
    )

    if quantidade_tokens <= max_tokens:
        return [texto]

    meio = len(texto) // 2

    # Primeiro tenta dividir por parágrafo.
    corte = texto.rfind(
        "\n\n",
        0,
        meio
    )

    # Depois tenta por linha.
    if corte < len(texto) // 4:

        corte = texto.rfind(
            "\n",
            0,
            meio
        )

    # Se não encontrar um bom separador,
    # corta aproximadamente pela metade.
    if corte < len(texto) // 4:

        corte = meio

    esquerda = texto[
        :corte
    ].strip()

    direita = texto[
        corte:
    ].strip()

    if not esquerda or not direita:

        corte = meio

        esquerda = texto[
            :corte
        ].strip()

        direita = texto[
            corte:
        ].strip()

    if not esquerda or not direita:

        raise ValueError(
            "Não foi possível subdividir um chunk "
            "acima do limite de tokens."
        )

    partes = []

    partes.extend(
        dividir_texto_por_limite_tokens(
            esquerda,
            max_tokens
        )
    )

    partes.extend(
        dividir_texto_por_limite_tokens(
            direita,
            max_tokens
        )
    )

    return partes


def normalizar_chunks_limite_tokens(
    chunks
):
    """
    Mantém chunks normais e subdivide somente os que
    ultrapassam MAX_TOKENS_CHUNK.
    """

    chunks_normalizados = []

    for chunk in chunks:

        texto = chunk[
            "text"
        ]

        quantidade_tokens = contar_tokens(
            texto
        )

        if (
            quantidade_tokens
            <= MAX_TOKENS_CHUNK
        ):

            chunks_normalizados.append(
                chunk
            )

            continue

        print(
            f"   AVISO: chunk com "
            f"{quantidade_tokens} tokens."
        )

        print(
            "   Aplicando divisão de segurança..."
        )

        partes = (
            dividir_texto_por_limite_tokens(
                texto
            )
        )

        total_partes = len(
            partes
        )

        for numero_parte, parte in enumerate(
            partes,
            start=1
        ):

            metadata = dict(
                chunk.get(
                    "metadata",
                    {}
                )
            )

            metadata[
                "fallback_token_split"
            ] = True

            metadata[
                "original_tokens"
            ] = quantidade_tokens

            metadata[
                "fallback_part"
            ] = numero_parte

            metadata[
                "fallback_total_parts"
            ] = total_partes

            chunks_normalizados.append({
                "text":
                    parte,

                "metadata":
                    metadata
            })

    return chunks_normalizados


# ============================================================
# CARREGAR MARKDOWN
# ============================================================

def carregar_markdown(
    document_id
):

    caminho = (
        PASTA_RESULTS
        / document_id
        / "markdown"
        / f"{document_id}.md"
    )

    if not caminho.exists():

        raise FileNotFoundError(
            f"Markdown não encontrado:\n"
            f"{caminho}"
        )

    return caminho.read_text(
        encoding="utf-8"
    )


# ============================================================
# TESTE 3
# FIXO 1000
# ============================================================

def chunk_fixed_1000(
    texto
):

    splitter = CharacterTextSplitter(
        separator="",
        chunk_size=1000,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False,
    )

    textos = splitter.split_text(
        texto
    )

    return [
        {
            "text":
                trecho,

            "metadata":
                {}
        }

        for trecho in textos

        if trecho.strip()
    ]


# ============================================================
# CONTADOR DE SENTENÇAS
# ============================================================

def contar_sentencas(
    texto
):

    if not texto.strip():

        return 0

    return len(
        nltk.sent_tokenize(
            texto,
            language="portuguese"
        )
    )


# ============================================================
# TESTE 8
# 3 SENTENÇAS
# ============================================================

def chunk_three_sentences(
    texto
):

    splitter = NLTKTextSplitter(
        language="portuguese",
        separator=" ",
        chunk_size=3,
        chunk_overlap=0,
        length_function=contar_sentencas,
    )

    textos = splitter.split_text(
        texto
    )

    return [
        {
            "text":
                trecho,

            "metadata": {
                "unit":
                    "3_sentences"
            }
        }

        for trecho in textos

        if trecho.strip()
    ]


# ============================================================
# TESTE 9
# RECURSIVE
# ============================================================

def chunk_recursive(
    texto
):

    splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=[
                "\n\n",
                "\n",
                " ",
                ""
            ],
            length_function=len,
            is_separator_regex=False,
        )
    )

    textos = splitter.split_text(
        texto
    )

    return [
        {
            "text":
                trecho,

            "metadata": {
                "separators": [
                    "\\n\\n",
                    "\\n",
                    "space",
                    "character"
                ]
            }
        }

        for trecho in textos

        if trecho.strip()
    ]


# ============================================================
# EXECUTAR CHUNKING
# ============================================================

def executar_chunking(
    texto,
    test_id
):

    if test_id == 3:

        return chunk_fixed_1000(
            texto
        )

    if test_id == 8:

        return chunk_three_sentences(
            texto
        )

    if test_id == 9:

        return chunk_recursive(
            texto
        )

    raise ValueError(
        f"Estratégia inválida: "
        f"{test_id}"
    )


# ============================================================
# EMBEDDINGS
# ============================================================

def gerar_embeddings(
    textos
):
    """
    Gera embeddings em lotes respeitando:

    - no máximo BATCH_SIZE textos;
    - no máximo MAX_TOKENS_BATCH tokens por requisição.
    """

    embeddings = []

    total = len(
        textos
    )

    inicio = 0

    while inicio < total:

        lote = []

        tokens_lote = 0

        indice = inicio

        while (
            indice < total
            and len(lote) < BATCH_SIZE
        ):

            texto = textos[
                indice
            ]

            tokens_texto = contar_tokens(
                texto
            )

            # Segurança extra
            if (
                tokens_texto
                > MAX_TOKENS_CHUNK
            ):

                raise ValueError(
                    f"Chunk {indice + 1} "
                    f"ainda possui "
                    f"{tokens_texto} tokens "
                    f"após a validação."
                )

            # Se adicionar este texto ultrapassar
            # o limite total do batch, encerramos
            # o lote atual.
            if (
                lote
                and (
                    tokens_lote
                    + tokens_texto
                    > MAX_TOKENS_BATCH
                )
            ):

                break

            lote.append(
                texto
            )

            tokens_lote += (
                tokens_texto
            )

            indice += 1

        if not lote:

            raise RuntimeError(
                "Não foi possível montar "
                "um lote de embeddings."
            )

        fim = (
            inicio
            + len(lote)
        )

        print(
            f"      Embeddings "
            f"{inicio + 1}-{fim}/{total} "
            f"(~{tokens_lote} tokens)"
        )

        resposta = (
            client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=lote
            )
        )

        embeddings.extend(
            [
                item.embedding

                for item
                in resposta.data
            ]
        )

        inicio = fim

        time.sleep(
            PAUSA_BATCH
        )

    return embeddings


# ============================================================
# ESTATÍSTICAS
# ============================================================

def calcular_estatisticas(
    chunks
):

    if not chunks:

        return {
            "num_chunks": 0,
            "avg_chunk_size": 0,
            "min_chunk_size": 0,
            "max_chunk_size": 0,
            "avg_tokens": 0,
            "min_tokens": 0,
            "max_tokens": 0,
            "total_tokens": 0,
        }

    tamanhos = [
        len(
            chunk["text"]
        )

        for chunk
        in chunks
    ]

    tokens = [
        contar_tokens(
            chunk["text"]
        )

        for chunk
        in chunks
    ]

    return {
        "num_chunks":
            len(chunks),

        "avg_chunk_size":
            round(
                mean(
                    tamanhos
                ),
                2
            ),

        "min_chunk_size":
            min(
                tamanhos
            ),

        "max_chunk_size":
            max(
                tamanhos
            ),

        "avg_tokens":
            round(
                mean(
                    tokens
                ),
                2
            ),

        "min_tokens":
            min(
                tokens
            ),

        "max_tokens":
            max(
                tokens
            ),

        "total_tokens":
            sum(
                tokens
            ),
    }


# ============================================================
# PROCESSAR DOCUMENTO
# ============================================================

def processar(
    document_id,
    estrategia
):

    test_id = estrategia[
        "test_id"
    ]

    print(
        "\n"
        + "-"
        * 70
    )

    print(
        f"Documento: "
        f"{document_id}"
    )

    print(
        f"Estratégia: "
        f"{estrategia['strategy']}"
    )

    pasta = (
        PASTA_RESULTS
        / document_id
        / f"test_{test_id:02d}"
    )

    pasta.mkdir(
        parents=True,
        exist_ok=True
    )

    caminho_saida = (
        pasta
        / "chunks_embeddings.json"
    )

    # ========================================================
    # RESULTADO JÁ EXISTENTE
    # ========================================================

    if (
        caminho_saida.exists()
        and caminho_saida.stat().st_size > 0
    ):

        print(
            "   Resultado já existe."
        )

        print(
            "   Estratégia ignorada."
        )

        with caminho_saida.open(
            "r",
            encoding="utf-8"
        ) as arquivo:

            dados = json.load(
                arquivo
            )

        estatisticas = (
            calcular_estatisticas(
                dados
            )
        )

        embedding_dimension = None

        if dados:

            embedding = (
                dados[0].get(
                    "embedding"
                )
            )

            if embedding:

                embedding_dimension = (
                    len(
                        embedding
                    )
                )

        return {
            "document_id":
                document_id,

            "test_id":
                test_id,

            "strategy":
                estrategia[
                    "strategy"
                ],

            **estatisticas,

            "embedding_dimension":
                embedding_dimension,

            "status":
                "existing"
        }

    # ========================================================
    # CARREGAR MARKDOWN
    # ========================================================

    texto = carregar_markdown(
        document_id
    )

    # ========================================================
    # CHUNKING
    # ========================================================

    chunks = executar_chunking(
        texto,
        test_id
    )

    print(
        f"   Chunks antes da validação: "
        f"{len(chunks)}"
    )

    # ========================================================
    # CORRIGIR CHUNKS ACIMA DO LIMITE
    # ========================================================

    chunks = (
        normalizar_chunks_limite_tokens(
            chunks
        )
    )

    print(
        f"   Chunks após validação: "
        f"{len(chunks)}"
    )

    # ========================================================
    # ESTATÍSTICAS
    # ========================================================

    estatisticas = (
        calcular_estatisticas(
            chunks
        )
    )

    # ========================================================
    # TEXTOS DOS CHUNKS
    # ========================================================

    textos = [
        chunk[
            "text"
        ]

        for chunk
        in chunks
    ]

    # ========================================================
    # EMBEDDINGS
    # ========================================================

    embeddings = gerar_embeddings(
        textos
    )

    if (
        len(embeddings)
        != len(chunks)
    ):

        raise RuntimeError(
            "Quantidade de embeddings "
            "diferente da quantidade "
            "de chunks."
        )

    # ========================================================
    # JSON FINAL
    # ========================================================

    resultado_json = []

    for indice, (
        chunk,
        embedding
    ) in enumerate(
        zip(
            chunks,
            embeddings
        ),
        start=1
    ):

        chunk_id = (
            f"{document_id}_"
            f"test{test_id:02d}_"
            f"chunk{indice:04d}"
        )

        resultado_json.append({
            "chunk_id":
                chunk_id,

            "document_id":
                document_id,

            "document_name":
                f"{document_id}.pdf",

            "test_id":
                test_id,

            "strategy":
                estrategia[
                    "strategy"
                ],

            "text":
                chunk[
                    "text"
                ],

            "embedding":
                embedding,

            "metadata":
                chunk[
                    "metadata"
                ],
        })

    # ========================================================
    # SALVAR
    # ========================================================

    with caminho_saida.open(
        "w",
        encoding="utf-8"
    ) as arquivo:

        json.dump(
            resultado_json,
            arquivo,
            ensure_ascii=False,
            indent=2
        )

    embedding_dimension = (
        len(
            embeddings[0]
        )
        if embeddings
        else None
    )

    print(
        f"   Dimensão embedding: "
        f"{embedding_dimension}"
    )

    print(
        f"   OK -> "
        f"{caminho_saida}"
    )

    return {
        "document_id":
            document_id,

        "test_id":
            test_id,

        "strategy":
            estrategia[
                "strategy"
            ],

        **estatisticas,

        "embedding_dimension":
            embedding_dimension,

        "status":
            "generated"
    }


# ============================================================
# MAIN
# ============================================================

def executar():

    print("\n")
    print("=" * 70)
    print(
        "ATIVIDADE 04 - ETAPA 5"
    )
    print(
        "APLICAÇÃO DAS ESTRATÉGIAS FINALISTAS"
    )
    print("=" * 70)

    print(
        f"\nDocumentos: "
        f"{len(DOCUMENTOS)}"
    )

    print(
        f"Estratégias: "
        f"{len(ESTRATEGIAS)}"
    )

    print(
        f"Total de combinações: "
        f"{len(DOCUMENTOS) * len(ESTRATEGIAS)}"
    )

    preparar_nltk()

    resultados = []

    erros = []

    contador = 0

    total = (
        len(DOCUMENTOS)
        * len(ESTRATEGIAS)
    )

    # ========================================================
    # DOCUMENTOS
    # ========================================================

    for document_id in DOCUMENTOS:

        print("\n")
        print("#" * 70)

        print(
            f"DOCUMENTO: "
            f"{document_id}"
        )

        print("#" * 70)

        # ====================================================
        # ESTRATÉGIAS
        # ====================================================

        for estrategia in ESTRATEGIAS:

            contador += 1

            print(
                f"\nExperimento "
                f"{contador}/{total}"
            )

            try:

                resultado = processar(
                    document_id,
                    estrategia
                )

                resultados.append(
                    resultado
                )

            except Exception as erro:

                print(
                    f"ERRO: "
                    f"{erro}"
                )

                erros.append({
                    "document_id":
                        document_id,

                    "test_id":
                        estrategia[
                            "test_id"
                        ],

                    "strategy":
                        estrategia[
                            "strategy"
                        ],

                    "error":
                        str(
                            erro
                        )
                })

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {
        "embedding_model":
            EMBEDDING_MODEL,

        "selected_strategies":
            ESTRATEGIAS,

        "num_documents":
            len(
                DOCUMENTOS
            ),

        "documents":
            DOCUMENTOS,

        "expected_experiments":
            total,

        "successful_experiments":
            len(
                resultados
            ),

        "num_errors":
            len(
                erros
            ),

        "results":
            resultados,

        "errors":
            erros,
    }

    caminho_summary = (
        PASTA_RESULTS
        / "summary_finalistas.json"
    )

    with caminho_summary.open(
        "w",
        encoding="utf-8"
    ) as arquivo:

        json.dump(
            summary,
            arquivo,
            ensure_ascii=False,
            indent=2
        )

    # ========================================================
    # RESUMO FINAL
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "RESUMO FINAL"
    )
    print("=" * 70)

    print(
        f"\nExperimentos esperados: "
        f"{total}"
    )

    print(
        f"Experimentos disponíveis: "
        f"{len(resultados)}"
    )

    print(
        f"Erros: "
        f"{len(erros)}"
    )

    print(
        "\nEstratégias utilizadas:"
    )

    for estrategia in ESTRATEGIAS:

        print(
            f"- Teste "
            f"{estrategia['test_id']:02d}: "
            f"{estrategia['strategy']}"
        )

    if erros:

        print(
            "\nERROS:"
        )

        for erro in erros:

            print(
                f"- "
                f"{erro['document_id']} "
                f"| "
                f"{erro['strategy']}: "
                f"{erro['error']}"
            )

    print(
        f"\nSummary:\n"
        f"{caminho_summary}"
    )

    print("\n")
    print("=" * 70)

    if erros:

        print(
            "ETAPA 5 FINALIZADA COM ERROS"
        )

    else:

        print(
            "ETAPA 5 FINALIZADA COM SUCESSO"
        )

    print("=" * 70)


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    executar()