import json
from pathlib import Path
from statistics import mean

import nltk
import tiktoken

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    NLTKTextSplitter,
)


# ============================================================
# CONFIGURAÇÃO
# ============================================================

PASTA_ATUAL = Path(__file__).resolve().parent
PASTA_RESULTS = PASTA_ATUAL / "results"

# Nesta etapa inicial utilizaremos somente os três documentos
# das aulas anteriores.
DOCUMENTOS_TESTE = [
    "bioetica_e_ia",
    "escrita_academica_ia",
    "twitter_algoritmo",
]

# Tokenizador usado apenas para estimar a quantidade de tokens.
TOKENIZER = tiktoken.get_encoding("cl100k_base")


# ============================================================
# NLTK
# ============================================================

def garantir_recursos_nltk():
    """
    Verifica se os recursos necessários para segmentação
    de sentenças estão disponíveis.
    """

    recursos = [
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
    ]

    for pacote, caminho in recursos:

        try:
            nltk.data.find(caminho)

        except LookupError:

            print(
                f"Baixando recurso NLTK: {pacote}"
            )

            nltk.download(
                pacote,
                quiet=False
            )


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def contar_tokens(texto: str) -> int:
    """
    Retorna uma estimativa da quantidade de tokens.
    """

    return len(
        TOKENIZER.encode(texto)
    )


def carregar_markdown(document_id: str) -> str:
    """
    Carrega o Markdown previamente gerado pelo Docling.
    """

    caminho = (
        PASTA_RESULTS
        / document_id
        / "markdown"
        / f"{document_id}.md"
    )

    if not caminho.exists():

        raise FileNotFoundError(
            f"Markdown não encontrado:\n{caminho}"
        )

    return caminho.read_text(
        encoding="utf-8"
    )


def converter_chunks_simples(
    textos,
    metadata_extra=None
):
    """
    Converte uma lista de strings para uma estrutura
    padronizada de chunks.
    """

    metadata_extra = metadata_extra or {}

    return [
        {
            "text": texto,
            "metadata": dict(metadata_extra)
        }
        for texto in textos
        if texto.strip()
    ]


# ============================================================
# TESTES 1 A 6
# CHUNKING FIXO POR CARACTERES
# ============================================================

def chunking_fixo(
    texto,
    chunk_size,
    chunk_overlap
):
    """
    Divide o texto estritamente por quantidade de caracteres.

    separator="" faz com que o CharacterTextSplitter
    opere no nível de caracteres.
    """

    splitter = CharacterTextSplitter(
        separator="",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = splitter.split_text(
        texto
    )

    return converter_chunks_simples(
        chunks
    )


# ============================================================
# TESTE 7
# PARÁGRAFOS
# ============================================================

def contar_unidades_paragrafo(texto: str) -> int:
    """
    Função de tamanho utilizada para fazer cada
    parágrafo representar uma unidade.
    """

    if not texto.strip():
        return 0

    return 1


def chunking_paragrafos(texto):
    """
    Preserva cada parágrafo como um chunk independente.
    """

    splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1,
        chunk_overlap=0,
        length_function=contar_unidades_paragrafo,
        keep_separator=False,
    )

    chunks = splitter.split_text(
        texto
    )

    return converter_chunks_simples(
        chunks,
        {
            "unit": "paragraph"
        }
    )


# ============================================================
# TESTE 8
# 3 SENTENÇAS POR CHUNK
# ============================================================

def contar_sentencas(texto: str) -> int:
    """
    Conta sentenças usando o tokenizer do NLTK.

    Essa função é utilizada como length_function
    no NLTKTextSplitter.
    """

    if not texto.strip():
        return 0

    sentencas = nltk.sent_tokenize(
        texto,
        language="portuguese"
    )

    return len(sentencas)


def chunking_tres_sentencas(texto):
    """
    Utiliza NLTKTextSplitter e define o tamanho
    do chunk em número de sentenças.

    chunk_size = 3 significa:
    três sentenças por chunk.
    """

    splitter = NLTKTextSplitter(
        language="portuguese",
        separator=" ",
        chunk_size=3,
        chunk_overlap=0,
        length_function=contar_sentencas,
    )

    chunks = splitter.split_text(
        texto
    )

    return converter_chunks_simples(
        chunks,
        {
            "unit": "3_sentences"
        }
    )


# ============================================================
# TESTE 9
# RECURSIVE CHARACTER TEXT SPLITTER
# ============================================================

def chunking_recursivo(texto):
    """
    Estratégia hierárquica:

    1. Parágrafos
    2. Linhas
    3. Espaços
    4. Caracteres

    Configuração escolhida:
    chunk_size = 1000
    chunk_overlap = 100
    """

    splitter = RecursiveCharacterTextSplitter(
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

    chunks = splitter.split_text(
        texto
    )

    return converter_chunks_simples(
        chunks,
        {
            "separators": [
                "\\n\\n",
                "\\n",
                "space",
                "character"
            ]
        }
    )


# ============================================================
# TESTE 10
# MARKDOWN HEADER TEXT SPLITTER
# ============================================================

def chunking_markdown(texto):
    """
    Divide o Markdown utilizando sua estrutura
    hierárquica de headings.
    """

    headers_to_split_on = [
        ("#", "heading_1"),
        ("##", "heading_2"),
        ("###", "heading_3"),
        ("####", "heading_4"),
        ("#####", "heading_5"),
        ("######", "heading_6"),
    ]

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )

    documentos = splitter.split_text(
        texto
    )

    chunks = []

    for documento in documentos:

        metadata = dict(
            documento.metadata
        )

        # ----------------------------------------------------
        # IDENTIFICAR O HEADING MAIS ESPECÍFICO
        # ----------------------------------------------------

        heading_atual = None
        heading_level = None

        for nivel in range(
            6,
            0,
            -1
        ):

            chave = f"heading_{nivel}"

            if chave in metadata:

                heading_atual = metadata[
                    chave
                ]

                heading_level = nivel

                break

        # ----------------------------------------------------
        # METADADOS ADICIONAIS
        # ----------------------------------------------------

        metadata["heading"] = (
            heading_atual
        )

        metadata["heading_level"] = (
            heading_level
        )

        metadata["section"] = (
            metadata.get("heading_1")
            or metadata.get("heading_2")
        )

        metadata["subsection"] = (
            metadata.get("heading_2")
            or metadata.get("heading_3")
        )

        chunks.append({
            "text": documento.page_content,
            "metadata": metadata
        })

    return chunks


# ============================================================
# CONFIGURAÇÃO DOS 10 EXPERIMENTOS
# ============================================================

EXPERIMENTOS = [
    {
        "test_id": 1,
        "strategy": "fixed_200",
        "description": "200 caracteres sem overlap",
        "chunk_size": 200,
        "chunk_overlap": 0,
    },
    {
        "test_id": 2,
        "strategy": "fixed_500",
        "description": "500 caracteres sem overlap",
        "chunk_size": 500,
        "chunk_overlap": 0,
    },
    {
        "test_id": 3,
        "strategy": "fixed_1000",
        "description": "1000 caracteres sem overlap",
        "chunk_size": 1000,
        "chunk_overlap": 0,
    },
    {
        "test_id": 4,
        "strategy": "fixed_2000",
        "description": "2000 caracteres sem overlap",
        "chunk_size": 2000,
        "chunk_overlap": 0,
    },
    {
        "test_id": 5,
        "strategy": "fixed_overlap_50",
        "description": "500 caracteres com overlap 50",
        "chunk_size": 500,
        "chunk_overlap": 50,
    },
    {
        "test_id": 6,
        "strategy": "fixed_overlap_200",
        "description": "500 caracteres com overlap 200",
        "chunk_size": 500,
        "chunk_overlap": 200,
    },
    {
        "test_id": 7,
        "strategy": "paragraph",
        "description": "Divisão por parágrafos",
        "chunk_size": None,
        "chunk_overlap": 0,
    },
    {
        "test_id": 8,
        "strategy": "three_sentences",
        "description": "3 sentenças por chunk",
        "chunk_size": 3,
        "chunk_overlap": 0,
    },
    {
        "test_id": 9,
        "strategy": "recursive",
        "description": (
            "Recursive Character Text Splitter "
            "1000/100"
        ),
        "chunk_size": 1000,
        "chunk_overlap": 100,
    },
    {
        "test_id": 10,
        "strategy": "markdown_headers",
        "description": (
            "Divisão por headings Markdown"
        ),
        "chunk_size": None,
        "chunk_overlap": 0,
    },
]


# ============================================================
# EXECUTAR UM EXPERIMENTO
# ============================================================

def executar_chunking(
    texto,
    experimento
):

    test_id = experimento[
        "test_id"
    ]

    if test_id in {
        1, 2, 3, 4, 5, 6
    }:

        return chunking_fixo(
            texto,
            experimento["chunk_size"],
            experimento["chunk_overlap"],
        )

    if test_id == 7:

        return chunking_paragrafos(
            texto
        )

    if test_id == 8:

        return chunking_tres_sentencas(
            texto
        )

    if test_id == 9:

        return chunking_recursivo(
            texto
        )

    if test_id == 10:

        return chunking_markdown(
            texto
        )

    raise ValueError(
        f"Teste inválido: {test_id}"
    )


# ============================================================
# ESTATÍSTICAS
# ============================================================

def calcular_estatisticas(
    chunks,
    experimento
):
    """
    Calcula estatísticas solicitadas pela atividade.
    """

    tamanhos = [
        len(chunk["text"])
        for chunk in chunks
    ]

    tokens = [
        contar_tokens(
            chunk["text"]
        )
        for chunk in chunks
    ]

    if not tamanhos:

        return {
            "num_chunks": 0
        }

    chunk_size = experimento[
        "chunk_size"
    ]

    chunk_overlap = experimento[
        "chunk_overlap"
    ]

    # --------------------------------------------------------
    # OVERLAP
    # --------------------------------------------------------

    if (
        isinstance(chunk_size, int)
        and chunk_size > 0
        and chunk_overlap > 0
    ):

        percentual_overlap = (
            chunk_overlap
            / chunk_size
            * 100
        )

        chunks_sobrepostos = max(
            len(chunks) - 1,
            0
        )

    else:

        percentual_overlap = 0.0
        chunks_sobrepostos = 0

    return {
        "num_chunks":
            len(chunks),

        "avg_chunk_size":
            round(
                mean(tamanhos),
                2
            ),

        "min_chunk_size":
            min(tamanhos),

        "max_chunk_size":
            max(tamanhos),

        "avg_tokens":
            round(
                mean(tokens),
                2
            ),

        "min_tokens":
            min(tokens),

        "max_tokens":
            max(tokens),

        "total_tokens":
            sum(tokens),

        "overlapping_chunks":
            chunks_sobrepostos,

        "overlap_percent":
            round(
                percentual_overlap,
                2
            ),
    }


# ============================================================
# PREPARAR JSON DOS CHUNKS
# ============================================================

def preparar_chunks_json(
    document_id,
    experimento,
    chunks
):

    resultado = []

    for indice, chunk in enumerate(
        chunks,
        start=1
    ):

        chunk_id = (
            f"{document_id}_"
            f"test{experimento['test_id']:02d}_"
            f"chunk{indice:04d}"
        )

        resultado.append({
            "chunk_id":
                chunk_id,

            "document_id":
                document_id,

            "document_name":
                f"{document_id}.pdf",

            "test_id":
                experimento["test_id"],

            "strategy":
                experimento["strategy"],

            "chunk_size":
                experimento["chunk_size"],

            "chunk_overlap":
                experimento["chunk_overlap"],

            "text":
                chunk["text"],

            # Embedding será preenchido
            # na próxima etapa.
            "embedding":
                None,

            "metadata":
                chunk["metadata"],
        })

    return resultado


# ============================================================
# SALVAR RESULTADOS
# ============================================================

def salvar_resultado_teste(
    document_id,
    experimento,
    chunks_json
):

    test_id = experimento[
        "test_id"
    ]

    pasta_teste = (
        PASTA_RESULTS
        / document_id
        / f"test_{test_id:02d}"
    )

    pasta_teste.mkdir(
        parents=True,
        exist_ok=True
    )

    caminho = (
        pasta_teste
        / "chunks_preliminares.json"
    )

    caminho.write_text(
        json.dumps(
            chunks_json,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    return caminho


# ============================================================
# EXEMPLOS
# ============================================================

def gerar_exemplos(chunks):
    """
    Guarda até três exemplos para o resumo.
    """

    exemplos = []

    for chunk in chunks[:3]:

        texto = chunk[
            "text"
        ]

        if len(texto) > 300:

            texto = (
                texto[:300]
                + "..."
            )

        exemplos.append(
            texto
        )

    return exemplos


# ============================================================
# EXECUTAR OS 10 EXPERIMENTOS
# ============================================================

def executar_experimentos():

    print("\n")
    print("=" * 70)
    print(
        "ATIVIDADE 04 - ETAPA 2"
    )
    print(
        "10 ESTRATÉGIAS DE CHUNKING"
    )
    print("=" * 70)

    garantir_recursos_nltk()

    resumo_geral = []

    # ========================================================
    # DOCUMENTOS
    # ========================================================

    for numero_documento, document_id in enumerate(
        DOCUMENTOS_TESTE,
        start=1
    ):

        print("\n")
        print("#" * 70)

        print(
            f"DOCUMENTO "
            f"{numero_documento}/"
            f"{len(DOCUMENTOS_TESTE)}"
        )

        print(
            document_id
        )

        print("#" * 70)

        texto = carregar_markdown(
            document_id
        )

        print(
            f"\nTamanho original: "
            f"{len(texto):,} caracteres"
        )

        # ====================================================
        # TESTES
        # ====================================================

        for experimento in EXPERIMENTOS:

            print("\n" + "-" * 70)

            print(
                f"TESTE "
                f"{experimento['test_id']:02d}"
            )

            print(
                experimento[
                    "description"
                ]
            )

            # -----------------------------------------------
            # CHUNKING
            # -----------------------------------------------

            chunks = executar_chunking(
                texto,
                experimento
            )

            # -----------------------------------------------
            # ESTATÍSTICAS
            # -----------------------------------------------

            estatisticas = (
                calcular_estatisticas(
                    chunks,
                    experimento
                )
            )

            # -----------------------------------------------
            # PREPARAR JSON
            # -----------------------------------------------

            chunks_json = (
                preparar_chunks_json(
                    document_id,
                    experimento,
                    chunks
                )
            )

            # -----------------------------------------------
            # SALVAR
            # -----------------------------------------------

            caminho = (
                salvar_resultado_teste(
                    document_id,
                    experimento,
                    chunks_json
                )
            )

            # -----------------------------------------------
            # RESUMO
            # -----------------------------------------------

            registro = {
                "document_id":
                    document_id,

                "test_id":
                    experimento[
                        "test_id"
                    ],

                "strategy":
                    experimento[
                        "strategy"
                    ],

                "description":
                    experimento[
                        "description"
                    ],

                "chunk_size":
                    experimento[
                        "chunk_size"
                    ],

                "chunk_overlap":
                    experimento[
                        "chunk_overlap"
                    ],

                **estatisticas,

                "examples":
                    gerar_exemplos(
                        chunks
                    ),
            }

            resumo_geral.append(
                registro
            )

            # -----------------------------------------------
            # TERMINAL
            # -----------------------------------------------

            print(
                f"Chunks: "
                f"{estatisticas['num_chunks']}"
            )

            print(
                f"Tamanho médio: "
                f"{estatisticas['avg_chunk_size']}"
            )

            print(
                f"Mínimo: "
                f"{estatisticas['min_chunk_size']}"
            )

            print(
                f"Máximo: "
                f"{estatisticas['max_chunk_size']}"
            )

            print(
                f"Tokens médios: "
                f"{estatisticas['avg_tokens']}"
            )

            print(
                f"Overlap: "
                f"{estatisticas['overlap_percent']}%"
            )

            print(
                f"JSON: {caminho}"
            )

    # ========================================================
    # SALVAR SUMMARY
    # ========================================================

    caminho_resumo = (
        PASTA_RESULTS
        / "summary_chunking_preliminar.json"
    )

    caminho_resumo.write_text(
        json.dumps(
            {
                "documents":
                    DOCUMENTOS_TESTE,

                "num_documents":
                    len(
                        DOCUMENTOS_TESTE
                    ),

                "num_experiments":
                    len(
                        EXPERIMENTOS
                    ),

                "results":
                    resumo_geral,
            },
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    # ========================================================
    # RESUMO VISUAL
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "RESUMO DOS EXPERIMENTOS"
    )
    print("=" * 70)

    for document_id in DOCUMENTOS_TESTE:

        print(
            f"\nDOCUMENTO: "
            f"{document_id}"
        )

        print(
            "-" * 70
        )

        print(
            f"{'TESTE':<8}"
            f"{'ESTRATÉGIA':<25}"
            f"{'CHUNKS':<10}"
            f"{'MÉDIA':<12}"
            f"{'TOKENS':<12}"
        )

        for item in resumo_geral:

            if (
                item["document_id"]
                != document_id
            ):
                continue

            print(
                f"{item['test_id']:<8}"
                f"{item['strategy']:<25}"
                f"{item['num_chunks']:<10}"
                f"{item['avg_chunk_size']:<12}"
                f"{item['avg_tokens']:<12}"
            )

    print("\n")
    print(
        f"Resumo salvo em:\n"
        f"{caminho_resumo}"
    )

    print("\n")
    print("=" * 70)
    print(
        "ETAPA 2 FINALIZADA COM SUCESSO"
    )
    print("=" * 70)


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    executar_experimentos()